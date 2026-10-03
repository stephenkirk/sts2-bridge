"""PPO training on starter, Insatiable, or reconstructed library fights.

Actors sample training seeds; an asynchronous evaluator scores fixed held-out seeds.
The Insatiable recording is also evaluated; --train-on recorded reuses it for training.
Run artifacts and CLI examples are documented in agents/nn/README.md.
"""

import argparse
import hashlib
import json
import queue
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.multiprocessing as mp

from sts2bridge import CombatWorker

from . import fights as pools
from .encode import Vocab, collate, encode
from .model import Net
from .contracts import character
from .experiment import record as record_experiment, resolve as resolve_experiment
from .episode import (Scoring, fight_context, first, multi_pick, outcome, over,
                      play, send_start, start, step_rewards)
from .metrics import acceptance_summary, mean, summary

HERE = Path(__file__).resolve().parent


def warm_vocab(fights, n=150, seed=0, table=None):
    """Collect ids from random play-first episodes on every fight in the pool, recordings included, then freeze."""
    vocab, rng = Vocab(table), random.Random(seed)
    with CombatWorker() as w:
        for k in range(max(n, len(fights))):
            fight = fights[k % len(fights)]
            state = start(w, fight, None if k % 10 == 0 and fight.get("recording") else f"VOCAB-{k}")
            while True:
                encode(state, vocab)
                if over(state):
                    break
                legal = state["legal"] or [multi_pick(state)]
                plays = [a for a in legal if a["type"] != "end_turn"]
                state = w.step(rng.choice(plays if plays and rng.random() < 0.85 else legal))
    vocab.frozen = True
    return vocab


def transfer_weights(net, state):
    """Keep all learned embedding rows; only appended vocabulary rows are new."""
    state = dict(state)
    old = state["ids.weight"]
    if net.ids.weight.shape != old.shape:
        if net.ids.weight.shape[0] < old.shape[0] or net.ids.weight.shape[1] != old.shape[1]:
            raise ValueError("transfer can only grow the vocabulary")
        weight = net.ids.weight.detach().clone()
        weight[:old.shape[0]] = old.to(weight.device)
        state["ids.weight"] = weight
    net.load_state_dict(state)


# ---- processes ----

def actor(rank, cfg, vocab_table, weights_q, out_q, fights, train_on, reward, envs, turn_cost_hp):
    """Batch policy inference across `envs` workers and overlap their game steps.

    Episodes are tagged with their starting weight version; weights can change mid-episode.
    """
    scoring = Scoring(reward, turn_cost_hp)
    torch.set_num_threads(1)
    vocab = Vocab(vocab_table, frozen=True)
    net = Net(**cfg).eval()
    rng = random.Random(1000 + rank)
    version = -1
    workers = [CombatWorker() for _ in range(envs)]
    weights = [f.get("sample_weight", 1.0) for f in fights] if any("sample_weight" in f for f in fights) else None
    fresh = lambda: ((rng.choices(fights, weights=weights, k=1)[0] if weights else rng.choice(fights)),
                     None if train_on == "recorded" else f"TRAIN-{rank}-{rng.getrandbits(40):x}")
    try:
        while version < 0:
            version, sd = weights_q.get()
            if version == "stop":
                return
            net.load_state_dict(sd)
        playing = [fresh() for _ in workers]
        for w, (fight, seed) in zip(workers, playing):
            send_start(w, fight, seed)
        states = [w.receive() for w in workers]
        eps = [dict(version=version, steps=[], hps=[], turns=[], hp_lost=0, start=first(s)) for s in states]
        while True:
            try:
                while True:
                    v, sd = weights_q.get_nowait()
                    if v == "stop":
                        return
                    version = v
                    net.load_state_dict(sd)
            except queue.Empty:
                pass
            asking = [i for i, s in enumerate(states) if not over(s) and s["legal"]]
            if asking:
                encoded = [encode(states[i], vocab) for i in asking]
                with torch.no_grad():
                    logits, _ = net(collate(encoded))
                dist = torch.distributions.Categorical(logits=logits)
                picks = dist.sample()
                logps = dist.log_prob(picks)
            chosen = {i: (k, int(picks[k])) for k, i in enumerate(asking)}
            before_hps = [s["obs"]["player"]["hp"] for s in states]
            for i, (w, s) in enumerate(zip(workers, states)):
                if over(s):
                    res = dict(outcome(s, **eps[i]["start"], scoring=scoring), kind=playing[i][0]["kind"], fight=playing[i][0]["name"],
                               hp_lost=eps[i]["hp_lost"], **fight_context(playing[i][0]))
                    if eps[i]["steps"]:
                        res["rewards"] = step_rewards(eps[i]["hps"], res, eps[i]["start"]["max0"], eps[i]["turns"], scoring=scoring)
                    out_q.put((eps[i]["version"], eps[i]["steps"], res, rank, vocab.diagnostics()))
                    eps[i] = None
                    playing[i] = fresh()
                    send_start(w, *playing[i])
                elif i in chosen:
                    k, a = chosen[i]
                    toks, acts = encoded[k]
                    eps[i]["steps"].append((toks, acts, a, float(logps[k])))
                    eps[i]["hps"].append(s["obs"]["player"]["hp"])
                    eps[i]["turns"].append(s["obs"]["turn"])
                    w.send("step", action=s["legal"][a])
                else:
                    w.send("step", action=multi_pick(s))
            states = [w.receive() for w in workers]
            for i, s in enumerate(states):
                if eps[i] is None:
                    eps[i] = dict(version=version, steps=[], hps=[], turns=[], hp_lost=0, start=first(s))
                else:
                    eps[i]["hp_lost"] += max(0, before_hps[i] - s["obs"]["player"]["hp"])
    finally:
        for w in workers:
            w.close()


def evaluator(cfg, vocab_table, weights_q, out_q, fights, n_seeds, n_sampled, reward, turn_cost_hp, eval_prefix):
    scoring = Scoring(reward, turn_cost_hp)
    torch.set_num_threads(1)
    vocab = Vocab(vocab_table, frozen=True)
    net = Net(**cfg).eval()
    with CombatWorker() as w:
        evaluated_initial = False
        while True:
            item = weights_q.get()
            while evaluated_initial and not weights_q.empty():
                item = weights_q.get()
            version, sd, episode_count = item
            if version == "stop":
                return
            net.load_state_dict(sd)
            t0 = time.time()
            vocab = Vocab(vocab_table, frozen=True)  # diagnostics for this evaluation only
            cards = {}  # character -> card id -> [decisions playable, decisions played]
            held = summary([play(w, net, vocab, f, f"{eval_prefix}-{i}", greedy=True, record=False,
                                 cards=cards.setdefault(character(f), {}), scoring=scoring)[1]
                            for f in fights for i in range(n_seeds)], fights=True)
            ev = dict(version=version, episodes=episode_count, heldout=held)
            ev["heldout_cards"] = cards
            ev["heldout_vocab"] = vocab.diagnostics()
            recorded = [f for f in fights if f.get("recording")]
            if len(recorded) == 1:
                ev["recorded_greedy"] = play(w, net, vocab, recorded[0], None, greedy=True, record=False, scoring=scoring)[1]
                ev["recorded_sampled_win"] = mean(play(w, net, vocab, recorded[0], None, record=False, scoring=scoring)[1]["won"]
                                                  for _ in range(n_sampled))
            out_q.put(dict(ev, eval_s=round(time.time() - t0, 1)))
            evaluated_initial = True


def save_best(out, ev):
    """Bind the reported score to the exact weights the asynchronous evaluator used."""
    snapshot = torch.load(out / "snapshots" / f"it{ev['version']:05d}.pt", weights_only=False)
    torch.save(dict(snapshot, eval=ev), out / "best.pt")


# ---- learner ----

TOKEN_KEYS = ("kind", "ident", "sub", "nums", "varn", "varv", "mask")


def minibatches(lens, size, bucket):
    """Shuffle sample indices into batches, optionally grouping similar token counts.

    Noise in the length sort varies bucket membership between epochs.
    """
    n = len(lens)
    if not bucket:
        perm = torch.randperm(n)
        return [perm[j:j + size] for j in range(0, n, size)]
    order = torch.argsort(lens.float() + 4 * torch.rand(n))
    chunks = [order[j:j + size] for j in range(0, n, size)]
    return [chunks[i] for i in torch.randperm(len(chunks))]


def ppo_update(net, opt, episodes, args, device):
    """Compute per-episode GAE and run clipped PPO updates.

    Pad once per update; bucketed minibatches trim token and action padding to local maxima.
    """
    episodes = [(steps, res) for steps, res in episodes if steps]
    samples = [s for steps, _ in episodes for s in steps]
    cpu = collate([(t, a) for t, a, _, _ in samples])
    lens, alens = cpu["mask"].sum(1), cpu["act_mask"].sum(1)
    big = {k: v.to(device) for k, v in cpu.items()}
    chosen = torch.tensor([s[2] for s in samples], device=device)
    old = torch.tensor([s[3] for s in samples], device=device)
    n = len(samples)
    amp = (lambda: torch.autocast(device_type=device, dtype=torch.bfloat16)) if args.amp == "bf16" else nullcontext

    def take(idx):
        b = {k: v[idx.to(device)] for k, v in big.items()}
        if args.bucket:
            T, A = int(lens[idx].max()), int(alens[idx].max())
            b = {k: v[:, :T] if k in TOKEN_KEYS else v[:, :A] for k, v in b.items()}
        return b

    def forward(idx):
        with amp():
            logits, value = net(take(idx))
        return logits.float(), value.float()

    values = torch.zeros(n)
    with torch.no_grad():
        for idx in minibatches(lens, 2048, args.bucket):
            values[idx] = forward(idx)[1].cpu()
    values = values.tolist()
    advs, returns, k = [], [], 0
    for steps, res in episodes:
        v = values[k:k + len(steps)] + [0.0]
        gae, adv = 0.0, [0.0] * len(steps)
        for t in reversed(range(len(steps))):
            r = res["rewards"][t]
            gae = r + v[t + 1] - v[t] + args.lam * gae
            adv[t] = gae
        advs += adv
        returns += [a + b for a, b in zip(adv, v)]
        k += len(steps)
    advs_t, returns_t = torch.tensor(advs, device=device), torch.tensor(returns, device=device)
    advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
    stats = []
    for _ in range(args.epochs):
        for idx in minibatches(lens, args.minibatch, args.bucket):
            logits, value = forward(idx)
            idx = idx.to(device)
            dist = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(chosen[idx]) - old[idx])
            a = advs_t[idx]
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            vl = (value - returns_t[idx]).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pg + args.vf * vl - args.ent * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats.append((pg.item(), vl.item(), ent.item(), ((ratio - 1).abs() > args.clip).float().mean().item()))
    s = [mean(x) for x in zip(*stats)]
    return dict(samples=n, pg=s[0], vloss=s[1], entropy=s[2], clipfrac=s[3])


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pool", choices=["starter", "insatiable", "library"], default="starter")
    ap.add_argument("--pool-file", help="library curriculum manifest from agents.nn.library")
    ap.add_argument("--extend-vocab", action="store_true", help="collect training vocabulary while preserving checkpoint IDs")
    ap.add_argument("--train-on", choices=["seeds", "recorded"], default="seeds", help="recorded: the Insatiable's")
    ap.add_argument("--actors", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=240, help="episodes per update")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, help="EVAL-* seeds per fight; default 100 for the Insatiable, 10 per starter fight")
    ap.add_argument("--eval-sampled", type=int, default=20)
    ap.add_argument("--eval-prefix", default="EVAL", help="seed namespace; use a fresh prefix for a final comparison")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reward", choices=["hp", "damage", "survival"],
                    help="default hp for the starter pool, damage for the Insatiable")
    ap.add_argument("--max-lag", type=int, default=2, help="drop episodes played by weights older than this many updates")
    ap.add_argument("--envs", type=int, default=2, help="fights each actor plays at once, one worker each")
    ap.add_argument("--bucket", action=argparse.BooleanOptionalAction, default=True,
                    help="minibatches of similar length, padded only to their own")
    ap.add_argument("--amp", choices=["none", "bf16"], default="none", help="autocast the learner's passes")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name> from its ckpt.pt for another --hours")
    ap.add_argument("--init-from", type=Path, help="branch from checkpoint weights and vocabulary with a fresh optimizer and counters")
    ap.add_argument("--turn-cost-hp", type=float, default=0.05, help="HP-equivalent cost per advanced turn, for --reward hp")
    ap.add_argument("--experiment", help="built-in experiment name or JSON definition path")
    ap.add_argument("--dry-run", action="store_true", help="print resolved conditions without starting workers or writing artifacts")
    return ap


def main(argv=None):
    ap = parser()
    args, experiment = resolve_experiment(ap, sys.argv[1:] if argv is None else argv)
    if args.resume and args.init_from:
        ap.error("--resume and --init-from are mutually exclusive")
    if args.turn_cost_hp < 0:
        ap.error("--turn-cost-hp must be nonnegative")
    if args.pool == "library" and not args.pool_file:
        ap.error("--pool library requires --pool-file")
    if args.resume and args.extend_vocab:
        ap.error("vocabulary extension needs a new branch, not --resume")
    if args.init_from and args.pool == "library" and not args.extend_vocab:
        ap.error("a library branch needs --extend-vocab for its new cards and encounters")
    if args.train_on == "recorded" and args.pool != "insatiable":
        ap.error("--train-on recorded needs --pool insatiable")
    one = args.pool == "insatiable"
    args.reward = args.reward or ("damage" if one else "hp")
    args.eval_seeds = args.eval_seeds or (100 if one else 10)
    headline = "win" if one else "reward"  # what best.pt keeps the best of
    scoring = dict(reward=args.reward, turn_cost_hp=args.turn_cost_hp if args.reward == "hp" else 0.0,
                   eval_prefix=args.eval_prefix, eval_seeds=args.eval_seeds)
    if experiment:
        experiment["resolved"].update(reward=args.reward, eval_seeds=args.eval_seeds)
        experiment["selection_metric"] = headline
    if args.dry_run:
        print(json.dumps(experiment or dict(resolved=vars(args)), indent=2, default=str))
        return
    fights = pools.pool(args.pool, pool_file=args.pool_file)
    eval_fights = pools.pool(args.pool, pool_file=args.pool_file, split="validation") if args.pool == "library" else fights
    pool_metadata = {}
    if args.pool == "library":
        pool_bytes = Path(args.pool_file).read_bytes()
        scoring["pool_digest"] = hashlib.sha256(pool_bytes).hexdigest()

    out = HERE / "runs" / args.name
    resumed = None
    if args.resume:
        # The vocabulary is the checkpoint's: ids are numbered as the weights learned them.
        resumed = torch.load(out / "ckpt.pt", weights_only=False)
        if resumed.get("pool") != args.pool:
            ap.error(f"{out}/ckpt.pt trained on --pool {resumed.get('pool')}")
        previous_scoring = resumed.get("scoring")
        if previous_scoring is None:
            # Legacy runs keep their configuration in the log rather than in the checkpoint.
            configs = [json.loads(line)["args"] for line in (out / "log.jsonl").read_text().splitlines()
                       if "args" in json.loads(line)]
            if configs:
                old_args = configs[-1]
                old_reward = old_args.get("reward") or ("damage" if one else "hp")
                previous_scoring = dict(reward=old_reward,
                                        turn_cost_hp=old_args.get("turn_cost_hp", 0.0) if old_reward == "hp" else 0.0,
                                        eval_prefix=old_args.get("eval_prefix", "EVAL"),
                                        eval_seeds=old_args.get("eval_seeds") or (100 if one else 10))
        if previous_scoring is None or previous_scoring != scoring:
            ap.error("resume needs matching saved reward/evaluation settings; use --init-from to change scoring")
        if "opt" not in resumed:
            print("ckpt.pt has no optimizer state; Adam starts fresh", flush=True)
    elif (out / "log.jsonl").exists() or (out / "ckpt.pt").exists():
        ap.error(f"{out} already has a run; --resume it or pick another --name")
    initial = resumed
    if args.init_from:
        initial = torch.load(args.init_from, weights_only=False)
        if initial.get("pool") != args.pool and not args.extend_vocab:
            ap.error(f"{args.init_from} trained on --pool {initial.get('pool')}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        experiment = record_experiment(out, experiment, args.resume)
    except ValueError as exc:
        ap.error(str(exc))
    if experiment:
        experiment["resolved"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        experiment["selection_metric"] = headline
        pool_metadata["experiment"] = dict(id=experiment["definition"]["id"], sha256=experiment["sha256"])
    log = open(out / "log.jsonl", "a")
    log.write(json.dumps({"args": dict(vars(args), init_from=str(args.init_from) if args.init_from else None),
                          "experiment": experiment,
                          "resumed_at": resumed and resumed["it"],
                          "initialization": dict(mode="resume" if resumed else "branch" if initial else "fresh",
                                                 checkpoint=str(args.init_from.resolve()) if args.init_from else
                                                 str((out / "ckpt.pt").resolve()) if resumed else None,
                                                 iteration=initial.get("it") if initial else None,
                                                 episodes=initial.get("episodes") if initial else None,
                                                 optimizer="restored" if resumed and "opt" in resumed else "fresh")}) + "\n")
    log.flush()
    (out / "fights.json").write_text(json.dumps(fights, indent=1) + "\n")
    if args.pool == "library":
        (out / "pool.json").write_bytes(pool_bytes)
        (out / "eval_fights.json").write_text(json.dumps(eval_fights, indent=1) + "\n")
        pool_metadata.update(pool_file=str((out / "pool.json").resolve()), pool_digest=scoring["pool_digest"])
    torch.manual_seed(0 if resumed is None else resumed["it"])
    torch.set_num_threads(args.threads)

    t0 = time.time()
    warm_episodes = max(150, len(fights) * (3 if args.pool == "library" else 1))
    if initial:
        vocab, cfg = Vocab(initial["vocab"], frozen=True), initial["cfg"]
        if args.extend_vocab:
            vocab = warm_vocab(fights, n=warm_episodes, table=initial["vocab"])
            cfg = dict(cfg, vocab_size=max(cfg["vocab_size"], len(vocab) + 8))
            print(f"vocabulary extended: {len(initial['vocab'])} -> {len(vocab.table)} IDs", flush=True)
            log.write(json.dumps({"vocab_added": sorted(set(vocab.table) - set(initial["vocab"])),
                                  "vocab_inherited": len(initial["vocab"])}) + "\n")
            log.flush()
    else:
        vocab = warm_vocab(fights, n=warm_episodes)
        print(f"vocab {len(vocab)} ids in {time.time() - t0:.0f}s", flush=True)
        cfg = dict(vocab_size=len(vocab) + 8, d=args.d, layers=args.layers)
    net = Net(**cfg).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if initial:
        transfer_weights(net, initial["state"])
    if resumed:
        if "opt" in resumed:
            opt.load_state_dict(resumed["opt"])
    print(f"{sum(p.numel() for p in net.parameters()):,} parameters", flush=True)

    ctx = mp.get_context("spawn")
    out_q, eval_out = ctx.Queue(), ctx.Queue()
    weight_qs = [ctx.Queue() for _ in range(args.actors)]
    procs = [ctx.Process(target=actor, args=(r, cfg, vocab.table, weight_qs[r], out_q, fights, args.train_on, args.reward, args.envs,
                                           args.turn_cost_hp), daemon=True)
             for r in range(args.actors)]
    eval_q = ctx.Queue()
    procs.append(ctx.Process(target=evaluator, args=(cfg, vocab.table, eval_q, eval_out, eval_fights, args.eval_seeds,
                                                     args.eval_sampled, args.reward, args.turn_cost_hp, args.eval_prefix), daemon=True))
    for p in procs:
        p.start()

    def publish(version):
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        for q in weight_qs:
            q.put((version, sd))
        return sd

    it, total_eps, best_heldout, before = 0, 0, -float("inf"), 0.0
    if resumed:
        it, total_eps = resumed["it"], resumed.get("episodes", 0)
        best_heldout, before = resumed.get("best", best_heldout), resumed.get("minutes", 0.0)
    deadline = time.time() + args.hours * 3600
    actor_vocab = {}
    sd = publish(it)
    (out / "snapshots").mkdir(exist_ok=True)
    torch.save(dict(cfg=cfg, vocab=vocab.table, pool=args.pool, state=sd, it=it, episodes=total_eps, scoring=scoring, **pool_metadata),
               out / "snapshots" / f"it{it:05d}.pt")
    eval_q.put((it, sd, total_eps))
    while time.time() < deadline:
        t_it = time.time()
        episodes, lags, unknown, dropped = [], [], 0, 0
        acceptance = []
        while len(episodes) < args.episodes:
            version, steps, res, rank, diagnostics = out_q.get()
            actor_vocab[rank] = diagnostics
            lags.append(it - version)
            acceptance.append((it - version <= args.max_lag, res))
            if it - version <= args.max_lag:
                episodes.append((steps, res))
            else:
                dropped += 1
        t_collect = time.time() - t_it
        stats = ppo_update(net, opt, episodes, args, args.device)
        it += 1
        total_eps += len(episodes)
        sd = publish(it)
        train = summary([r for _, r in episodes])
        missing = Counter()
        for diagnostics in actor_vocab.values():
            missing.update(diagnostics["counts"])
        unknown = sum(missing.values())
        row = dict(it=it, episodes=total_eps, min=round(before + (time.time() - t0) / 60, 1),
                   train_win=train["win"], train_reward=train["reward"], train_kept=train["kept"],
                   train_boss=train["boss"], train_turn=train["turn"], train_by_char=train["by_char"],
                   collect_s=round(t_collect, 1), update_s=round(time.time() - t_it - t_collect, 1),
                   lag=mean(lags), dropped=dropped, unknown_ids=unknown, **stats)
        row["unknown_id_counts"] = dict(missing.most_common())
        row["id_lookups"] = sum(d["lookups"] for d in actor_vocab.values())
        row["acceptance"] = acceptance_summary(acceptance)
        if "by_pool" in train:
            row["train_by_pool"] = train["by_pool"]
        if it % args.eval_every == 0:
            eval_q.put((it, sd, total_eps))
            # Every evaluated policy is kept, so shift.py can show how play changed across training.
            (out / "snapshots").mkdir(exist_ok=True)
            torch.save(dict(cfg=cfg, vocab=vocab.table, pool=args.pool, state=sd, it=it, episodes=total_eps, scoring=scoring, **pool_metadata),
                       out / "snapshots" / f"it{it:05d}.pt")
        try:
            while True:
                ev = eval_out.get_nowait()
                row["eval"] = ev
                if ev["heldout"][headline] >= best_heldout:
                    best_heldout = ev["heldout"][headline]
                    save_best(out, ev)
        except queue.Empty:
            pass
        log.write(json.dumps(row) + "\n"); log.flush()
        ev = row.get("eval")
        cells = lambda d: " ".join(f"{k[:4]} {v['win']:.2f}/{v['kept']:.2f}" for k, v in d.items())
        print(f"it {it:4d} eps {total_eps:7d} {row['min']:6.1f}m | train reward {row['train_reward']:+.3f} "
              f"win/kept {cells(train['by_char'])} turn {row['train_turn']:4.1f} | ent {stats['entropy']:.2f} "
              f"v {stats['vloss']:.3f} | {row['collect_s']}s+{row['update_s']}s", flush=True)
        if ev:
            rec = ev.get("recorded_greedy")
            print(f"  EVAL@{ev['version']} heldout reward {ev['heldout']['reward']:+.3f} win {ev['heldout']['win']:.2f} "
                  f"boss {ev['heldout']['boss']:.1f} kept {ev['heldout']['kept']:.2f} | {cells(ev['heldout']['by_char'])}"
                  + (f" | recorded {'WIN' if rec['won'] else 'loss'} (hp {rec['hp']}, boss {rec['boss']}) "
                     f"sampled {ev['recorded_sampled_win']:.2f}" if rec else ""), flush=True)
        torch.save(dict(cfg=cfg, vocab=vocab.table, pool=args.pool, state={k: v.cpu() for k, v in net.state_dict().items()},
                        opt=opt.state_dict(), it=it, episodes=total_eps, best=best_heldout, minutes=row["min"], scoring=scoring, **pool_metadata),
                   out / "ckpt.pt")

    for q in weight_qs:
        q.put(("stop", None))
    eval_q.put(("stop", None, None))
    for p in procs:
        p.join(timeout=10)


if __name__ == "__main__":
    main()
