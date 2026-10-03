"""Play one continuous native run to its end: fights by rollout search on snapshot copies, the rest by priors.

    python3 -m agents.run.play_run --seed WIN1 [--character CHARACTER.IRONCLAD] [--sims 10] [--priors priors.json]

One main worker holds the run (``start_run`` / ``run_step``); it is never rebuilt. At every combat the main worker
writes the game's own recording of the fight (``run_combat_snapshot``), whose initial state is the run as it entered
the room. A pool of sim workers ``load`` that file and play lines on the copy. The main run then plays the chosen
line, and every step's state hash must equal the hash the copy produced for the same step, or the runner stops.

The search is clairvoyant: a copy draws the same cards and rolls the same intents as the run will, so it sees what a
player cannot. That is fine for driving the harness through a whole run. It is not a policy to learn from.

Search, per fight (``search_line``): a turn-level beam search at several HP weights finds a first line (``beam_line``),
then a local search keeps the best line (the incumbent) and rolls out variations of it, each keeping the incumbent up
to a random cut and continuing with a randomised policy. Once a line wins, the search goes on until a patience budget
passes without a better line, then the main run plays it. Fights the snapshot cannot re-enter (a fight started from inside an event) fall back to the rollout policy.

Everything outside combat follows a preferences file (``priors``): a card is taken when its score beats the act's
skip score, and relics, Ancients, events and removals go to the best-rated option.
"""

import argparse
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sts2bridge import CombatWorker, WorkerError

from .priors import Priors

HERE = Path(__file__).resolve().parent
BASICS = ("CARD.STRIKE_", "CARD.DEFEND_")


# ---- combat -------------------------------------------------------------------------------------------------------

def incoming_damage(obs):
    return sum((i.get("damage") or 0) * (i.get("hits") or 1)
               for e in obs["enemies"] if e["alive"] and e["intent"] for i in e["intent"]["intents"])


def play_score(card, target, obs, threat):
    """Rough worth of one play from the numbers on the card: damage that lands, block that is needed."""
    v = card.get("vars") or {}
    alive = [e for e in obs["enemies"] if e["alive"]]
    dmg = (v.get("Damage") or v.get("CalculatedDamage") or 0) * max(1, v.get("Repeat") or 1)
    blk = v.get("Block") or v.get("CalculatedBlock") or 0
    score = 0.0
    if dmg:
        hit = [target] if target is not None else alive if card["target"] in ("AllEnemies", "RandomEnemy") else []
        for e in hit:
            need = e["hp"] + e["block"]
            score += min(dmg, need)
            if dmg >= need:
                score += 10 + incoming_damage({"enemies": [e]})
    if blk:
        score += 1.3 * min(blk, max(0, threat))
    if card["type"] == "Power":
        score += 14 if obs["turn"] <= 3 else 6
    if not dmg and not blk and card["type"] != "Power":
        score += 5
    if card["type"] in ("Status", "Curse"):
        score = 1.0
    cost = card["cost"] if card["cost"] is not None else obs["player"]["energy"] or 0
    return score / (1 + 0.4 * cost)


def rollout_action(c, rng, style):
    """One randomised input at combat boundary ``c``. ``style`` varies per rollout so a batch spreads out."""
    if c["boundary"] == "awaiting_input" and style["scored"]:
        obs = c["obs"]
        threat = incoming_damage(obs) - obs["player"]["block"]
        enemies = {e["slot"]: e for e in obs["enemies"]}
        options, weights = [], []
        for a in c["legal"]:
            if a["type"] == "play":
                score = play_score(obs["hand"][a["hand"]], enemies.get(a["target"]), obs, threat)
            elif a["type"] == "potion":
                score = style["potion_score"]
            else:
                score = style["end_score"]
            options.append(a)
            weights.append(math.exp(min(50.0, score / style["temp"])))
        return rng.choices(options, weights)[0]
    if c["boundary"] == "awaiting_choice":
        if c["legal"]:
            return rng.choice(c["legal"])
        ch = c["choice"]
        k = rng.randint(ch["min"], min(ch["max"], len(ch["options"])))
        return {"type": "choose", "picks": sorted(rng.sample(range(len(ch["options"])), k))}
    obs = c["obs"]
    hand = obs["hand"]
    threat = incoming_damage(obs) - obs["player"]["block"]
    alive = [e for e in obs["enemies"] if e["alive"]]
    weakest = min(alive, key=lambda e: e["hp"] + e["block"])["slot"] if alive else None
    options, weights = [], []
    for a in c["legal"]:
        if a["type"] == "play":
            card = hand[a["hand"]]
            w = 1.0
            if card["type"] == "Power":
                w = 3.0 if obs["turn"] <= 2 else 1.5
            elif card["type"] == "Attack":
                w = style["attack"] * (2.0 if threat <= 0 else 1.0)
            elif card["type"] == "Skill":
                w = style["skill"] * (2.0 if threat > 0 else 0.7)
            elif card["type"] in ("Status", "Curse"):
                w = 0.3
            if card["cost"] == 0:
                w *= 1.5
            if a["target"] is not None and a["target"] == weakest:
                w *= style["focus"]
        elif a["type"] == "potion":
            w = style["potion"]
        else:
            w = style["end"] if len(c["legal"]) > 1 else 1.0
        options.append(a)
        weights.append(w)
    return rng.choices(options, weights)[0]


def random_style(rng, scored=0.75):
    return {"attack": rng.uniform(0.6, 2.0), "skill": rng.uniform(0.6, 2.0), "focus": rng.uniform(1.0, 4.0),
            "potion": rng.choice((0.0, 0.02, 0.1, 0.4)), "end": rng.uniform(0.02, 0.25),
            # Most rollouts score plays by the card's numbers; the rest keep the type-weighted policy for spread.
            "scored": rng.random() < scored, "temp": rng.uniform(1.5, 8.0),
            "potion_score": rng.choice((-50.0, 0.0, 5.0, 15.0)), "end_score": rng.uniform(-2.0, 3.0)}


def value(c, potion_worth):
    """Terminal value of a rolled-out fight: winning dominates, then HP kept, then potions kept."""
    obs = c["obs"]
    p = obs["player"]
    potions = sum(1 for x in p["potions"] if x)
    if c["boundary"] == "terminal" and p["hp"] > 0:
        return 10_000 + p["hp"] + 0.5 * p["max_hp"] + potion_worth * potions
    total = sum(e["max_hp"] for e in obs["enemies"]) or 1
    left = sum(e["hp"] for e in obs["enemies"] if e["alive"])
    return -1000 * left / total + (obs["turn"] or 0)


class SimPool:
    """Sim workers, one per thread. Each rollout re-enters the snapshot and replays the committed prefix."""

    def __init__(self, n):
        self.workers = [CombatWorker() for _ in range(n)]
        self._free = list(self.workers)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(n)

    def _with_worker(self, fn):
        with self._lock:
            w = self._free.pop()
        try:
            return fn(w)
        finally:
            with self._lock:
                self._free.append(w)

    def map(self, fn, items):
        return list(self._pool.map(lambda item: self._with_worker(lambda w: fn(w, item)), items))

    def close(self):
        self._pool.shutdown()
        for w in self.workers:
            w.close()


MAX_FIGHT_STEPS = 400
MAX_TURN_STEPS = 40


def rollout(w, mcr, keep, seed, potion_worth, scored=0.75):
    """Re-enter the fight, replay ``keep`` (a line's opening inputs), then continue with the rollout policy."""
    rng = random.Random(seed)
    style = random_style(rng, scored)
    c = w.load(mcr)
    actions, hashes, turn_steps = [], [], 0
    for a in keep:
        if c["boundary"] == "terminal":
            break
        c = w.step(a)
        actions.append(a)
        hashes.append(c["state_hash"])
        turn_steps = 0 if a["type"] == "end_turn" else turn_steps + 1
    while c["boundary"] != "terminal" and len(actions) < MAX_FIGHT_STEPS:
        a = rollout_action(c, rng, style)
        if c["boundary"] == "awaiting_input" and turn_steps >= MAX_TURN_STEPS:
            a = {"type": "end_turn"}
        turn_steps = 0 if a["type"] == "end_turn" else turn_steps + 1
        c = w.step(a)
        actions.append(a)
        hashes.append(c["state_hash"])
    return value(c, potion_worth), actions, hashes


DEBUFFS = {"POWER.WEAK_POWER", "POWER.VULNERABLE_POWER", "POWER.FRAIL_POWER", "POWER.POISON_POWER",
           "POWER.CONSTRICT_POWER", "POWER.ENTANGLED_POWER", "POWER.NO_DRAW_POWER", "POWER.SHRINK_POWER"}


def state_score(c, hp_weight, potion_worth):
    """Mid-fight worth of a turn-start state, for the beam: HP kept against enemy HP left, plus lasting buffs."""
    if c["boundary"] == "terminal":
        v = value(c, potion_worth)
        return v if v >= 10_000 else -100_000 + v
    o = c["obs"]
    p = o["player"]
    alive = [e for e in o["enemies"] if e["alive"]]
    score = hp_weight * p["hp"] - sum(e["hp"] + e["block"] for e in alive) - 8 * len(alive)
    for pw in p["powers"]:
        score += (-2 if pw["id"] in DEBUFFS else 1.5) * min(abs(pw["amount"]), 10)
    for e in alive:
        for pw in e["powers"]:
            if pw["id"] == "POWER.STRENGTH_POWER":
                score -= 2 * pw["amount"]
            elif pw["id"] in DEBUFFS:
                score += 1.0 * min(pw["amount"], 5)
    score += potion_worth * sum(1 for x in p["potions"] if x)
    return score


def play_turn(w, mcr, prefix, seed, scored):
    """Re-enter the fight, replay ``prefix``, then play one turn with the rollout policy, through the enemy turn."""
    rng = random.Random(seed)
    style = random_style(rng, scored)
    c = w.load(mcr)
    for a in prefix:
        c = w.step(a)
    actions, hashes, steps = [], [], 0
    while c["boundary"] != "terminal":
        a = rollout_action(c, rng, style)
        if c["boundary"] == "awaiting_input" and steps >= MAX_TURN_STEPS:
            a = {"type": "end_turn"}
        c = w.step(a)
        actions.append(a)
        hashes.append(c["state_hash"])
        steps += 1
        if a["type"] == "end_turn":
            break
    return actions, hashes, c


def beam_line(mcr, pool, kind, budget, hp_weight, rng_seed=""):
    """Turn-level beam search: expand each kept state with sampled one-turn plans, score the state after the enemy
    turn, keep the best distinct states. Returns the best finished line as (value, actions, hashes) and the cost."""
    potion_worth = {"boss": 2, "elite": 5, "monster": 10}[kind]
    beam = [([], [])]  # (actions, hashes) from the fight's start, each ending at a turn start
    done, spent, seed = [], 0, 0
    for _turn in range(budget["beam_turns"]):
        jobs = [(node, seed + i * 1000 + k) for i, node in enumerate(beam) for k in range(budget["beam_children"])]
        seed += 1_000_000
        results = pool.map(lambda w, job: (job[0], play_turn(w, mcr, job[0][0], job[1], budget.get("scored", 1.0))), jobs)
        spent += len(results)
        children = {}
        for (actions, hashes), (more, more_hashes, c) in results:
            line = (actions + more, hashes + more_hashes)
            if c["boundary"] == "terminal":
                v = value(c, potion_worth)
                done.append((v, *line))
                continue
            key = c["state_hash"]
            score = state_score(c, hp_weight, potion_worth)
            if key not in children or score > children[key][0]:
                children[key] = (score, line)
        if not children:
            break
        beam = [line for _, line in sorted(children.values(), key=lambda x: -x[0])[:budget["beam_width"]]]
        if done and max(d[0] for d in done) >= 10_000 and len(done) >= budget["beam_width"]:
            break
    if not done:
        return None, spent
    return max(done, key=lambda d: d[0]), spent


def search_line(mcr, pool, kind, budget, rng_seed=""):
    """The best line found for the fight in ``mcr``: (value, actions, state hashes), with the rollouts spent.

    The copy is deterministic, so this is single-player planning: keep the best line found (the incumbent) and try
    variations of it. Each rollout keeps the incumbent up to a random cut and continues from there with the rollout
    policy; a share restart from the fight's start. A line replaces the incumbent only if it ends better. Once a line
    wins, the search goes on until a patience budget passes without a better one.
    """
    potion_worth = {"boss": 2, "elite": 5, "monster": 10}[kind]
    rng = random.Random(rng_seed)
    best, rollouts, seed, since_better = None, 0, 0, 0

    def beams():
        nonlocal best, rollouts, beam_spent
        for hp_weight in budget["beam_hp_weights"][kind]:
            line, spent = beam_line(mcr, pool, kind, budget, hp_weight, rng_seed)
            rollouts += spent
            beam_spent += spent
            if line and (best is None or line[0] > best[0]):
                best = line

    # Beam lines at several HP weights first; the mutation search below refines the best one.
    beam_spent = 0
    beams()
    while rollouts - beam_spent < budget["max_rollouts"]:
        if best is None:
            keeps = [[] for _ in range(budget["per_round"])]
        else:
            line = best[1]
            keeps = [[] if rng.random() < 0.15 else line[:rng.randrange(len(line))] for _ in range(budget["per_round"])]
        batch = pool.map(lambda w, job: rollout(w, mcr, job[1], job[0], potion_worth, budget.get("scored", 1.0)),
                         list(zip(range(seed, seed + len(keeps)), keeps)))
        seed += len(keeps)
        rollouts += len(batch)
        since_better += len(batch)
        for cand in batch:
            if best is None or cand[0] > best[0]:
                best, since_better = cand, 0
        if best[0] >= 10_000 and since_better >= budget["patience"][kind]:
            break
    return best, rollouts


def fight_kind(encounter):
    return "boss" if "BOSS" in encounter else "elite" if "ELITE" in encounter else "monster"


def search_fight(main, state, pool, log, budget):
    """Find a line for the current run combat on snapshot copies (``search_line``), then play it through ``main``,
    checking every step's state hash against the copy's."""
    mcr = main.workdir / "fight.mcr"
    main.run_combat_snapshot(mcr)
    enc = state["combat"]["obs"]["encounter"] or ""
    t0 = time.time()
    best, rollouts = search_line(mcr, pool, fight_kind(enc), budget, enc)
    _, actions, hashes = best
    for i, (a, h) in enumerate(zip(actions, hashes)):
        state = main.run_step(a)
        got = state["combat"]["state_hash"] if state.get("combat") else None
        if got is not None and got != h:
            raise RuntimeError(f"run and snapshot diverged at combat step {i + 1}: {got} != {h}")
        if state["boundary"] != "combat":
            break
    if state["boundary"] == "combat":
        raise RuntimeError("the searched line ended but the run's combat did not")
    log(f"    fight {enc}: {len(actions)} inputs, {rollouts} rollouts, {time.time() - t0:.1f}s, best value {best[0]:.0f}")
    return state


def play_fight_unsearched(main, state, log):
    rng, style = random.Random(0), random_style(random.Random(0), scored=1.0)
    style.update(end=0.01, potion=0.05)
    n = 0
    while state["boundary"] == "combat":
        c = state["combat"] or main.run_observe()["combat"]
        state = main.run_step(rollout_action(c, rng, style))
        n += 1
    log(f"    fight (no snapshot, rollout policy): {n} inputs")
    return state


# ---- everything else ----------------------------------------------------------------------------------------------

class Chooser:
    def __init__(self, priors, log):
        self.p = priors
        self.log = log
        self.choice_context = None  # what the next card choice is for: upgrade, remove, or None (unknown)

    def card_worth(self, card_id, upgrades=0):
        if card_id.startswith(BASICS):
            return 1000.0
        if card_id == "CARD.BASH":
            return 1300.0
        return self.p.card(card_id, upgrades)

    def map(self, s):
        o = s["obs"]
        hp = o["hp"] / o["max_hp"]
        pri = {"Treasure": 5, "Monster": 3, "Unknown": 2.5, "Shop": 1 + (o["gold"] >= 150) * 2.5,
               "RestSite": 1 + (hp < 0.6) * 5, "Elite": 4 if hp > 0.7 else 0, "Boss": 9, "Ancient": 9}
        return max(s["legal"], key=lambda a: pri.get(a["point_type"], 1))

    def event(self, s):
        def score(a):
            key = a["key"]
            option = key.rsplit(".", 1)[-1]
            if option in self.p.ancient:
                return 100 * self.p.ancient[option]
            return self.p.event.get(key, 0) + (0 if a["proceed"] else 0.1)
        return max(s["legal"], key=score)

    def rewards(self, s):
        legal = s["legal"]
        by_type = {}
        for a in legal:
            if a["type"] == "take_reward":
                by_type.setdefault(s["rewards"][a["index"]]["type"], []).append(a)
        for t in ("RelicReward", "GoldReward", "PotionReward"):
            if t in by_type:
                return by_type[t][0]
        if "CardReward" in by_type:
            act = s["obs"]["act"]
            best = max(by_type["CardReward"], key=lambda a: self.p.card(s["rewards"][a["index"]]["cards"][a["card"]]))
            card = s["rewards"][best["index"]]["cards"][best["card"]]
            if self.p.card(card) > self.p.skip(act):
                return best
            # Nothing worth taking: a reroll (Driftwood) or a sacrifice (Pael's Wing) beats a plain skip.
            for alt in ("REROLL", "SACRIFICE"):
                offer = next((a for a in legal if a.get("alternative") == alt), None)
                if offer:
                    return offer
        if "CardRemovalReward" in by_type:
            self.choice_context = "remove"
            return by_type["CardRemovalReward"][0]
        other = [a for a in legal if a["type"] == "take_reward"
                 and s["rewards"][a["index"]]["type"] not in ("CardReward",)]
        if other:
            return other[0]
        return next((a for a in legal if a["type"] == "skip_rewards"), legal[0])

    def rest(self, s):
        o = s["obs"]
        ids = {a["id"]: a for a in s["legal"]}
        # The rest site before a boss is the act's last room with a map choice after it: heal unless nearly full.
        before_boss = o.get("next_point_types") == ["Boss"]
        if (o["hp"] < (0.9 if before_boss else 0.6) * o["max_hp"] or "SMITH" not in ids) and "HEAL" in ids:
            return ids["HEAL"]
        if "SMITH" in ids:
            self.choice_context = "upgrade"
            return ids["SMITH"]
        return s["legal"][0]

    def shop(self, s, bought):
        o = s["obs"]
        entries = {e["index"]: e for e in s["shop"]}
        buys = [a for a in s["legal"] if a["type"] == "buy"]
        removal = [a for a in buys if a["item_type"] == "MerchantCardRemovalEntry"]
        if removal and any(c.startswith(BASICS) for c in o["deck"]):
            self.choice_context = "remove"
            return removal[0]

        def worth(a):
            e = entries[a["index"]]
            if a["item_type"] == "MerchantCardEntry":
                return self.p.card(e["id"]) - self.p.skip(o["act"]) - 150
            if a["item_type"] == "MerchantRelicEntry":
                return 400 * self.p.relic.get(e["id"], 0.3)
            if a["item_type"] == "MerchantPotionEntry" and sum(1 for _ in o["potions"]) < 2:
                return 20
            return -1
        good = [a for a in buys if worth(a) > 0 and a["index"] not in bought]
        if good:
            return max(good, key=worth)
        return {"type": "leave_shop"}

    def card_choice(self, s, choice):
        """A card choice outside combat: upgrade, remove, transform, obtain... The game does not say which."""
        opts = choice["options"]
        deck = set(s["obs"]["deck"])
        context = self.choice_context
        if context is None and not any(o["id"] in deck for o in opts):
            context = "obtain"
        if context in ("upgrade", "obtain"):
            order = sorted(range(len(opts)), key=lambda i: self.card_worth(opts[i]["id"], opts[i]["upgrades"]), reverse=True)
        else:
            def badness(i):
                cid = opts[i]["id"]
                if opts[i]["type"] in ("Curse", "Status"):
                    return -10_000
                return self.card_worth(cid, opts[i]["upgrades"]) - 3 * self.p.remove.get(cid, 0)
            order = sorted(range(len(opts)), key=badness)
        n = min(max(choice["min"], 1), choice["max"], len(opts))
        self.choice_context = None
        return {"type": "choose", "picks": sorted(order[:n])}

    def bundle(self, s):
        bundles = s["bundle_choice"]["bundles"]
        best = max(range(len(bundles)), key=lambda i: sum(self.p.card(c) for c in bundles[i]["cards"]) / max(1, len(bundles[i]["cards"])))
        return {"type": "bundle", "index": best}

    def treasure_relic(self, s):
        picks = [a for a in s["legal"] if a["index"] is not None]
        if not picks:
            return s["legal"][0]
        return max(picks, key=lambda a: self.p.relic.get(a["id"], 0.3))


# ---- the run ------------------------------------------------------------------------------------------------------

def play(seed, character="CHARACTER.IRONCLAD", ascension=10, sims=10, priors=None, budget=None, out=None,
         verbose=True):
    """``sims=0`` plays every fight with the rollout policy alone: a fast sweep of the run adapters."""
    budget = budget or {"per_round": 40, "max_rollouts": 30_000,
                        "patience": {"monster": 400, "elite": 1500, "boss": 4000},
                        "beam_width": 40, "beam_children": 24, "beam_turns": 25,
                        "beam_hp_weights": {"monster": (1.0, 2.5), "elite": (0.6, 1.0, 2.5),
                                            "boss": (0.4, 0.6, 1.0, 1.5, 2.5)}}
    lines = []
    live = open(Path(out).with_suffix(".log"), "w") if out else None

    def log(msg):
        lines.append(msg)
        if verbose:
            print(msg, flush=True)
        if live:
            live.write(msg + "\n")
            live.flush()

    priors = Priors(character, priors)
    chooser = Chooser(priors, log)
    pool = SimPool(sims) if sims else None
    main = CombatWorker()
    t0 = time.time()
    result = {"seed": seed, "character": character, "ascension": ascension}
    try:
        state = main.start_run(character, seed, ascension=ascension, unlocks="all")
        identity = state["run_state_identity"]
        floor, prev_boundary, bought, repeats, last_seen = None, None, set(), 0, None
        for _ in range(5000):
            b = state["boundary"]
            o = state["obs"]
            if state["run_state_identity"] != identity:
                raise RuntimeError("run state was rebuilt")
            if (o["act"], o["floor"]) != floor:
                floor = (o["act"], o["floor"])
                bought = set()
                log(f"act {o['act'] + 1} floor {o['floor']:2d} {o['room'] or '-':14s} hp {o['hp']}/{o['max_hp']} "
                    f"gold {o['gold']} deck {len(o['deck'])} relics {len(o['relics'])} [{b}]")
            if b == "terminal":
                break
            if b == "combat":
                if pool is None or prev_boundary == "event" or state["combat"] is None:
                    state = play_fight_unsearched(main, state, log)
                else:
                    try:
                        state = search_fight(main, state, pool, log, budget)
                    except WorkerError as e:
                        if "combat" not in main.run_observe()["boundary"] or state["combat"]["obs"]["turn"] != 1:
                            raise
                        log(f"    snapshot unusable ({e}); rollout policy instead")
                        state = play_fight_unsearched(main, state, log)
                prev_boundary = b
                continue
            if b == "awaiting_map":
                a = chooser.map(state)
            elif b == "event":
                a = chooser.event(state)
            elif b == "awaiting_rewards":
                a = chooser.rewards(state)
            elif b == "awaiting_proceed":
                a = {"type": "proceed"}
            elif b == "rest":
                a = chooser.rest(state)
            elif b == "shop":
                a = chooser.shop(state, bought)
                if a["type"] == "buy":
                    bought.add(a["index"])
            elif b == "treasure":
                a = {"type": "open_chest"}
            elif b == "treasure_relic":
                a = chooser.treasure_relic(state)
            elif b == "awaiting_bundle":
                a = chooser.bundle(state)
            elif b == "awaiting_room_choice":
                a = chooser.card_choice(state, state["choice"])
            else:
                raise RuntimeError(f"no decision rule for boundary {b}")
            if verbose and b not in ("awaiting_map", "awaiting_proceed"):
                log(f"    {b}: {describe(a, state)}")
            prev_boundary = b
            seen = json.dumps([b, state["legal"], state["obs"]["hp"], state["obs"]["gold"], state["obs"]["deck"]])
            repeats = repeats + 1 if seen == last_seen else 0
            last_seen = seen
            if repeats >= 3:
                raise RuntimeError(f"no progress at {b} after {a}")
            state = main.run_step(a)
        o = state["obs"]
        result.update(boundary=state["boundary"], act=o["act"], floor=o["floor"], hp=o["hp"], max_hp=o["max_hp"],
                      deck=o["deck"], relics=[f"{r['id']['Category']}.{r['id']['Entry']}" for r in o["relics"]],
                      victory=state["boundary"] == "terminal" and o["hp"] > 0, decisions=state["decision"])
    except Exception as e:
        result.update(error=f"{type(e).__name__}: {e}")
        log(f"ERROR {result['error']}")
        try:
            result["last"] = main.run_observe()
        except Exception:
            pass
    finally:
        result.update(seconds=round(time.time() - t0, 1), trace=str(main.run_trace_path), worker_log=str(main.log_path))
        main.close()
        if pool:
            pool.close()
    if out:
        live.close()
        Path(out).write_text(json.dumps({**result, "log": lines}, indent=1))
    return result


def describe(a, s):
    if a["type"] == "take_reward":
        r = s["rewards"][a["index"]]
        if "alternative" in a:
            return f"{a['alternative']} {r['type']} {r['cards']}"
        return f"take {r['type']} " + (r["cards"][a["card"]] if "card" in a else str(r.get("id") or r.get("gold") or ""))
    if a["type"] == "event_option":
        return a["key"]
    if a["type"] == "buy":
        return f"buy {a['item_type']} {next(e['id'] for e in s['shop'] if e['index'] == a['index'])} for {a['cost']}"
    if a["type"] == "choose" and s.get("choice"):
        return "choose " + ", ".join(s["choice"]["options"][i]["id"] for i in a["picks"])
    return json.dumps(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--character", default="CHARACTER.IRONCLAD")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--sims", type=int, default=10)
    ap.add_argument("--priors", help="per-character priors JSON; see priors.py")
    ap.add_argument("--out")
    args = ap.parse_args()
    r = play(args.seed, args.character, args.ascension, args.sims, args.priors, out=args.out)
    print(json.dumps({k: v for k, v in r.items() if k not in ("deck", "last")}, indent=1))
    raise SystemExit(0 if r.get("victory") else 1)
