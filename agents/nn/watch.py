"""Serve training metrics and optional checkpoint rematches at localhost:8765.

--rematch accepts a JSON list of {"name", "spec", "you": {"lost", "turns"}}.
Results are appended to runs/<name>/rematch.jsonl; only the newest snapshot is replayed.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .contracts import character, heldout

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# HP lost per fight over wins, from docs/nn-experiments-2026-09-30.md (2026-09-29): random card spending on the starter pool's
# EVAL seeds, and the user's first fight of 210 A10 runs.
BASELINES = {
    "DEFECT": dict(spend=12.1, you=3.6), "IRONCLAD": dict(spend=15.1, you=7.4),
    "NECROBINDER": dict(spend=12.3, you=1.6), "REGENT": dict(spend=14.0, you=4.6), "SILENT": dict(spend=15.1, you=2.7),
}


def runs():
    found = [p.parent for p in RUNS.glob("*/log.jsonl") if p.parent.name != "bench"]
    return [p.name for p in sorted(found, key=lambda p: (p / "log.jsonl").stat().st_mtime, reverse=True)]


def hp0_by_char(run):
    path = RUNS / run / "fights.json"
    if not path.exists():
        return {}
    fights = json.loads(path.read_text())
    return {character(f): f["spec"]["player"]["current_hp"] for f in fights
            if "current_hp" in f["spec"].get("player", {})}


def run_metadata(sessions):
    if not sessions:
        return {}
    original, latest = sessions[0], sessions[-1]
    args = latest["args"]
    initialization = latest.get("initialization")
    if initialization is None:
        resumed = latest.get("resumed_at") is not None
        initialization = dict(mode="resume" if resumed else "branch" if args.get("init_from") else "fresh",
                              checkpoint=args.get("init_from"), iteration=latest.get("resumed_at"),
                              optimizer="restored" if resumed else "fresh")
    return dict(initialization=initialization,
                origin_checkpoint=original.get("initialization", {}).get("checkpoint") or original["args"].get("init_from"),
                pool=args.get("pool", "starter"), reward=args.get("reward") or
                ("damage" if args.get("pool") == "insatiable" else "hp"),
                turn_cost_hp=args.get("turn_cost_hp", 0), eval_prefix=args.get("eval_prefix", "EVAL"),
                eval_seeds=args.get("eval_seeds"), sessions=len(sessions))


_cache = {}


def read_json(path):
    """fights.json and friends only change when a run starts, and a library pool is a few megabytes."""
    stamp = path.stat().st_mtime
    if _cache.get(path, (None,))[0] != stamp:
        _cache[path] = stamp, json.loads(path.read_text())
    return _cache[path][1]


def library(run, vocab_added):
    """Return validation provenance and card coverage across training and validation loadouts."""
    train_path, eval_path = RUNS / run / "fights.json", RUNS / run / "eval_fights.json"
    if not eval_path.exists():
        return None
    name = lambda i: i.split(".", 1)[-1]
    scenarios = {}
    for f in read_json(eval_path):
        p = f["spec"]["player"]
        scenarios[f["name"]] = dict(char=name(f["spec"]["character"]), encounter=name(f["spec"]["encounter"]),
                                    floor=f["source"]["floor"], pool=f.get("encounter_pool"), hp=p["current_hp"],
                                    max_hp=p["max_hp"], outcome=f["source"]["outcome"], you=f["source"].get("you"),
                                    deck=[name(c["id"]) + "+" * c.get("current_upgrade_level", 0) for c in p["deck"]],
                                    relics=[name(r["id"]) for r in p["relics"]])
    cards = {}
    for f in read_json(train_path):
        char = name(f["spec"]["character"])
        for cid in {c["id"] for c in f["spec"]["player"]["deck"]}:
            c = cards.setdefault(name(cid), dict(decks=0, chars=set()))
            c["decks"] += 1
            c["chars"].add(char)
    trained = set(cards)
    for s in scenarios.values():
        for cid in {c.rstrip("+") for c in s["deck"]}:
            cards.setdefault(cid, dict(decks=0, chars=set()))["chars"].add(s["char"])
    added = {name(v.split(":", 1)[1]) for v in vocab_added or () if v.startswith("card:")}
    return dict(scenarios=scenarios, loadouts=len(read_json(train_path)),
                cards={cid: dict(decks=c["decks"], chars=sorted(c["chars"]), new=cid in added, trained=cid in trained)
                       for cid, c in cards.items()},
                vocab_known=vocab_added is not None)


def data(run, rematch):
    hp0 = hp0_by_char(run)
    fight_path = RUNS / run / "fights.json"
    chars = {f["name"]: character(f) for f in read_json(fight_path)} if fight_path.exists() else {}
    def lost(char, stats):
        # Library fights have different entry HP even within one character.
        if "hp_change" in stats:
            return round(stats["hp_change"], 2)
        return round((1 - stats["kept"]) * hp0[char], 2) if char in hp0 else None
    rows, evals, before, sessions, vocab_added = [], [], 0, [], None
    for line in (RUNS / run / "log.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # The learner may currently be writing the last line.
        if "it" not in r:
            if "args" in r:
                sessions.append(r)
            if "vocab_added" in r:
                vocab_added = r["vocab_added"]
            continue
        # fights played per second: an update's fights, used and dropped, over its collect and learn time
        spent = r["collect_s"] + r["update_s"]
        fps = round((r["episodes"] - before + r["dropped"]) / spent, 1) if spent and before else None
        before = r["episodes"]
        rows.append(dict(it=r["it"], fights=r["episodes"], min=r["min"], fps=fps, entropy=r["entropy"],
                         vloss=r["vloss"], clipfrac=r["clipfrac"], dropped=r["dropped"], lag=r["lag"],
                         train={c: lost(c, v) for c, v in r["train_by_char"].items()}))
        if "eval" in r:
            ev = r["eval"]
            stats = heldout(ev)
            evals.append(dict(it=ev["version"], fights=ev.get("episodes", r["episodes"]), min=r["min"],
                              timeouts=stats.get("timeouts"),
                              win_total=stats["win"], reward=stats["reward"],
                              by_pool={k: dict(lost=v["hp_change"], win=v["win"], n=v["n"])
                                       for k, v in stats.get("by_pool", {}).items()},
                              cards=ev.get("heldout_cards"),
                              by_char={c: lost(c, v) for c, v in stats["by_char"].items()},
                              win={c: v["win"] for c, v in stats["by_char"].items()},
                              by_fight={k: dict(lost=lost(chars.get(k), v), win=v["win"], turns=v.get("turn_p95"))
                                        for k, v in stats.get("by_fight", {}).items()}))
    return dict(run=run, runs=runs(), hp0=hp0, baselines=BASELINES, rows=rows, evals=evals,
                metadata=run_metadata(sessions), library=library(run, vocab_added),
                rematch=rematch.results(run) if rematch else None)


class Rematch:
    """Replay each requested run's newest snapshot in a background worker."""

    def __init__(self, path):
        self.fights = json.loads(Path(path).read_text())
        self.lock = threading.Lock()
        self.wanted = set()
        threading.Thread(target=self.loop, daemon=True).start()

    def results(self, run):
        self.wanted.add(run)
        path = RUNS / run / "rematch.jsonl"
        done = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        return dict(fights=[dict(name=f["name"], char=character(f), you=f["you"])
                            for f in self.fights], snapshots=done)

    def loop(self):
        import torch

        from sts2bridge import CombatWorker

        from .show import load, policy
        torch.set_num_threads(1)
        with CombatWorker() as w:
            while True:
                for run in list(self.wanted):
                    snaps = sorted((RUNS / run / "snapshots").glob("it*.pt"))
                    path = RUNS / run / "rematch.jsonl"
                    seen = {json.loads(l)["snapshot"] for l in path.read_text().splitlines()} if path.exists() else set()
                    if snaps and snaps[-1].name not in seen:
                        try:
                            net, vocab, ck = load(snaps[-1])
                        except Exception:  # noqa: BLE001 - a snapshot mid-write; the next pass reads it
                            continue
                        out = [self.play(w, net, vocab, policy, f["spec"]) for f in self.fights]
                        with open(path, "a") as fh:
                            fh.write(json.dumps(dict(snapshot=snaps[-1].name, it=ck.get("it"),
                                                     fights_played=ck.get("episodes"), fights=out)) + "\n")
                time.sleep(3)

    @staticmethod
    def play(w, net, vocab, policy, spec):
        short = lambda cid: cid.split(".")[-1].replace("_", " ").title()
        s = w.start(spec, hashes=False)
        hp, lost, turns = s["obs"]["player"]["hp"], 0, []
        while s["boundary"] != "terminal" and s["obs"]["turn"] <= 30:
            o = s["obs"]
            if not turns or turns[-1]["turn"] != o["turn"]:
                turns.append(dict(turn=o["turn"], hp=o["player"]["hp"], plays=[]))
            if not s["legal"]:
                a = {"type": "choose", "picks": list(range(s["choice"]["min"]))}
            else:
                a = s["legal"][int(policy(net, vocab, s)[0].argmax())]
            if a["type"] == "play":
                c, t = o["hand"][a["hand"]], a["target"]
                turns[-1]["plays"].append(short(c["id"]) + ("" if t is None else f" → {short(o['enemies'][t]['id'])}"))
            elif a["type"] == "choose" and a["picks"]:
                turns[-1]["plays"].append("pick " + short(s["choice"]["options"][a["picks"][0]]["id"]))
            s = w.step(a)
            now = s["obs"]["player"]["hp"]
            lost += max(0, hp - now)  # HP taken, not counting a heal such as Burning Blood's at the end
            hp = now
        won = s["boundary"] == "terminal" and s["obs"]["player"]["hp"] > 0
        return dict(lost=lost, turns=s["obs"]["turn"], won=won, line=turns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rematch", help="a JSON list of fights to replay with each new snapshot")
    args = ap.parse_args()
    rematch = Rematch(args.rematch) if args.rematch else None
    page = (HERE / "watch.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body, kind = page, "text/html; charset=utf-8"
            elif url.path == "/data":
                names = runs()
                run = parse_qs(url.query).get("run", [names[0] if names else ""])[0]
                if run not in names:
                    self.send_error(404, "no such run")
                    return
                body, kind = json.dumps(data(run, rematch)).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    print(f"watching {RUNS} at http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
