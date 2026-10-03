"""Fight pools and random-policy baselines; see agents/nn/README.md.

A fight contains a seed-independent spec and optional recording or sampling metadata.
Starter encounters come from the native catalog; library fights come from a manifest.
"""

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from .contracts import character

from sts2bridge import CombatWorker, spec_from_run
from sts2bridge.fixtures import MCR, RUN

ASCENSION = 10
# Observed A10 entry HP is 80% of max, rounded down; native fresh-run setup starts at full HP.
START_HP = 0.8


def insatiable():
    run = json.loads(RUN.read_text())
    spec = spec_from_run(run, "ENCOUNTER.THE_INSATIABLE_BOSS", seed=None)
    return [dict(name="DEFECT/THE_INSATIABLE_BOSS", kind="boss", spec=spec,
                 recording=str(MCR))]


def starter(catalog):
    encounters = [e for e in catalog["encounters"] if e["act_index"] == 0 and e["acts"] and e["weak"]]
    return [dict(name=f"{c.split('.')[1]}/{e['id'].split('.')[1]}", kind=e["room_type"].lower(),
                 spec={"character": c, "ascension": ASCENSION, "encounter": e["id"],
                       "player": {"current_hp": int(START_HP * catalog["starting_hp"][c])}})
            for c in catalog["characters"] for e in encounters]


def pool(name, catalog=None, pool_file=None, split="train"):
    if name == "library":
        if pool_file is None:
            raise ValueError("library needs --pool-file from agents.nn.library")
        with open(pool_file) as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != 1 or manifest.get("pool") != "library":
            raise ValueError("unsupported library pool manifest")
        fights = manifest["splits"][split]
        if not fights:
            raise ValueError(f"library split {split} is empty")
        return fights
    if name == "insatiable":
        return insatiable()
    if catalog is None:
        with CombatWorker() as w:
            catalog = w.catalog()
    return starter(catalog)


def spec(fight, seed):
    return {**fight["spec"], "seed": seed}


# ---- check ----

def baseline(policy, rng, legal):
    plays = [a for a in legal if a["type"] != "end_turn"]
    return rng.choice(plays if policy == "spend" and plays else legal)


def _baseline_chunk(policy, fights, seeds, max_turns=30):
    rng = random.Random(0)
    out = []
    with CombatWorker() as w:
        for f in fights:
            for seed in seeds:
                try:
                    s = w.start(spec(f, seed), hashes=False, reuse_map=True)
                    hp0 = s["obs"]["player"]["hp"]
                    while s["boundary"] != "terminal" and s["obs"]["turn"] <= max_turns:
                        legal = s["legal"] or [{"type": "choose", "picks": list(range(s["choice"]["min"]))}]
                        s = w.step(baseline(policy, rng, legal))
                    o = s["obs"]
                    won = s["boundary"] == "terminal" and o["player"]["hp"] > 0 and not any(e["alive"] for e in o["enemies"])
                    out.append(dict(fight=f["name"], kind=f["kind"], character=character(f), won=won, kept=o["player"]["hp"] / hp0 if won else 0.0,
                                    turn=o["turn"]))
                except Exception as e:  # noqa: BLE001 - a sweep reports every failure, it does not stop at one
                    out.append(dict(fight=f["name"], kind=f["kind"], error=f"{type(e).__name__}: {e}"[:200]))
    return out


def check(fights, seeds=4, policy="spend", workers=8):
    """Report start failures, win rate, and HP kept by character for a baseline policy."""
    names = [f"EVAL-{i}" for i in range(seeds)]  # the evaluator's seeds, so a baseline and a network meet the same fights
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(_baseline_chunk, [policy] * workers, [fights[i::workers] for i in range(workers)],
                                     [names] * workers) for r in part]
    errors = [r for r in rows if "error" in r]
    for r in errors:
        print("FAILED", r["fight"], r["error"])
    by = defaultdict(list)
    for r in rows:
        if "error" not in r:
            by[r["character"]].append(r)
    print(f"{policy}: {len(fights)} fights x {seeds} seeds, {len(errors)} failed")
    for char, rs in sorted(by.items()):
        n = len(rs)
        print(f"  {char:12s} {len({r['fight'] for r in rs}):4d} fights  win {sum(r['won'] for r in rs) / n:.2f}"
              f"  HP kept {sum(r['kept'] for r in rs) / n:.2f}  turns {sum(r['turn'] for r in rs) / n:.1f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", choices=["insatiable", "starter"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--policy", choices=["spend", "random"], default="spend")
    args = ap.parse_args()
    check(pool(args.pool), args.seeds, args.policy)
