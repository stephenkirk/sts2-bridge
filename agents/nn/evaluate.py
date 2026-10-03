"""Compare two checkpoints greedily on identical fight seeds; never trains either network."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from sts2bridge import CombatWorker

from . import fights
from .episode import Scoring, play
from .metrics import mean, summary
from .show import load


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--seeds", type=int, default=10, help="seeds per fight")
    ap.add_argument("--seed-prefix", default="EVAL", help="use a fresh namespace for the final comparison")
    ap.add_argument("--turn-cost-hp", type=float, default=0.05)
    ap.add_argument("--output", type=Path, required=True, help="JSON with per-seed paired results and summaries")
    ap.add_argument("--pool-file", type=Path, help="override both checkpoints' fight pool with a library manifest")
    ap.add_argument("--split", choices=["validation", "test"], default="test", help="library split; test is never used by training")
    args = ap.parse_args()
    if args.seeds < 1 or args.turn_cost_hp < 0:
        ap.error("--seeds must be positive and --turn-cost-hp nonnegative")
    torch.set_num_threads(1)
    base_net, base_vocab, base_ck = load(args.baseline)
    cand_net, cand_vocab, cand_ck = load(args.candidate)
    if base_ck["pool"] != cand_ck["pool"] and not args.pool_file:
        ap.error("checkpoints must use the same fight pool")
    pool_name = "library" if args.pool_file else base_ck["pool"]
    pool_file = args.pool_file or cand_ck.get("pool_file")
    if pool_name == "library" and not args.pool_file:
        if not pool_file or hashlib.sha256(Path(pool_file).read_bytes()).hexdigest() != cand_ck.get("pool_digest"):
            ap.error("checkpoint's library manifest is missing or changed; supply --pool-file explicitly")
    scoring = Scoring("damage" if pool_name == "insatiable" else "hp", args.turn_cost_hp)
    pairs = []
    with CombatWorker() as worker:
        for fight in fights.pool(pool_name, pool_file=pool_file, split=args.split):
            for i in range(args.seeds):
                seed = f"{args.seed_prefix}-{i}"
                base = play(worker, base_net, base_vocab, fight, seed, greedy=True, record=False, scoring=scoring)[1]
                cand = play(worker, cand_net, cand_vocab, fight, seed, greedy=True, record=False, scoring=scoring)[1]
                pairs.append(dict(fight=fight["name"], seed=seed, baseline=base, candidate=cand))
    report = dict(baseline=str(args.baseline), candidate=str(args.candidate), seed_prefix=args.seed_prefix,
                  pool=pool_name, split=args.split if pool_name == "library" else None,
                  turn_cost_hp=args.turn_cost_hp,
                  summaries={k: summary([p[k] for p in pairs], fights=True) for k in ("baseline", "candidate")},
                  paired=dict(hp_change_delta=mean(p["candidate"]["hp_change"] - p["baseline"]["hp_change"] for p in pairs),
                              turns_delta=mean(p["candidate"]["turn"] - p["baseline"]["turn"] for p in pairs),
                              wins_gained=sum(p["candidate"]["won"] and not p["baseline"]["won"] for p in pairs),
                              wins_lost=sum(p["baseline"]["won"] and not p["candidate"]["won"] for p in pairs)),
                  pairs=pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(summaries=report["summaries"], paired=report["paired"]), indent=2))


if __name__ == "__main__":
    main()
