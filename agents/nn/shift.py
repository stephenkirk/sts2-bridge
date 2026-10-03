"""Compare checkpoint play on the recorded fight and pinned winning line.

Each checkpoint produces its own greedy trajectory and action probabilities along the
same hand-played win. Greedy trajectories can diverge; the winning-line probe holds
states fixed. Per-turn probabilities multiply the probabilities of the winning actions.
--json exports both views. See agents/nn/README.md for commands.
"""

import argparse
import json
import math
from pathlib import Path

from sts2bridge import CombatWorker
from sts2bridge.fixtures import MCR, WIN

from .show import label, line, load, policy, short


def probe(worker, net, vocab, inputs):
    """[(turn, winning move, p(winning move), top pick, p(top pick))] along the hand-win."""
    state, out = worker.load(MCR), []
    for action in inputs:
        p, _ = policy(net, vocab, state)
        i = state["legal"].index(action)
        top = int(p.argmax())
        out.append((state["obs"]["turn"], label(state, action), float(p[i]), label(state, state["legal"][top]),
                    float(p[top])))
        state = worker.step(action)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    inputs = json.loads(WIN.read_text())["inputs"]
    rows, prev = [], None
    with CombatWorker() as w:
        for path in args.ckpts:
            net, vocab, ck = load(path)
            turns, end = line(w, net, vocab)
            pr = probe(w, net, vocab, inputs)
            rows.append(dict(ckpt=path, it=ck.get("it"), episodes=ck.get("episodes"), turns=turns, end=end, probe=pr))

            name = Path(path).stem + (f" (it {ck['it']})" if ck.get("it") else "")
            print(f"\n=== {name}: {'WIN' if end['won'] else 'loss'}, hp {end['hp']}, boss {end['boss'] or 'dead'}")
            for k, t in enumerate(turns):
                cards = " · ".join(short(m) for m, _ in t["moves"] if m != "end")
                changed = prev is None or k >= len(prev) or [m for m, _ in prev[k]["moves"]] != [m for m, _ in t["moves"]]
                print(f"  {'*' if changed and prev is not None else ' '} t{t['turn']} hp {t['hp']:2d} boss {t['boss']:3d} | {cards}")
            prev = turns

            print("  hand-win probe, p(the whole turn as the win played it):")
            for turn in sorted({x[0] for x in pr}):
                moves = [x for x in pr if x[0] == turn]
                joint = math.prod(x[2] for x in moves)
                worst = min(moves, key=lambda x: x[2])
                note = f"  weakest: {short(worst[1])} {worst[2]:.2f}" + (
                    f", prefers {short(worst[3])} {worst[4]:.2f}" if worst[3] != worst[1] else "")
                print(f"    t{turn} {joint:6.3f}{note}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
