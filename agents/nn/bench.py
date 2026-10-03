"""Benchmark fixed-budget Insatiable training against an accepted result.

Train for --minutes, excluding warm-up, then evaluate the last checkpoint greedily on
200 EVAL seeds and the recording. Lower mean enemy HP is better. Results are appended
to bench/results.jsonl; bench/ratchet.json stores the accepted result. See README.md.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

BENCH = HERE / "bench"
RESULTS, RATCHET = BENCH / "results.jsonl", BENCH / "ratchet.json"
EVAL_SEEDS = 200


def _eval_chunk(ckpt, seeds):
    import torch
    torch.set_num_threads(1)
    from sts2bridge import CombatWorker

    from .fights import insatiable
    from .show import load
    from .episode import Scoring, play
    net, vocab, _ = load(ckpt)
    fight = insatiable()[0]
    with CombatWorker() as w:
        return [play(w, net, vocab, fight, s, greedy=True, record=False, scoring=Scoring("damage", 0))[1] for s in seeds]


def evaluate(ckpt, workers=8):
    seeds = [f"EVAL-{i}" for i in range(EVAL_SEEDS)]
    with ProcessPoolExecutor(workers) as ex:
        held = [r for part in ex.map(_eval_chunk, [ckpt] * workers, [seeds[i::workers] for i in range(workers)])
                for r in part]
        rec = ex.submit(_eval_chunk, ckpt, [None]).result()[0]
    return dict(heldout_boss=round(sum(r["boss"] for r in held) / len(held), 1),
                heldout_win=sum(r["won"] for r in held) / len(held),
                heldout_turn=round(sum(r["turn"] for r in held) / len(held), 2),
                recorded=dict(won=rec["won"], boss=rec["boss"], turn=rec["turn"]))


def summarise(log):
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    its = [r for r in rows if "it" in r]
    wall = sum(r["collect_s"] + r["update_s"] for r in its)
    used, dropped = its[-1]["episodes"], sum(r.get("dropped", 0) for r in its)
    tail = its[-max(1, len(its) // 10):]
    return dict(updates=len(its), wall_s=round(wall), fights_used=used, fights_dropped=dropped,
                learn_dps=round(sum(r["samples"] for r in its) / wall), used_fps=round(used / wall, 1),
                played_fps=round((used + dropped) / wall, 1),
                update_share=round(sum(r["update_s"] for r in its) / wall, 2),
                train_boss_tail=round(sum(r["train_boss"] for r in tail) / len(tail), 1))


def git_rev():
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "agents/nn", "sts2bridge", "dotnet"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return rev + ("+dirty" if dirty else "")


def show(row, bar):
    keys = ["heldout_boss", "heldout_win", "learn_dps", "used_fps", "played_fps", "update_share", "fights_used",
            "train_boss_tail"]
    print(f"\n{row['note']!r} @ {row['rev']}  ({row['minutes']} min, args {' '.join(row['args']) or '-'})")
    for k in keys:
        v, b = row[k], bar and bar.get(k)
        delta = f"   bar {b}  ({v - b:+.3g})" if isinstance(b, (int, float)) else ""
        print(f"  {k:16s} {v}{delta}")
    print(f"  recorded fight   {row['recorded']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--note", default="")
    ap.add_argument("--accept", nargs="?", const="last", metavar="NOTE",
                    help="make the last result, or the latest result with NOTE, the bar; run nothing")
    ap.add_argument("train_args", nargs="*", help="passed to train.py after --")
    args = ap.parse_args()
    BENCH.mkdir(exist_ok=True)
    bar = json.loads(RATCHET.read_text()) if RATCHET.exists() else None

    if args.accept is not None:
        rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
        selected = rows[-1] if args.accept == "last" else next(
            (row for row in reversed(rows) if row["note"] == args.accept), None)
        if selected is None:
            ap.error(f"no result with note {args.accept!r}")
        RATCHET.write_text(json.dumps(selected, indent=1) + "\n")
        print(f"bar is now {selected['note']!r} @ {selected['rev']}: held-out boss {selected['heldout_boss']}")
        return

    name = "bench/" + time.strftime("%m%d-%H%M%S")
    py = ROOT / ".venv" / "bin" / "python"
    cmd = [str(py), "-m", "agents.nn.train", "--name", name, "--hours", str(args.minutes / 60),
           "--eval-every", "1000000", "--pool", "insatiable",
           *args.train_args]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    run = HERE / "runs" / name
    row = dict(note=args.note, rev=git_rev(), when=time.strftime("%Y-%m-%d %H:%M"), minutes=args.minutes,
               args=args.train_args, **summarise(run / "log.jsonl"), **evaluate(run / "ckpt.pt"))
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    show(row, bar)


if __name__ == "__main__":
    main()
