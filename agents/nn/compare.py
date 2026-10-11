"""Serve two runs side by side at localhost:8772/?a=<run>&b=<run>, matched by fights played.

Reads runs/<name>/log.jsonl. A background worker runs the shift.py hand-win probe on each new
snapshot of a requested Insatiable run and appends it to runs/<name>/probe.jsonl.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .contracts import compare_conditions, condition_id, heldout
from .watch import HERE, RUNS, run_metadata, runs


def lines(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the writer may be mid-line
    return out


def arm(run):
    rows, evals, sessions, before = [], [], [], 0
    for r in lines(RUNS / run / "log.jsonl"):
        if "it" not in r:
            if "args" in r:
                sessions.append(r)
            continue
        spent = r["collect_s"] + r["update_s"]
        fps = round((r["episodes"] - before + r["dropped"]) / spent, 1) if spent and before else None
        before = r["episodes"]
        rows.append(dict(it=r["it"], fights=r["episodes"], min=r["min"], fps=fps, train_win=r["train_win"],
                         entropy=r["entropy"], vloss=r["vloss"], clipfrac=r["clipfrac"], dropped=r["dropped"]))
        if "eval" in r:
            ev, stats = r["eval"], heldout(r["eval"])
            cards = {}
            for by_card in (ev.get("heldout_cards") or {}).values():
                for cid, (playable, played) in by_card.items():
                    c = cards.setdefault(cid, [0, 0])
                    c[0] += playable
                    c[1] += played
            evals.append(dict(it=ev["version"], fights=ev.get("episodes", r["episodes"]), win=stats["win"],
                              boss=stats["boss"], turn=stats["turn"], n=stats["n"], cards=cards,
                              recorded=ev.get("recorded_greedy") and {k: ev["recorded_greedy"][k] for k in ("won", "hp", "boss", "turn")},
                              recorded_sampled=ev.get("recorded_sampled_win"), each=ev.get("heldout_fights")))
            evals[-1].update(conditions=ev.get("conditions"), condition_id=condition_id(ev.get("conditions")))
    return dict(run=run, metadata=run_metadata(sessions), rows=rows, evals=sorted(evals, key=lambda e: e["fights"]),
                probe=sorted({p["snapshot"]: p for p in lines(RUNS / run / "probe.jsonl")}.values(), key=lambda p: p["fights"]))


def comparison_checks(arms):
    """Check each distinct pair of exams, including evaluations before a resume."""
    exams = [{e["condition_id"]: e["conditions"] for e in a["evals"]} for a in arms]
    return {f"{ka}/{kb}": compare_conditions(a, b)
            for ka, a in exams[0].items() for kb, b in exams[1].items()}


def turn5(rows):
    """Hand-win probe rows [(turn, move, p(move), top, p(top))] reduced to the decisions this experiment watches.

    The hand win plays Synchronize+, Hologram+ -> Boot Sequence+, Boot Sequence+, Quadcast on turn 5, so `quadcast` is
    p(Quadcast) with Synchronize+'s Focus already up: the evoke the win spent it on.
    """
    t5 = [r for r in rows if r[0] == 5]
    pick = lambda name: next((r[2] for r in t5 if r[1] == name), None)
    joint = {}
    for t, _, p, _, _ in rows:
        joint[t] = joint.get(t, 1.0) * p
    return dict(sync_first=t5[0][2] if t5 and t5[0][1] == "CARD.SYNCHRONIZE+" else None,
                boot=pick("-> CARD.BOOT_SEQUENCE+"), quadcast=pick("CARD.QUADCAST"),
                top_at_quadcast=next((r[3] for r in t5 if r[1] == "CARD.QUADCAST"), None), top_after_hologram=next((r[3] for r in t5 if r[1].startswith("->")), None),
                joint={str(k): v for k, v in joint.items()})


class Probe:
    """Probe each new snapshot of the requested runs once, in a single background worker."""

    def __init__(self):
        self.wanted = set()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        import torch

        from sts2bridge import CombatWorker
        from sts2bridge.fixtures import WIN

        from .shift import probe
        from .show import load
        torch.set_num_threads(1)
        inputs = json.loads(WIN.read_text())["inputs"]
        with CombatWorker() as w:
            while True:
                for run in list(self.wanted):
                    path = RUNS / run / "probe.jsonl"
                    seen = {p["snapshot"] for p in lines(path) if "quadcast" in p}  # older rows are probed again
                    for snap in sorted((RUNS / run / "snapshots").glob("it*.pt")):
                        if snap.name in seen:
                            continue
                        try:
                            net, vocab, ck = load(snap)
                        except Exception:  # noqa: BLE001 - a snapshot mid-write; the next pass reads it
                            break
                        if ck.get("pool") != "insatiable":
                            self.wanted.discard(run)
                            break
                        row = dict(snapshot=snap.name, it=ck.get("it"), fights=ck.get("episodes") or 0,
                                   **turn5(probe(w, net, vocab, inputs)))
                        with open(path, "a") as fh:
                            fh.write(json.dumps(row) + "\n")
                time.sleep(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--no-probe", action="store_true", help="skip the turn-5 probe worker")
    args = ap.parse_args()
    prober = None if args.no_probe else Probe()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body, kind = (HERE / "compare.html").read_bytes(), "text/html; charset=utf-8"
            elif url.path == "/data":
                q, names = parse_qs(url.query), runs()
                picked = [q.get(k, [names[i] if len(names) > i else ""])[0] for i, k in enumerate(("a", "b"))]
                if any(p not in names for p in picked):
                    self.send_error(404, "no such run")
                    return
                if prober:
                    prober.wanted.update(picked)
                arms = [arm(p) for p in picked]
                body, kind = json.dumps(dict(runs=names, arms=arms, checks=comparison_checks(arms))).encode(), "application/json"
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

    print(f"comparing runs in {RUNS} at http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
