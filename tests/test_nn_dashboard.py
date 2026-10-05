"""Dashboard provenance and launch commands; no subprocesses or game workers."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.nn import compare, launch, pair, watch


class DashboardTests(unittest.TestCase):
    def test_launcher_passes_training_flags_and_separates_watch_flags(self):
        args = argparse.Namespace(name="branch", watch_port=8766, watch_rematch="rematch.json")
        watcher, trainer = launch.commands(args, ["--init-from", "it00340.pt", "--hours", "0.3"])
        self.assertEqual(watcher, [sys.executable, "-m", "agents.nn.watch", "--port", "8766", "--rematch", "rematch.json"])
        self.assertEqual(trainer, [sys.executable, "-m", "agents.nn.train", "--name", "branch",
                                  "--init-from", "it00340.pt", "--hours", "0.3"])

    def test_branch_metadata(self):
        metadata = watch.run_metadata([dict(args=dict(init_from="it00340.pt", reward="hp", turn_cost_hp=0.05),
                                            initialization=dict(mode="branch", checkpoint="it00340.pt", iteration=340,
                                                                optimizer="fresh"))])
        self.assertEqual(metadata["initialization"]["iteration"], 340)
        self.assertEqual(metadata["origin_checkpoint"], "it00340.pt")
        self.assertEqual(metadata["turn_cost_hp"], 0.05)

    def test_resumed_branch_retains_original_source(self):
        metadata = watch.run_metadata([dict(args=dict(init_from="it00340.pt")),
                                      dict(args=dict(resume=True), resumed_at=20)])
        self.assertEqual(metadata["initialization"]["mode"], "resume")
        self.assertEqual(metadata["origin_checkpoint"], "it00340.pt")

    def test_legacy_run_is_fresh_without_turn_cost(self):
        metadata = watch.run_metadata([dict(args=dict(pool="starter"), resumed_at=None)])
        self.assertEqual(metadata["initialization"]["mode"], "fresh")
        self.assertEqual(metadata["turn_cost_hp"], 0)

    def test_old_and_new_evaluations_render_identically_with_arbitrary_names(self):
        stats = dict(win=1, reward=0.9, by_char={"DEFECT": dict(win=1, kept=0.9)},
                     by_fight={"renamed fight": dict(win=1, kept=0.9)})
        versions = [dict(heldout=stats), {"heldout_" + k: v for k, v in stats.items()}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(watch, "RUNS", Path(tmp)):
            run = Path(tmp) / "branch"
            run.mkdir()
            (run / "fights.json").write_text(json.dumps([
                dict(name="renamed fight", spec=dict(character="CHARACTER.DEFECT", player=dict(current_hp=60)))
            ]))
            rendered = []
            for ev in versions:
                row = dict(it=1, episodes=10, min=1, collect_s=1, update_s=1, dropped=0,
                           entropy=0, vloss=0, clipfrac=0, lag=0, train_by_char={}, eval=dict(version=1, **ev))
                (run / "log.jsonl").write_text(json.dumps(row) + "\n")
                data = watch.data("branch", None)
                self.assertEqual(data["hp0"], {"DEFECT": 60})
                self.assertEqual(data["evals"][0]["by_fight"]["renamed fight"]["lost"], 6)
                rendered.append(data["evals"])
            self.assertEqual(*rendered)

    def test_nested_summary_takes_precedence_over_legacy_copies(self):
        from agents.nn.contracts import heldout
        self.assertEqual(heldout(dict(heldout=dict(win=1), heldout_win=0)), dict(win=1))

    def test_startup_header_and_partial_log_are_readable(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(watch, "RUNS", Path(tmp)):
            run = Path(tmp) / "branch"
            run.mkdir()
            (run / "log.jsonl").write_text(json.dumps(dict(args=dict(init_from="it00340.pt"))) + '\n{"it":')
            data = watch.data("branch", None)
            self.assertEqual(data["metadata"]["initialization"]["mode"], "branch")
            self.assertEqual(data["rows"], [])
            self.assertEqual(data["hp0"], {})


class CompareTests(unittest.TestCase):
    def test_pair_passes_training_flags_to_both_arms(self):
        args = argparse.Namespace(arm=[("w", "insatiable-library-win"), ("d", "insatiable-library-damage")], port=8772)
        server, trainers = pair.commands(args, ["--init-from", "init.pt", "--actors", "3"])
        self.assertEqual(server, [sys.executable, "-m", "agents.nn.compare", "--port", "8772"])
        self.assertEqual(trainers[1], [sys.executable, "-m", "agents.nn.train", "--name", "d", "--experiment",
                                       "insatiable-library-damage", "--init-from", "init.pt", "--actors", "3"])
        with self.assertRaises(argparse.ArgumentTypeError):
            pair.arm("no-equals")

    def test_turn5_picks_the_watched_decisions(self):
        rows = [(4, "CARD.TURBO+", .5, "CARD.TURBO+", .5), (5, "CARD.SYNCHRONIZE+", .1, "CARD.HOLOGRAM+", .8),
                (5, "CARD.HOLOGRAM+", .9, "CARD.HOLOGRAM+", .9), (5, "-> CARD.BOOT_SEQUENCE+", .02, "-> CARD.REFRACT+", .7)]
        out = compare.turn5(rows)
        self.assertEqual((out["sync_first"], out["boot"], out["top_after_hologram"]), (.1, .02, "-> CARD.REFRACT+"))
        self.assertAlmostEqual(out["joint"]["5"], .1 * .9 * .02)

    def test_arm_sorts_evaluations_and_merges_card_counts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(compare, "RUNS", Path(tmp)):
            run = Path(tmp) / "w"
            run.mkdir()
            base = dict(collect_s=1, update_s=1, dropped=0, entropy=1, vloss=0, clipfrac=0, train_win=0)
            ev = lambda v: dict(version=v, episodes=v * 10, heldout=dict(win=.1, boss=100, turn=5, n=100),
                                heldout_cards={"DEFECT": {"CARD.ZAP": [4, 1]}}, heldout_fights=[dict(seed=0, won=True)])
            rows = [dict(args=dict(reward="win")), dict(it=2, episodes=20, min=1, eval=ev(2), **base),
                    dict(it=3, episodes=30, min=1, eval=ev(0), **base)]
            (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + '\n{"it":')
            (run / "probe.jsonl").write_text(json.dumps(dict(snapshot="it00000.pt", fights=0, boot=.01)) + "\n")
            out = compare.arm("w")
            self.assertEqual([e["it"] for e in out["evals"]], [0, 2])
            self.assertEqual(out["evals"][0]["cards"], {"CARD.ZAP": [4, 1]})
            self.assertEqual(out["evals"][0]["each"], [dict(seed=0, won=True)])
            self.assertEqual(out["metadata"]["reward"], "win")
            self.assertEqual(out["probe"][0]["boot"], .01)
