"""Dashboard provenance and launch commands; no subprocesses or game workers."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.nn import launch, watch


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
