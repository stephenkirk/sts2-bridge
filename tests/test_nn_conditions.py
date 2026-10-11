"""Evaluation comparability is about the exam, independently of training settings."""

import copy
import unittest

from agents.nn.contracts import compare_conditions, condition_id, digest, evaluation_conditions
from agents.nn.compare import comparison_checks


class EvaluationConditionsTests(unittest.TestCase):
    def conditions(self, *, hp=60, seed="EVAL-0", build="game-v1", max_turns=30):
        return evaluation_conditions(
            [dict(name="DEFECT/BOSS", spec=dict(character="CHARACTER.DEFECT", player=dict(current_hp=hp)))],
            [seed], max_turns=max_turns, build={"game/sts2.dll": build, "worker/CombatWorker.dll": "bridge-v1",
                                             "worker/GodotSharp.dll": "stubs-v1", "python/episode.py": "runner-v1", "python/worker.py": "client-v1"})

    def test_same_exam_matches_without_training_settings(self):
        a, b = self.conditions(), self.conditions()
        self.assertEqual(compare_conditions(a, b)["status"], "matching")
        self.assertNotIn("reward", a)
        # JSON object order and scenario provenance do not change the exam.
        c = evaluation_conditions(
            [dict(name="DEFECT/BOSS", spec=dict(player=dict(current_hp=60), character="CHARACTER.DEFECT", seed="ignored"),
                  source=dict(run="different history"), sample_weight=5)],
            ["EVAL-0"], max_turns=30, build=b["build"])
        self.assertEqual(a, c)

    def test_same_fight_name_and_seed_number_do_not_hide_changed_conditions(self):
        a = self.conditions()
        for b, reason in ((self.conditions(hp=30), "fights"), (self.conditions(seed="FINAL-0"), "seeds"),
                          (self.conditions(build="game-v2"), "build"), (self.conditions(max_turns=10), "method")):
            with self.subTest(reason=reason):
                check = compare_conditions(a, b)
                self.assertEqual(check["status"], "different")
                self.assertIn(reason, check["reason"])

    def test_missing_incomplete_or_tampered_metadata_cannot_certify_a_match(self):
        a = self.conditions()
        bad = copy.deepcopy(a)
        bad["seeds"] = ["CHANGED-0"]
        missing_build = self.conditions()
        missing_build["build"]["game/sts2.dll"] = None
        missing_build["sha256"] = digest({k: v for k, v in missing_build.items() if k != "sha256"})
        partial_build = copy.deepcopy(a)
        del partial_build["build"]["worker/GodotSharp.dll"]
        partial_build["sha256"] = digest({k: v for k, v in partial_build.items() if k != "sha256"})
        for b in (None, {}, bad, missing_build, partial_build, dict(a, format_version=2)):
            with self.subTest(conditions=b):
                self.assertEqual(condition_id(b), "unknown")
                self.assertEqual(compare_conditions(a, b)["status"], "unknown")

    def test_checks_follow_each_evaluation_instead_of_latest_session_settings(self):
        old, new = self.conditions(), self.conditions(seed="NEW-0")
        row = lambda c: dict(conditions=c, condition_id=condition_id(c))
        arms = [dict(evals=[row(old), row(new)]), dict(evals=[row(old), row(None)])]
        checks = comparison_checks(arms)
        self.assertEqual(checks[f"{old['sha256']}/{old['sha256']}"]["status"], "matching")
        self.assertEqual(checks[f"{new['sha256']}/{old['sha256']}"]["status"], "different")
        self.assertEqual(checks[f"{old['sha256']}/unknown"]["status"], "unknown")
