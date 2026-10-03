"""Reward and checkpoint regressions using synthetic data; no game workers."""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    import torch
    from agents.nn import train, episode, metrics
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires nn dependencies")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.scoring = episode.Scoring("hp", 0.05)

    def result(self, hp=59, turn=3, terminal=True, alive=False):
        state = dict(boundary="terminal" if terminal else "awaiting_input",
                     obs=dict(player=dict(hp=hp), turn=turn, enemies=[dict(hp=5 if alive else 0, alive=alive)]))
        return episode.outcome(state, boss0=20, hp0=60, max0=75, scoring=self.scoring)

    def test_turn_cost_is_immediate_and_rewards_sum_to_outcome(self):
        res = self.result()
        rewards = episode.step_rewards([60, 60, 59], res, 75, [1, 2, 2], scoring=self.scoring)
        self.assertAlmostEqual(rewards[0], -0.05 / 75)
        self.assertAlmostEqual(rewards[1], -1 / 75)
        self.assertAlmostEqual(sum(rewards), res["reward"])
        self.assertAlmostEqual(res["reward"], 1 - 1 / 75 - 0.1 / 75)

    def test_same_turn_actions_have_no_turn_charge(self):
        res = self.result(turn=1)
        rewards = episode.step_rewards([60, 60], res, 75, [1, 1], scoring=self.scoring)
        self.assertEqual(rewards[0], 0)

    def test_timeout_remains_worse_than_low_hp_win(self):
        timeout = self.result(hp=60, turn=31, terminal=False, alive=True)
        win = self.result(hp=1, turn=30)
        self.assertTrue(timeout["timeout"])
        self.assertFalse(win["timeout"])
        self.assertLess(timeout["reward"], win["reward"])

    def test_damage_reward_unchanged(self):
        self.scoring = episode.Scoring("damage", 0)
        res = self.result()
        self.assertEqual(res["reward"], 1)
        self.assertEqual(episode.step_rewards([60, 59], res, 75, [1, 3], scoring=self.scoring), [0, 1])

    def test_scoring_calls_do_not_contaminate_each_other(self):
        hp = self.result()["reward"]
        self.scoring = episode.Scoring("survival", 0)
        self.assertEqual(self.result()["reward"], 1)
        self.scoring = episode.Scoring("hp", 0.05)
        self.assertEqual(self.result()["reward"], hp)

    def test_scoring_rejects_invalid_settings(self):
        for mode, cost in [("typo", 0), ("hp", -1), ("hp", float("nan"))]:
            with self.assertRaises(ValueError):
                episode.Scoring(mode, cost)

    def test_shared_play_records_identity_and_handles_unlearned_choices(self):
        from agents.nn.encode import Vocab
        from agents.nn.model import Net
        initial = dict(boundary="awaiting_choice", legal=[], choice=dict(min=2),
                       obs=dict(player=dict(hp=60, max_hp=75), turn=1,
                                enemies=[dict(hp=20, alive=True)]))
        final = dict(boundary="terminal", legal=[],
                     obs=dict(player=dict(hp=59), turn=3, enemies=[dict(hp=0, alive=False)]))
        class Worker:
            def step(inner, action):
                self.assertEqual(action, dict(type="choose", picks=[0, 1]))
                return final
        fight = dict(name="arbitrary label", kind="monster", spec=dict(character="CHARACTER.DEFECT"))
        with patch.object(episode, "start", return_value=initial):
            steps, result = episode.play(Worker(), Net(vocab_size=8, d=8, layers=1), Vocab(),
                                         fight, "SEED", scoring=self.scoring)
        self.assertEqual(steps, [])
        self.assertEqual(result["character"], "DEFECT")
        self.assertEqual(result["fight"], "arbitrary label")
        self.assertEqual(result["hp_lost"], 1)
        self.assertAlmostEqual(result["reward"], self.result()["reward"])

    def test_acceptance_includes_discarded_long_failures(self):
        counts = metrics.acceptance_summary([(True, self.result()),
                                          (False, self.result(turn=31, terminal=False, alive=True))])
        self.assertEqual(counts["win/1-5"], dict(accepted=1, dropped=0))
        self.assertEqual(counts["timeout/21+"], dict(accepted=0, dropped=1))

    def test_best_saves_evaluated_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "snapshots").mkdir()
            torch.save(dict(state=dict(weight=torch.tensor([340.])), it=340),
                       out / "snapshots" / "it00340.pt")
            torch.save(dict(state=dict(weight=torch.tensor([430.])), it=430), out / "ckpt.pt")
            train.save_best(out, dict(version=340, heldout_reward=0.9))
            best = torch.load(out / "best.pt", weights_only=False)
            self.assertEqual(best["state"]["weight"].item(), 340)
            self.assertEqual(best["it"], best["eval"]["version"])

    def test_summary_reports_timeout_and_length_tail(self):
        rs = [dict(self.result(), kind="weak", character="DEFECT", fight="display label", hp_lost=1),
              dict(self.result(turn=31, terminal=False, alive=True), kind="weak", character="DEFECT", fight="display label", hp_lost=0)]
        summary = metrics.summary(rs)
        self.assertEqual(set(summary["by_char"]), {"DEFECT"})
        self.assertEqual(summary["timeouts"], 1)
        self.assertEqual(summary["turn_p95"], 31)
        self.assertEqual(summary["hp_lost_wins"], 1)

    def test_card_counts_offer_each_id_once_per_decision(self):
        state = dict(obs=dict(hand=[dict(id="STRIKE"), dict(id="STRIKE"), dict(id="ZAP")]),
                     legal=[dict(type="play", hand=0), dict(type="play", hand=1), dict(type="end_turn")])
        cards = {}
        episode.count_cards(cards, state, 1)
        episode.count_cards(cards, state, 2)
        self.assertEqual(cards, {"STRIKE": [2, 1]})  # an unplayable Zap is never offered

    def test_vocabulary_transfer_preserves_learned_weights(self):
        from agents.nn.model import Net
        from agents.nn.encode import Vocab
        vocab = Vocab({"card:STRIKE": 2, "card:DEFEND": 3})
        self.assertEqual(vocab("card:NEW"), 4)
        old = Net(vocab_size=8, d=8, layers=1)
        new = Net(vocab_size=12, d=8, layers=1)
        train.transfer_weights(new, old.state_dict())
        self.assertTrue(torch.equal(old.ids.weight, new.ids.weight[:8]))
        for key, value in old.state_dict().items():
            if key != "ids.weight":
                self.assertTrue(torch.equal(value, new.state_dict()[key]), key)

    def test_unknown_diagnostics_preserve_frozen_vocabulary(self):
        from agents.nn.encode import Vocab
        vocab = Vocab({"power:KNOWN": 2}, frozen=True)
        self.assertEqual(vocab(None), 0)
        self.assertEqual(vocab("power:KNOWN"), 2)
        self.assertEqual(vocab("power:NEW"), 1)
        self.assertEqual(vocab("power:NEW"), 1)
        self.assertEqual(vocab("card:NEW"), 1)
        self.assertEqual(vocab.table, {"power:KNOWN": 2})
        self.assertEqual(vocab.diagnostics(), dict(lookups=4, unknown=3,
                                                 counts={"power:NEW": 2, "card:NEW": 1}))
