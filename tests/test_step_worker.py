"""The recorded combat again, but played through the worker's ``step`` interface by an ordinary caller.

    python3 -m unittest tests.test_step_worker -v

The caller sees only what a policy would: the observation, the legal inputs, and a pending choice. It translates
each recorded input into that vocabulary, checks it is legal, and sends it. The game's own checkpoints must
still match the recording 49/49. Needs ``./setup.sh``; without it the tests skip.
"""

import re
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from collections import Counter
from statistics import mean

from sts2bridge.fixtures import MCR
from sts2bridge.worker import GAME_DRIVEN, LIB, PROJECT, CombatWorker, WorkerError, recorded_action

from .test_recorded_combat import CHECKPOINT_IDS, excerpt, save_dir_snapshot


def play_recording(worker, tape):
    """Load the fight and step every caller-driven tape event through the worker. Returns the replies in order:
    [(event or None for the load, action sent, boundary state before it, reply)]."""
    state = worker.load(MCR)
    trace = [(None, None, None, state)]
    for event in tape["events"]:
        if event["kind"] in GAME_DRIVEN:
            continue
        action = recorded_action(event, state)
        before, state = state, worker.step(action)
        trace.append((event, action, before, state))
    return trace


@unittest.skipUnless(LIB.exists() and shutil.which("dotnet"), "run ./setup.sh first")
class StepWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saves_before = save_dir_snapshot()
        subprocess.run(["dotnet", "build", "-v", "q", "-c", "Release", str(PROJECT)], check=True, capture_output=True)
        cls.worker = CombatWorker()
        cls.tape = cls.worker.tape(MCR)
        cls.trace = play_recording(cls.worker, cls.tape)
        cls.again = play_recording(cls.worker, cls.tape)  # second load in the same process

    @classmethod
    def tearDownClass(cls):
        cls.worker.close()
        shutil.rmtree(cls.worker.workdir, ignore_errors=True)

    def reported_checkpoints(self, trace):
        return [c for *_, reply in trace for c in reply["checkpoints"]]

    def test_every_checkpoint_matches_through_step(self):
        recorded = {c["id"]: (c["hash"], _normalise(c["context"])) for c in self.tape["checkpoints"]}
        local = self.reported_checkpoints(self.trace)
        self.assertEqual([c["id"] for c in local], CHECKPOINT_IDS)
        mismatched = [c for c in local if recorded[c["id"]] != (c["hash"], _normalise(c["context"]))]
        self.assertEqual(mismatched, [])

    def test_terminal_state_matches_the_recordings(self):
        final = self.trace[-1][3]
        run_end = excerpt()["combat"]["boundaries"][-1]["state"]
        self.assertEqual(final["boundary"], "terminal")
        self.assertEqual(final["obs"]["player"]["hp"], run_end["hp"])
        self.assertEqual([(e["id"], e["hp"]) for e in final["obs"]["enemies"]], [("MONSTER.THE_INSATIABLE", 90)])

    def test_every_recorded_input_was_legal_where_it_was_played(self):
        for event, action, before, _ in self.trace[1:]:
            with self.subTest(event=event["i"]):
                expected = "awaiting_choice" if action["type"] == "choose" else "awaiting_input"
                self.assertEqual(before["boundary"], expected)
                self.assertIn(action, before["legal"])

    def test_boundaries(self):
        # 27 caller inputs: the six ReadyToBeginEnemyTurnActions and the resume after the choice are the game's.
        steps = self.trace[1:]
        self.assertEqual(len(steps), 34 - 7)
        self.assertEqual(self.trace[0][3]["boundary"], "awaiting_input")
        by_event = {event["i"]: reply for event, _, _, reply in steps}
        hologram = by_event[24]
        self.assertEqual(hologram["boundary"], "awaiting_choice")
        self.assertEqual((hologram["choice"]["min"], hologram["choice"]["max"]), (1, 1))
        self.assertIn("CARD.COOLHEADED", [o["id"] for o in hologram["choice"]["options"]])
        self.assertEqual({k: hologram["choice"][k] for k in ("screen", "prompt", "source")},
                         {"screen": "FromCombatPile", "prompt": "HOLOGRAM.selectionScreenPrompt", "source": "CARD.HOLOGRAM"})
        self.assertEqual(by_event[32]["boundary"], "terminal")  # the last end turn; the enemy turn kills the player
        self.assertEqual({r["boundary"] for i, r in by_event.items() if i not in (24, 32)}, {"awaiting_input"})

    def test_the_game_queues_its_own_turn_transitions(self):
        # On the singleplayer net service CombatManager enqueues ReadyToBeginEnemyTurnAction after phase one of
        # every end turn, and resumes the choice-paused action itself. The worker queues neither.
        by_game = Counter()
        for event, _, _, reply in self.trace[1:]:
            names = Counter(reply["enqueued_by_game"])
            if event["kind"] == "end_turn":
                self.assertEqual(names["ReadyToBeginEnemyTurnAction"], 1, event)
            by_game += names
        recorded = sum(e["kind"] == "ready_to_begin_enemy_turn" for e in self.tape["events"])
        self.assertEqual(by_game["ReadyToBeginEnemyTurnAction"], recorded)

    def test_reloading_in_the_same_worker_reproduces_the_fight(self):
        self.assertEqual([r["state_hash"] for *_, r in self.again], [r["state_hash"] for *_, r in self.trace])
        self.assertEqual(self.reported_checkpoints(self.again), self.reported_checkpoints(self.trace))

    def test_a_different_input_takes_a_different_path(self):
        # Negative control, and the point of the interface: end turn 1 immediately instead of playing Modded.
        state = self.worker.load(MCR)
        recorded = {c["id"]: c["hash"] for c in self.tape["checkpoints"]}
        state = self.worker.step({"type": "end_turn"})
        self.assertEqual((state["boundary"], state["obs"]["turn"]), ("awaiting_input", 2))
        diverged = [c for c in state["checkpoints"] if recorded.get(c["id"]) != c["hash"]]
        self.assertTrue(diverged, "skipping every turn-1 play left every checkpoint unchanged")

    def test_illegal_inputs_are_refused_without_touching_state(self):
        state = self.worker.load(MCR)
        for bad in ({"type": "play", "hand": 99}, {"type": "choose", "picks": [0]}, {"type": "potion", "slot": 0}):
            with self.subTest(bad=bad), self.assertRaises(WorkerError):
                self.worker.step(bad)
        self.assertEqual(self.worker.observe()["state_hash"], state["state_hash"])

    def test_costs_are_reported_separately(self):
        load_ms = self.trace[0][3]["ms"]["load"]
        step_ms = [r["ms"]["step"] for *_, r in self.trace[1:]]
        print(f"\nworker boot {self.worker.boot_ms:.0f} ms, combat load {load_ms} ms, "
              f"step mean {mean(step_ms):.1f} ms / max {max(step_ms):.1f} ms over {len(step_ms)} inputs")
        self.assertLess(load_ms, 5_000)
        self.assertLess(mean(step_ms), 1_000)

    def test_real_saves_untouched(self):
        self.assertEqual(save_dir_snapshot(), self.saves_before)

    def test_boot_reports_the_game_library_actually_resolved(self):
        expected = Path(os.environ.get("STS2_LIB") or LIB.parent).resolve()
        self.assertEqual(Path(self.worker.game_lib), expected)


def _normalise(context):
    # PlayCardAction.ToString embeds a per-process card instance number, e.g. "CARD.MODDED (37885600)".
    return re.sub(r"\(\d+\)", "(#)", context or "")


if __name__ == "__main__":
    unittest.main()
