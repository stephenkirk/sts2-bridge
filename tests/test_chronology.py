"""The .run chronology reader: floors, rooms, actions, shops and relic history from a run's history file.

    python3 -m unittest tests.test_chronology -v

Pure Python against a synthetic run; no ``./setup.sh`` needed.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from sts2bridge.chronology import main, reconstruct_run, render_text


def sample_run() -> dict:
    def stats(**actions: object) -> dict:
        return {
            "player_id": 1,
            "current_hp": 50,
            "max_hp": 60,
            "current_gold": 99,
            "damage_taken": 0,
            "hp_healed": 0,
            "max_hp_gained": 0,
            "max_hp_lost": 0,
            "gold_gained": 0,
            "gold_lost": 0,
            "gold_spent": 0,
            "gold_stolen": 0,
            **actions,
        }

    def point(point_type: str, room_type: str, point_stats: dict) -> dict:
        return {
            "map_point_type": point_type,
            "rooms": [{"room_type": room_type, "turns_taken": 0}],
            "player_stats": [point_stats],
        }

    return {
        "schema_version": 9,
        "build_id": "vTEST",
        "platform_type": "steam",
        "game_mode": "standard",
        "seed": "seed",
        "start_time": 123,
        "run_time": 456,
        "ascension": 10,
        "win": False,
        "was_abandoned": False,
        "killed_by_encounter": "ENCOUNTER.TEST",
        "killed_by_event": "NONE.NONE",
        "acts": ["ACT.ONE", "ACT.TWO", "ACT.THREE"],
        "map_point_history": [[
            point(
                "ancient",
                "event",
                stats(relic_choices=[
                    {"choice": "RELIC.WINGED_BOOTS", "was_picked": True}
                ], ancient_choice=[{
                    "title": {"key": "WINGED_BOOTS.title", "table": "relics"},
                    "was_chosen": True,
                }]),
            ),
            point("elite", "elite", stats()),
            {
                "map_point_type": "unknown",
                "rooms": [
                    {"room_type": "event", "model_id": "EVENT.TEST", "turns_taken": 0},
                    {"room_type": "monster", "model_id": "ENCOUNTER.TEST", "turns_taken": 2},
                ],
                "player_stats": [stats(relics_removed=["RELIC.WINGED_BOOTS"])],
            },
        ]],
        "players": [{
            "id": 1,
            "character": "CHARACTER.TEST",
            "relics": [{"id": "RELIC.STARTER", "floor_added_to_deck": 1}],
        }],
    }


class ReconstructRunTests(unittest.TestCase):
    def test_addresses_rooms_actions_and_relic_history(self) -> None:
        result = reconstruct_run(sample_run(), "sample.run", "abc")
        self.assertEqual([floor["floor"] for floor in result["floors"]], [1, 2, 3])
        self.assertEqual(result["acts"][0]["floor_end"], 3)
        self.assertFalse(result["acts"][1]["reached"])
        self.assertEqual(len(result["floors"][2]["rooms"]), 2)
        self.assertEqual(result["floors"][2]["room_types"], ["event", "monster"])
        self.assertTrue(result["floors"][2]["is_combat"])
        self.assertIn("relics_removed", result["floors"][2]["actions"])
        self.assertEqual(result["player"]["starting_relics"], ["RELIC.STARTER"])
        self.assertEqual(
            result["floors"][1]["relics"]["held_before"],
            ["RELIC.STARTER", "RELIC.WINGED_BOOTS"],
        )
        self.assertEqual(result["floors"][2]["relics"]["held_after"], ["RELIC.STARTER"])
        self.assertEqual(result["warnings"], [])

    def test_text_is_compact_and_can_expand_relics(self) -> None:
        result = reconstruct_run(sample_run(), "sample.run", "abc")
        output = render_text(result, show_relics=True)
        self.assertIn("-- Act 1: ACT.ONE --", output)
        self.assertIn("unknown", output)
        self.assertIn("event:TEST + monster:TEST", output)
        self.assertIn("ancient=WINGED_BOOTS", output)
        self.assertIn("relics after: STARTER, WINGED_BOOTS", output)

    def test_shop_relic_is_not_counted_as_two_acquisitions(self) -> None:
        run = sample_run()
        point = run["map_point_history"][0][0]
        point["map_point_type"] = "shop"
        point["rooms"] = [{"room_type": "shop", "turns_taken": 0}]
        point["player_stats"][0].update({
            "gold_spent": 231,
            "cards_gained": [
                {"id": "CARD.SLICE"},
                {"id": "CARD.MADNESS"},
            ],
            "bought_colorless": ["CARD.MADNESS"],
            "relic_choices": [
                {"choice": "RELIC.WINGED_BOOTS", "was_picked": True},
                {"choice": "RELIC.ANCHOR", "was_picked": False},
            ],
            "bought_relics": ["RELIC.WINGED_BOOTS"],
            "bought_potions": ["POTION.FIRE_POTION"],
            "potion_choices": [
                {"choice": "POTION.FIRE_POTION", "was_picked": True}
            ],
            "cards_removed": [{"id": "CARD.STRIKE_TEST"}],
            "potion_discarded": ["POTION.SWIFT_POTION"],
        })
        result = reconstruct_run(run, "sample.run", "abc")
        floor = result["floors"][0]
        relics = floor["relics"]
        self.assertEqual(relics["gained"], ["RELIC.WINGED_BOOTS"])
        self.assertEqual(relics["gain_sources"][0]["source"], "shop_purchase")
        self.assertEqual(floor["shop"]["gold_spent"], 231)
        self.assertEqual(
            floor["shop"]["card_ids_bought"], ["CARD.SLICE", "CARD.MADNESS"]
        )
        self.assertEqual(
            floor["shop"]["colorless_card_ids_bought"], ["CARD.MADNESS"]
        )
        self.assertEqual(
            floor["shop"]["relic_ids_bought"], ["RELIC.WINGED_BOOTS"]
        )
        self.assertEqual(
            floor["shop"]["potion_ids_bought"], ["POTION.FIRE_POTION"]
        )
        self.assertEqual(
            floor["shop"]["card_ids_removed"], ["CARD.STRIKE_TEST"]
        )
        self.assertEqual(
            floor["shop"]["potion_ids_discarded"], ["POTION.SWIFT_POTION"]
        )
        self.assertEqual(
            len(floor["shop"]["recorded_inventory_after"]["relic_choices"]), 2
        )
        text = render_text(result)
        self.assertIn(
            "shop spent=231 buy cards=SLICE,MADNESS relics=WINGED_BOOTS "
            "potions=FIRE_POTION",
            text,
        )

    def test_cli_jsonl_emits_one_self_describing_record_per_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.run"
            path.write_text(json.dumps(sample_run()))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main([str(path), "--format", "jsonl"])
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(status, 0)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["format_version"], 1)
        self.assertEqual(records[0]["floor"], 1)
        self.assertEqual(records[0]["run"]["start_time"], 123)


if __name__ == "__main__":
    unittest.main()
