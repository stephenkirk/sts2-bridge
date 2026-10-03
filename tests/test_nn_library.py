"""Curriculum provenance, split isolation and state reconstruction; no game workers."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from agents.nn import fights, library


class LibraryTests(unittest.TestCase):
    def scenario(self, rid, pool="weak", band="2-5", character="CHARACTER.DEFECT", serial=0):
        return dict(name=f"{character}/{rid}/{serial}", source=dict(run_id=rid, outcome="win"),
                    spec=dict(character=character, encounter=f"ENCOUNTER.{pool}"), kind="monster",
                    encounter_pool=pool, floor_band=band, loadout_id=f"{rid}-{serial}")

    def fixture(self, actions=None, extra_rooms=None):
        starter = [dict(id="CARD.STRIKE_DEFECT", floor_added_to_deck=1)]
        gained = dict(id="CARD.ZAP", floor_added_to_deck=2)
        rooms = [[dict(room_type="event")], [dict(room_type="monster", model_id="ENCOUNTER.SLIMES_WEAK")],
                 [dict(room_type="monster", model_id="ENCOUNTER.SLIMES_WEAK")]]
        if extra_rooms:
            rooms[1] += extra_rooms
        mutations = [{}, dict(cards_gained=[dict(id="CARD.ZAP")]), dict(upgraded_cards=["CARD.ZAP"])]
        if actions:
            mutations = actions
        points, floors = [], []
        for i, (rs, act) in enumerate(zip(rooms, mutations), 1):
            state = dict(current_hp=50-i, max_hp=75, current_gold=99)
            points.append(dict(rooms=rs, player_stats=[dict(state, **act)]))
            floors.append(dict(floor=i, act=1, act_floor=i, act_id="ACT.OVERGROWTH", rooms=rs,
                               player_state=state, actions=act, relics=dict(held_before=["RELIC.CRACKED_CORE"])))
        final = starter + [dict(id="CARD.ASCENDERS_BANE", floor_added_to_deck=1),
                           dict(gained, current_upgrade_level=1)]
        run = dict(ascension=10, build_id="v0.111.0", win=False, was_abandoned=False,
                   map_point_history=[points], players=[dict(character="CHARACTER.DEFECT", deck=final)])
        catalog = dict(encounters=[dict(id="ENCOUNTER.SLIMES_WEAK", act_index=0, acts=["ACT.OVERGROWTH"], weak=True)],
                       starting_deck={"CHARACTER.DEFECT": ["CARD.STRIKE_DEFECT"]}, ascenders_bane_ascension=5)
        return run, dict(floors=floors, warnings=[]), starter, catalog

    def test_builder_needs_only_run_saves_and_catalog(self):
        run, _, starter, catalog = self.fixture()
        run.update(seed="TEST", start_time=1, game_mode="standard", acts=["ACT.OVERGROWTH"])
        run["players"][0].update(id=1, relics=[dict(id="RELIC.CRACKED_CORE", floor_added_to_deck=1)])
        for point in run["map_point_history"][0]:
            point["map_point_type"] = "monster"
            point["player_stats"][0]["player_id"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "history"
            history.mkdir()
            (history / "test.run").write_text(json.dumps(run))
            report = library.build(history, catalog)
        self.assertEqual(report["reconstructed_runs"], 1)
        self.assertEqual(report["reconstruction_rejections"], [])
        scenarios = [f for fs in report["splits"].values() for f in fs]
        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0]["source"]["state"], "inferred")

    def test_forward_state_excludes_reward_and_future_upgrade(self):
        run, chronology, starter, catalog = self.fixture()
        fs = library.reconstruct_fights(run, "run", chronology, starter, catalog)
        self.assertNotIn("CARD.ZAP", [c["id"] for c in fs[0]["spec"]["player"]["deck"]])
        zap = next(c for c in fs[1]["spec"]["player"]["deck"] if c["id"] == "CARD.ZAP")
        self.assertEqual(zap.get("current_upgrade_level", 0), 0)
        self.assertEqual(fs[0]["spec"]["player"]["current_hp"], 49)
        self.assertEqual(fs[0]["source"]["state"], "inferred")
        self.assertEqual(fs[0]["spec"]["player"]["potions"], [])
        # Entered at 49 (floor 1's state) and left at 48 (floor 2's).
        self.assertEqual(fs[0]["source"]["you"], dict(hp_change=1, won=True, damage_taken=None, turns=None))

    def test_event_combat_is_excluded_without_advancing_queue(self):
        run, chronology, starter, catalog = self.fixture(extra_rooms=[dict(room_type="event")])
        fs = library.reconstruct_fights(run, "run", chronology, starter, catalog)
        self.assertEqual(len(fs), 1)
        # The first room is monster, so this multi-room point still advances the normal queue.
        self.assertEqual(fs[0]["normal_combat_index"], 2)
        chronology["floors"][1]["rooms"].reverse()
        fs = library.reconstruct_fights(run, "run", chronology, starter, catalog)
        self.assertEqual(fs[0]["normal_combat_index"], 1)

    def test_weak_pool_is_catalog_membership_not_floor_band(self):
        run, chronology, starter, catalog = self.fixture()
        # This same weak encounter can be encountered after intervening non-combat nodes.
        inserted = []
        points = []
        for n in range(2, 9):
            f = deepcopy(chronology["floors"][0])
            f.update(floor=n, act_floor=n)
            inserted.append(f)
            points.append(deepcopy(run["map_point_history"][0][0]))
        for f, n in zip(chronology["floors"][1:], (9, 10)):
            f.update(floor=n, act_floor=n)
        chronology["floors"][1:1] = inserted
        run["map_point_history"][0][1:1] = points
        run["players"][0]["deck"][-1]["floor_added_to_deck"] = 9
        fs = library.reconstruct_fights(run, "run", chronology, starter, catalog)
        self.assertEqual(fs[0]["encounter_pool"], "weak")
        self.assertEqual(fs[0]["floor_band"], "6-10")
        self.assertEqual(fs[0]["normal_combat_index"], 1)
        self.assertEqual(len(fs[0]["spec"]["run"]["map_point_history"][0]), 8)

    def test_enchantment_amount_is_preserved(self):
        deck = [dict(id="CARD.DEFEND", current_upgrade_level=1)]
        library.apply_cards(deck, dict(cards_enchanted=[dict(card=dict(id="CARD.DEFEND", current_upgrade_level=1,
                                                                     enchantment=dict(id="ENCHANTMENT.REPLAY", amount=2)))]), 3)
        self.assertEqual(deck[0]["enchantment"], dict(id="ENCHANTMENT.REPLAY", amount=2))

    def test_terminal_mismatch_rejects_run(self):
        run, chronology, starter, catalog = self.fixture()
        run["players"][0]["deck"].append(dict(id="CARD.FUTURE_UNKNOWN"))
        with self.assertRaises(library.ReconstructionError):
            library.reconstruct_fights(run, "run", chronology, starter, catalog)

    def test_split_keeps_runs_together_and_excludes_inventory_leakage(self):
        groups = {str(i): [self.scenario(str(i), serial=j) for j in range(5)] for i in range(20)}
        # Deliberately repeat one exact inventory across all source runs.
        for fs in groups.values():
            fs[0]["loadout_id"] = "shared"
        splits, removed = library.select(groups)
        run_sets = [{f["source"]["run_id"] for f in splits[s]} for s in ("train", "validation", "test")]
        inventories = [{f["loadout_id"] for f in splits[s]} for s in ("train", "validation", "test")]
        for sets in (run_sets, inventories):
            for i in range(3):
                for j in range(i):
                    self.assertFalse(sets[i] & sets[j])
        self.assertTrue(all(splits.values()))
        per_run = defaultdict(int)
        for fs in splits.values():
            for f in fs:
                per_run[f["source"]["run_id"]] += 1
        self.assertLessEqual(max(per_run.values()), 2)
        again, _ = library.select(deepcopy(groups))
        self.assertEqual(splits, again)

    def test_sampling_balances_characters_and_encounter_pools(self):
        fs = []
        for character, count in (("DEFECT", 20), ("REGENT", 2)):
            for pool in ("weak", "regular", "elite"):
                fs += [self.scenario(f"{character}-{pool}-{i}", pool=pool, character=character) for i in range(count)]
        library.add_weights(fs)
        weights = defaultdict(float)
        for f in fs:
            weights[(f["spec"]["character"], f["encounter_pool"])] += f["sample_weight"]
        self.assertAlmostEqual(sum(weights.values()), 1)
        for character in ("DEFECT", "REGENT"):
            self.assertAlmostEqual(weights[character, "weak"], 0.1)
            self.assertAlmostEqual(weights[character, "regular"], 0.25)
            self.assertAlmostEqual(weights[character, "elite"], 0.15)

    def test_pool_loader_keeps_validation_and_test_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pool.json"
            p.write_text(json.dumps(dict(format_version=1, pool="library", splits=dict(train=["train"], validation=["val"], test=["test"]))))
            self.assertEqual(fights.pool("library", pool_file=p, split="validation"), ["val"])
            self.assertEqual(fights.pool("library", pool_file=p, split="test"), ["test"])
