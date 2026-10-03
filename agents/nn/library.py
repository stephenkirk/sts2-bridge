"""Build an Act 1 combat curriculum from the player's completed .run saves, with explicit inferred state.

Uses sts2bridge.chronology for each run and the worker catalog for starter decks.
Never writes to game saves.
The exported specs are training scenarios, not historical replays.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from sts2bridge import CombatWorker, chronology as chronology_module, reconstruct_run


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def card_key(card):
    return (card["id"], card.get("current_upgrade_level", 0),
            json.dumps(card.get("enchantment"), sort_keys=True), json.dumps(card.get("props"), sort_keys=True))


class ReconstructionError(ValueError):
    pass


def take_card(deck, snapshot, floor=None, upgraded=False):
    """Prefer recorded copy/state, retaining ambiguity when distinct copies qualify."""
    cid = snapshot if isinstance(snapshot, str) else snapshot["id"]
    candidates = [i for i, c in enumerate(deck) if c["id"] == cid and
                  (not upgraded or c.get("current_upgrade_level", 0) > 0)]
    if isinstance(snapshot, dict):
        exact = [i for i in candidates if card_key(deck[i]) == card_key(snapshot)]
        if exact:
            candidates = exact
    if floor is not None:
        dated = [i for i in candidates if deck[i].get("floor_added_to_deck") == floor]
        if dated:
            candidates = dated
    if not candidates:
        raise ReconstructionError(f"no matching card for {cid}")
    ambiguous = len({card_key(deck[i]) for i in candidates}) > 1
    return deck.pop(candidates[0]), ambiguous


def apply_cards(deck, actions, floor):
    """Sparse action arrays are a same-floor bundle; their internal order is inferred."""
    ambiguities = []
    for c in actions.get("cards_removed", []):
        _, ambiguous = take_card(deck, c)
        if ambiguous:
            ambiguities.append("removed_copy")
    for change in actions.get("cards_transformed", []):
        _, ambiguous = take_card(deck, change["original_card"])
        if ambiguous:
            ambiguities.append("transformed_copy")
        card = deepcopy(change["final_card"])
        card.setdefault("floor_added_to_deck", floor)
        deck.append(card)
    for c in actions.get("cards_gained", []):
        card = deepcopy(c)
        card.setdefault("floor_added_to_deck", floor)
        deck.append(card)
    for field, delta in (("upgraded_cards", 1), ("downgraded_cards", -1)):
        for cid in actions.get(field, []):
            card, ambiguous = take_card(deck, cid, upgraded=delta < 0)
            if ambiguous:
                ambiguities.append(field + "_copy")
            card["current_upgrade_level"] = card.get("current_upgrade_level", 0) + delta
            deck.append(card)
    for entry in actions.get("cards_enchanted", []):
        # The recorded card is after enchantment. Match its prior copy without that field.
        before = deepcopy(entry["card"])
        before.pop("enchantment", None)
        card, ambiguous = take_card(deck, before)
        if ambiguous:
            ambiguities.append("enchanted_copy")
        enchantment = entry["card"].get("enchantment")
        if not isinstance(enchantment, dict) or "amount" not in enchantment:
            raise ReconstructionError("enchantment amount missing")
        card["enchantment"] = deepcopy(enchantment)
        deck.append(card)
    return ambiguities


def starting_decks(catalog):
    return {character: [dict(id=card, floor_added_to_deck=1) for card in deck]
            for character, deck in catalog["starting_deck"].items()}


def reconstruct_fights(run, source_id, chronology, starter, catalog):
    """Forward replay prevents terminal upgrades/enchantments from leaking into earlier decks."""
    deck = deepcopy(starter)
    # The game adds the starting curse from this ascension on.
    if run["ascension"] >= catalog["ascenders_bane_ascension"]:
        deck.append(dict(id="CARD.ASCENDERS_BANE", floor_added_to_deck=1))
    fights, ambiguities = [], []
    known = {e["id"]: e for e in catalog["encounters"]}
    character = run["players"][0]["character"]
    previous = None
    normal_combat_index = 0
    history = []
    for floor in chronology["floors"]:
        rooms = floor["rooms"]
        number = floor["floor"]
        if floor["act"] == 1 and rooms and rooms[0].get("room_type") == "monster":
            normal_combat_index += 1
        # Events can mutate the loadout before opening combat on the same node: omit those.
        if floor["act"] == 1 and previous and len(rooms) == 1 and rooms[0].get("room_type") in ("monster", "elite"):
            room = rooms[0]
            encounter = known.get(room.get("model_id"))
            hp, max_hp = previous.get("current_hp"), previous.get("max_hp")
            if encounter and encounter["act_index"] == 0 and floor["act_id"] in encounter["acts"] and hp and max_hp:
                # Keep ownership from the chronology's before-node relics; never copy terminal counters.
                relics = [dict(id=rid) for rid in floor["relics"]["held_before"]]
                floor_band = "2-5" if number <= 5 else "6-10" if number <= 10 else "11+"
                encounter_pool = "elite" if room["room_type"] == "elite" else "weak" if encounter["weak"] else "regular"
                spec = dict(character=character, ascension=run["ascension"], encounter=room["model_id"], act=0,
                            player=dict(deck=deepcopy(deck), relics=relics, potions=[], current_hp=hp, max_hp=max_hp,
                                        gold=previous.get("current_gold", 0)),
                            run=dict(map_point_history=[deepcopy(history)]))
                fingerprint = digest(dict(character=character, deck=sorted(card_key(c) for c in deck),
                                          relics=sorted(r["id"] for r in relics)))
                # How the run itself went, for comparison only: its own shuffle, with whatever potions it held.
                point = run["map_point_history"][floor["act"] - 1][floor["act_floor"] - 1]
                after = floor["player_state"].get("current_hp")
                you = dict(hp_change=None if after is None else hp - after, won=bool(after),
                           damage_taken=point["player_stats"][0].get("damage_taken"),
                           turns=point["rooms"][0].get("turns_taken"))
                fights.append(dict(name=f"{character.split('.')[1]}/{room['model_id'].split('.')[1]}/{source_id[:12]}-F{number}",
                                   kind=room["room_type"], floor_band=floor_band, encounter_pool=encounter_pool,
                                   normal_combat_index=normal_combat_index if room["room_type"] == "monster" else None,
                                   spec=spec, loadout_id=fingerprint,
                                   source=dict(run_id=source_id, floor=number, act=1, build=run["build_id"],
                                               content_sha256=chronology.get("source", {}).get("sha256"),
                                               outcome="win" if run["win"] else "abandoned" if run["was_abandoned"] else "loss",
                                               state="inferred", copy_ambiguities=list(ambiguities), you=you,
                                               assumptions=["same-floor card mutations applied as a bundle",
                                                            "entry HP and gold carried from prior node",
                                                            "relic props reset to game defaults", "empty potion belt",
                                                            "fresh seed and RNG, not a historical replay"])))
        ambiguities.extend(apply_cards(deck, floor["actions"], number))
        previous = floor["player_state"]
        # Keep the serialized history for game floor/encounter context; it is not a policy input.
        history.append(run["map_point_history"][floor["act"] - 1][floor["act_floor"] - 1])
    if Counter(card_key(c) for c in deck) != Counter(card_key(c) for c in run["players"][0]["deck"]):
        raise ReconstructionError("terminal deck did not reconcile")
    if chronology["warnings"]:
        raise ReconstructionError("relic chronology warnings: " + "; ".join(chronology["warnings"]))
    return fights


def split_runs(groups):
    """Stratify by character and win/loss, grouping duplicate saves of the same run."""
    strata = defaultdict(list)
    for rid, fights in groups.items():
        f = fights[0]
        strata[(f["spec"]["character"], f["source"]["outcome"])].append(rid)
    split = {}
    for ids in strata.values():
        ids.sort(key=lambda rid: digest(["split-v1", rid]))
        n = len(ids)
        held = max(1, round(n * 0.15)) if n >= 3 else 0
        for i, rid in enumerate(ids):
            split[rid] = "test" if i < held else "validation" if i < 2 * held else "train"
    return split


def select(groups, per_bucket=2):
    assignment = split_runs(groups)
    splits = {k: [] for k in ("train", "validation", "test")}
    for rid, fights in groups.items():
        buckets = defaultdict(list)
        for f in fights:
            buckets[(f["floor_band"], f["encounter_pool"])].append(f)
        for bucket in buckets.values():
            bucket.sort(key=lambda f: digest(["select-v1", f["name"]]))
            selected, seen = [], set()
            for f in bucket:
                identity = (f["loadout_id"], f["spec"]["encounter"])
                if identity not in seen:
                    selected.append(f)
                    seen.add(identity)
                if len(selected) >= per_bucket:
                    break
            splits[assignment[rid]].extend(selected)
    # Runs stay together; exclude exact inventory overlaps from the new branch's training.
    seen, removed = set(), Counter()
    for name in ("test", "validation", "train"):
        prior = set(seen)
        keep = []
        for f in splits[name]:
            if f["loadout_id"] in prior:
                removed[name] += 1
            else:
                keep.append(f)
        splits[name] = keep
        seen.update(f["loadout_id"] for f in keep)
    add_weights(splits["train"])
    return splits, dict(removed)


def add_weights(fights):
    """Equal characters, then 20% weak / 50% regular / 30% elite where available.

    Within a pool: equal floor bands -> source runs -> scenarios. Floor bands are sampling bins,
    never an inference of which encounter pool the game used.
    """
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for f in fights:
        tree[f["spec"]["character"]][f["encounter_pool"]][f["floor_band"]][f["source"]["run_id"]].append(f)
    pool_weights = dict(weak=0.2, regular=0.5, elite=0.3)
    for pools in tree.values():
        total = sum(pool_weights[p] for p in pools)
        for pool, bands in pools.items():
            for runs in bands.values():
                for fs in runs.values():
                    for f in fs:
                        f["sample_weight"] = pool_weights[pool] / (total * len(tree) * len(bands) * len(runs) * len(fs))


def build(history, catalog, build_id="v0.111.0", ascension=10):
    starters = starting_decks(catalog)
    groups, skipped, seen, audited = {}, Counter(), set(), []
    files = sorted(history.glob("*.run"))
    for path in files:
        run = json.loads(path.read_text())
        if len(run.get("players", [])) != 1 or run.get("game_mode") != "standard" or run.get("modifiers"):
            skipped["not unmodified solo standard"] += 1
            continue
        if run.get("build_id") != build_id:
            skipped["different build"] += 1
            continue
        if run.get("ascension") != ascension:
            skipped["different ascension"] += 1
            continue
        player = run["players"][0]
        rid = digest([run["seed"], run["start_time"], player["character"]])
        if rid in seen:
            skipped["duplicate run"] += 1
            continue
        seen.add(rid)
        try:
            chronology = reconstruct_run(run, path.name, digest(run))
            fs = reconstruct_fights(run, rid, chronology, starters[player["character"]], catalog)
        except (ReconstructionError, ValueError, KeyError) as exc:
            skipped["reconstruction rejected"] += 1
            audited.append(dict(run_id=rid, reason=str(exc)))
            continue
        if fs:
            groups[rid] = fs
    splits, overlap_removed = select(groups)
    summary = {name: dict(fights=len(fs), runs=len({f["source"]["run_id"] for f in fs}),
                          loadouts=len({f["loadout_id"] for f in fs}),
                          by_character=dict(Counter(f["spec"]["character"] for f in fs)),
                          by_floor_band_pool=dict(Counter(f["floor_band"] + "/" + f["encounter_pool"] for f in fs)))
               for name, fs in splits.items()}
    return dict(format_version=1, pool="library", build=build_id, ascension=ascension,
                provenance="inferred scenarios from dated run actions; not historical replays",
                chronology_reader_sha256=hashlib.sha256(Path(chronology_module.__file__).read_bytes()).hexdigest(),
                input_runs=len(files), reconstructed_runs=len(groups), skipped=dict(skipped),
                reconstruction_rejections=audited, overlap_removed=overlap_removed, summary=summary, splits=splits)


def validate(report):
    """Contract checks only: no policy scores or held-out fight outcomes are inspected."""
    from .encode import Vocab, encode
    checks = []
    with CombatWorker() as worker:
        for split, fs in report["splits"].items():
            for fight in fs:
                try:
                    state = worker.start(dict(fight["spec"], seed="LIBRARY-CONTRACT-CHECK"), hashes=False, reuse_map=True)
                    if state["boundary"] == "terminal" or not state["legal"] or state["obs"]["player"]["hp"] <= 0:
                        raise ValueError("scenario has no live initial policy decision")
                    encode(state, Vocab())
                    checks.append(dict(split=split, fight=fight["name"], ok=True))
                except Exception as exc:
                    checks.append(dict(split=split, fight=fight["name"], ok=False, error=str(exc)))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", type=Path, help="completed-run library; auto-discover a single local Steam profile by default")
    ap.add_argument("--build", default="v0.111.0")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, help="cached bridge catalog with starter decks; otherwise query the native worker")
    ap.add_argument("--validate", action="store_true", help="check every selected scenario starts and can be encoded; no training")
    args = ap.parse_args()
    if args.history is None:
        roots = [p for p in (Path.home() / "Library/Application Support/SlayTheSpire2/steam").glob("*/profile*/saves/history")
                 if next(p.glob("*.run"), None) is not None]
        if len(roots) != 1:
            ap.error("specify --history: expected one nonempty local run library")
        args.history = roots[0]
    if args.catalog:
        catalog = json.loads(args.catalog.read_text())
    else:
        with CombatWorker() as worker:
            catalog = worker.catalog()
    report = build(args.history, catalog, args.build, args.ascension)
    if args.validate:
        report["contract_checks"] = validate(report)
        errors = [c for c in report["contract_checks"] if not c["ok"]]
        if errors:
            ap.error("scenario contract failures: " + json.dumps(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("input_runs", "reconstructed_runs", "skipped", "overlap_removed", "summary")}, indent=2))


if __name__ == "__main__":
    main()
