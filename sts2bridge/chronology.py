"""Reconstruct a floor-by-floor chronology from a completed STS2 .run file."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


FORMAT_VERSION = 1
STATE_FIELDS = {
    "player_id",
    "current_hp",
    "max_hp",
    "current_gold",
    "damage_taken",
    "hp_healed",
    "max_hp_gained",
    "max_hp_lost",
    "gold_gained",
    "gold_lost",
    "gold_spent",
    "gold_stolen",
    "stolen_loot",
    "is_affected_by_fur_coat",
}
RESOURCE_FIELDS = tuple(field for field in STATE_FIELDS if field != "player_id")


class RunFormatError(ValueError):
    """Raised when input is not a supported completed-run shape."""


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RunFormatError(f"{name} must be an array")
    return value


def _load_run(path: str) -> tuple[dict[str, Any], str, str]:
    if path == "-":
        raw = sys.stdin.buffer.read()
        source = "<stdin>"
    else:
        input_path = Path(path)
        raw = input_path.read_bytes()
        source = str(input_path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunFormatError(f"invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RunFormatError("top-level value must be an object")
    return value, source, hashlib.sha256(raw).hexdigest()


def _select_player(run: dict[str, Any], player_id: int | None) -> dict[str, Any]:
    players = _require_list(run.get("players"), "players")
    if player_id is None:
        if len(players) != 1:
            raise RunFormatError(
                "run has multiple players; select one with --player PLAYER_ID"
            )
        player = players[0]
    else:
        player = next(
            (
                candidate
                for candidate in players
                if isinstance(candidate, dict) and candidate.get("id") == player_id
            ),
            None,
        )
        if player is None:
            raise RunFormatError(f"player {player_id} is not present in the run")
    if not isinstance(player, dict) or not isinstance(player.get("id"), int):
        raise RunFormatError("each terminal player must be an object with an integer id")
    return player


def _player_stats(point: dict[str, Any], player_id: int, address: str) -> dict[str, Any]:
    stats = _require_list(point.get("player_stats"), f"{address}.player_stats")
    matches = [
        entry
        for entry in stats
        if isinstance(entry, dict) and entry.get("player_id") == player_id
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RunFormatError(
            f"{address} must contain exactly one player_stats entry for player {player_id}"
        )
    return matches[0]


def _picked_relics(stats: dict[str, Any]) -> list[tuple[str, str]]:
    gains: dict[str, str] = {}
    for choice in stats.get("relic_choices", []):
        if (
            isinstance(choice, dict)
            and choice.get("was_picked") is True
            and isinstance(choice.get("choice"), str)
        ):
            gains[choice["choice"]] = "relic_choice"
    for relic_id in stats.get("bought_relics", []):
        if isinstance(relic_id, str):
            # Shops serialize the purchased relic both as a picked inventory
            # choice and in the explicit purchase array. It is one acquisition.
            gains[relic_id] = "shop_purchase"
    return list(gains.items())


def _counter_items(counter: Counter[str]) -> list[str]:
    return sorted(counter.elements())


def _remove(counter: Counter[str], items: Iterable[str], warnings: list[str]) -> None:
    for item in items:
        if counter[item] <= 0:
            warnings.append(f"could not reverse relic gain {item!r} from terminal state")
            continue
        counter[item] -= 1
        if counter[item] == 0:
            del counter[item]


def _model_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [
        value.get("id") if isinstance(value, dict) else value
        for value in values
        if (isinstance(value, str) or isinstance(value, dict))
        and isinstance(value.get("id") if isinstance(value, dict) else value, str)
    ]


def _shop_summary(
    stats: dict[str, Any], room_types: list[str]
) -> dict[str, Any] | None:
    if "shop" not in room_types:
        return None
    cards_bought = stats.get("cards_gained", [])
    cards_removed = stats.get("cards_removed", [])
    potions_discarded = stats.get("potion_discarded", [])
    return {
        # The save provides only the node total, not a price for each purchase.
        "gold_spent": stats.get("gold_spent", 0),
        "cards_bought": cards_bought,
        "card_ids_bought": _model_ids(cards_bought),
        "colorless_card_ids_bought": _model_ids(stats.get("bought_colorless", [])),
        "relic_ids_bought": _model_ids(stats.get("bought_relics", [])),
        "potion_ids_bought": _model_ids(stats.get("bought_potions", [])),
        "cards_removed": cards_removed,
        "card_ids_removed": _model_ids(cards_removed),
        "potions_discarded": potions_discarded,
        "potion_ids_discarded": _model_ids(potions_discarded),
        # Shop choice arrays describe the recorded post-session inventory, not
        # the exact initial stock. Courier-like restocking can affect it.
        "recorded_inventory_after": {
            "card_choices": stats.get("card_choices", []),
            "relic_choices": stats.get("relic_choices", []),
            "potion_choices": stats.get("potion_choices", []),
        },
    }


def reconstruct_run(
    run: dict[str, Any], source: str, content_sha256: str, player_id: int | None = None
) -> dict[str, Any]:
    """Return an exact-addressed, enriched chronology for one terminal player."""
    player = _select_player(run, player_id)
    selected_id = player["id"]
    acts = _require_list(run.get("acts"), "acts")
    histories = _require_list(run.get("map_point_history"), "map_point_history")
    if len(histories) > len(acts):
        raise RunFormatError("map_point_history contains more acts than acts")

    floors: list[dict[str, Any]] = []
    act_summaries: list[dict[str, Any]] = []
    global_floor = 0
    for act_index, act_id in enumerate(acts):
        history = histories[act_index] if act_index < len(histories) else []
        history = _require_list(history, f"map_point_history[{act_index}]")
        floor_start = global_floor + 1 if history else None
        for point_index, point in enumerate(history):
            if not isinstance(point, dict):
                raise RunFormatError(
                    f"map_point_history[{act_index}][{point_index}] must be an object"
                )
            global_floor += 1
            address = f"map_point_history[{act_index}][{point_index}]"
            stats = _player_stats(point, selected_id, address)
            rooms = _require_list(point.get("rooms"), f"{address}.rooms")
            room_types = [
                room.get("room_type")
                for room in rooms
                if isinstance(room, dict) and isinstance(room.get("room_type"), str)
            ]
            gains = _picked_relics(stats)
            removed = [
                relic_id
                for relic_id in stats.get("relics_removed", [])
                if isinstance(relic_id, str)
            ]
            floors.append(
                {
                    "floor": global_floor,
                    "act": act_index + 1,
                    "act_id": act_id,
                    "act_floor": point_index + 1,
                    "map_point_type": point.get("map_point_type"),
                    "room_types": room_types,
                    "is_combat": any(
                        room_type in {"monster", "elite", "boss"}
                        for room_type in room_types
                    ),
                    "rooms": rooms,
                    "shop": _shop_summary(stats, room_types),
                    "player_state": {
                        key: stats[key] for key in RESOURCE_FIELDS if key in stats
                    },
                    "actions": {
                        key: value
                        for key, value in stats.items()
                        if key not in STATE_FIELDS
                    },
                    "relics": {
                        "gained": [relic_id for relic_id, _ in gains],
                        "gain_sources": [
                            {"id": relic_id, "source": gain_source}
                            for relic_id, gain_source in gains
                        ],
                        "removed": removed,
                    },
                }
            )
        act_summaries.append(
            {
                "act": act_index + 1,
                "act_id": act_id,
                "reached": bool(history),
                "floor_start": floor_start,
                "floor_end": global_floor if history else None,
                "length": len(history),
            }
        )

    final_relics = _require_list(player.get("relics", []), "player.relics")
    explicit_by_floor: dict[int, set[str]] = {
        floor["floor"]: set(floor["relics"]["gained"]) for floor in floors
    }
    for relic in final_relics:
        if not isinstance(relic, dict) or not isinstance(relic.get("id"), str):
            raise RunFormatError("player.relics entries must have a string id")
        added = relic.get("floor_added_to_deck")
        if (
            isinstance(added, int)
            and 1 < added <= len(floors)
            and relic["id"] not in explicit_by_floor[added]
        ):
            floors[added - 1]["relics"]["gained"].append(relic["id"])
            floors[added - 1]["relics"]["gain_sources"].append(
                {"id": relic["id"], "source": "terminal_snapshot"}
            )
            explicit_by_floor[added].add(relic["id"])

    warnings: list[str] = []
    held = Counter(relic["id"] for relic in final_relics)
    for floor in reversed(floors):
        floor["relics"]["held_after"] = _counter_items(held)
        _remove(held, floor["relics"]["gained"], warnings)
        held.update(floor["relics"]["removed"])
        floor["relics"]["held_before"] = _counter_items(held)
    starting_relics = _counter_items(held)

    return {
        "format_version": FORMAT_VERSION,
        "source": {"path": source, "sha256": content_sha256},
        "run": {
            key: run.get(key)
            for key in (
                "schema_version",
                "build_id",
                "platform_type",
                "game_mode",
                "seed",
                "start_time",
                "run_time",
                "ascension",
                "win",
                "was_abandoned",
                "killed_by_encounter",
                "killed_by_event",
            )
        },
        "player": {
            "id": selected_id,
            "character": player.get("character"),
            "starting_relics": starting_relics,
            "final_relics": final_relics,
        },
        "acts": act_summaries,
        "floors": floors,
        "warnings": sorted(set(warnings)),
    }


def _compact_id(value: Any) -> str:
    if not isinstance(value, str):
        return "-" if value is None else str(value)
    return value.split(".", 1)[-1]


def _room_label(room: Any) -> str:
    if not isinstance(room, dict):
        return "?"
    room_type = _compact_id(room.get("room_type"))
    model_id = room.get("model_id")
    return f"{room_type}:{_compact_id(model_id)}" if model_id else room_type


def _chosen_keys(actions: dict[str, Any], field: str) -> list[str]:
    result: list[str] = []
    for value in actions.get(field, []):
        if field == "ancient_choice" and not value.get("was_chosen"):
            continue
        title = value.get("title", {})
        key = title.get("key") or value.get("TextKey")
        if isinstance(key, str):
            result.append(_compact_id(key.removesuffix(".title")))
    return result


def _action_label(floor: dict[str, Any]) -> str:
    actions = floor["actions"]
    bits: list[str] = []
    relics = floor["relics"]
    shop = floor["shop"]
    if shop is not None:
        purchases: list[str] = []
        for field, label in (
            ("card_ids_bought", "cards"),
            ("relic_ids_bought", "relics"),
            ("potion_ids_bought", "potions"),
        ):
            if shop[field]:
                purchases.append(
                    label + "=" + ",".join(map(_compact_id, shop[field]))
                )
        bits.append(
            f"shop spent={shop['gold_spent']}"
            + (" buy " + " ".join(purchases) if purchases else "")
        )
        if shop["card_ids_removed"]:
            bits.append(
                "remove=" + ",".join(map(_compact_id, shop["card_ids_removed"]))
            )
        if shop["potion_ids_discarded"]:
            bits.append(
                "discard="
                + ",".join(map(_compact_id, shop["potion_ids_discarded"]))
            )
    elif relics["gained"]:
        bits.append("relic +" + ",+".join(map(_compact_id, relics["gained"])))
    if relics["removed"]:
        bits.append("relic -" + ",-".join(map(_compact_id, relics["removed"])))
    for field, prefix in (("ancient_choice", "ancient"), ("event_choices", "event")):
        values = _chosen_keys(actions, field)
        if values:
            bits.append(f"{prefix}=" + ",".join(values))
    if actions.get("rest_site_choices"):
        bits.append("rest=" + ",".join(actions["rest_site_choices"]))
    for field, prefix in (
        ("cards_gained", "card +"),
        ("cards_removed", "card -"),
        ("upgraded_cards", "upgrade "),
        ("downgraded_cards", "downgrade "),
        ("potion_used", "use "),
        ("potion_discarded", "discard "),
    ):
        if shop is not None and field in {
            "cards_gained", "cards_removed", "potion_discarded"
        }:
            continue
        values = actions.get(field, [])
        ids = [value.get("id") if isinstance(value, dict) else value for value in values]
        if ids:
            bits.append(prefix + ",".join(map(_compact_id, ids)))
    choices = actions.get("card_choices", [])
    if choices and shop is None:
        picked = [
            _compact_id(choice.get("card", {}).get("id"))
            for choice in choices
            if choice.get("was_picked")
        ]
        bits.append(f"cards={len(choices)}" + (" pick=" + ",".join(picked) if picked else " skip"))
    return "; ".join(bits) or "-"


def render_text(reconstruction: dict[str, Any], show_relics: bool = False) -> str:
    run = reconstruction["run"]
    player = reconstruction["player"]
    outcome = "win" if run["win"] else "abandoned" if run["was_abandoned"] else "loss"
    lines = [
        f"Run {run['start_time']}  build={run['build_id']}  A{run['ascension']}  "
        f"character={_compact_id(player['character'])}  outcome={outcome}  "
        f"floors={len(reconstruction['floors'])}",
        f"source={reconstruction['source']['path']}  sha256={reconstruction['source']['sha256']}",
        "",
        "floor  act.node  point       rooms                              hp       gold  actions",
    ]
    prior_act = None
    for floor in reconstruction["floors"]:
        if floor["act"] != prior_act:
            lines.append(f"-- Act {floor['act']}: {floor['act_id']} --")
            prior_act = floor["act"]
        state = floor["player_state"]
        hp = (
            f"{state.get('current_hp', '?')}/{state.get('max_hp', '?')}"
            if "current_hp" in state or "max_hp" in state
            else "-"
        )
        rooms = " + ".join(_room_label(room) for room in floor["rooms"]) or "-"
        lines.append(
            f"{floor['floor']:>5}  {floor['act']}.{floor['act_floor']:02d}      "
            f"{str(floor['map_point_type']):<11} {rooms:<34} "
            f"{hp:<8} {str(state.get('current_gold', '-')):>5}  {_action_label(floor)}"
        )
        if show_relics:
            held = ", ".join(map(_compact_id, floor["relics"]["held_after"])) or "-"
            lines.append(f"       relics after: {held}")
    for warning in reconstruction["warnings"]:
        lines.append(f"warning: {warning}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_file", help="completed .run JSON file, or - for stdin")
    parser.add_argument(
        "--format", choices=("text", "json", "jsonl"), default="text",
        help="output format (default: text)",
    )
    parser.add_argument("--player", type=int, help="player id for a multiplayer run")
    parser.add_argument(
        "--show-relics", action="store_true",
        help="show held relics after every floor in text output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run, source, content_sha256 = _load_run(args.run_file)
        result = reconstruct_run(run, source, content_sha256, args.player)
    except (OSError, RunFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif args.format == "jsonl":
        common = {
            "format_version": result["format_version"],
            "source": result["source"],
            "run": result["run"],
            "player": {
                key: result["player"][key] for key in ("id", "character")
            },
        }
        for floor in result["floors"]:
            print(json.dumps({**common, **floor}, sort_keys=True))
    else:
        sys.stdout.write(render_text(result, args.show_relics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
