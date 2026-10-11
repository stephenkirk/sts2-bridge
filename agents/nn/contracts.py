"""Identity and the sole legacy evaluation-log adapter. See contracts.md."""

import hashlib
import json
from pathlib import Path


def digest(value):
    """Hash JSON content independently of object key order and whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def runtime_identity(game_lib):
    """Fingerprint the game, worker, stubs, and Python combat runner actually used."""
    from sts2bridge.worker import BINARY

    files = {}
    if game_lib:
        library = Path(game_lib)
        files = {"game/" + p.name: p for p in set(library.glob("*.dll")) | {library / "sts2.dll"}}
    worker_files = set(BINARY.parent.glob("*.dll")) | {
        BINARY, BINARY.parent / "GodotSharp.dll", BINARY.with_suffix(".deps.json"),
        BINARY.with_suffix(".runtimeconfig.json")}
    files.update({"worker/" + p.name: p for p in worker_files})
    files.update({"python/episode.py": Path(__file__).with_name("episode.py"),
                  "python/worker.py": Path(__file__).resolve().parents[2] / "sts2bridge" / "worker.py"})
    identity = {name: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                for name, path in sorted(files.items())}
    if not game_lib:
        identity["game/sts2.dll"] = None
    return identity


def evaluation_conditions(fights, seeds, *, max_turns, build):
    """Identity of the held-out exam; training reward and model settings may differ.

    Fight order is retained because maps are cached in evaluation order. Report names stay
    in the identity so consumers can safely pair rows by fight name and full seed.
    """
    conditions = dict(format_version=1,
                      fights=[dict(name=f["name"], spec_sha256=digest({k: v for k, v in f["spec"].items() if k != "seed"}))
                              for f in fights], seeds=list(seeds),
                      execution=dict(policy="greedy", max_turns=max_turns, start="spec", reuse_map=True,
                                     multi_pick="first_min"),
                      build=build)
    return dict(conditions, sha256=digest(conditions))


def condition_id(conditions):
    """Unrecorded, incomplete, or unsupported conditions never certify a match."""
    if not isinstance(conditions, dict) or conditions.get("format_version") != 1:
        return "unknown"
    fights, seeds, execution, build = (conditions.get(k) for k in ("fights", "seeds", "execution", "build"))
    if not isinstance(fights, list) or not fights or not all(
        isinstance(f, dict) and isinstance(f.get("name"), str) and isinstance(f.get("spec_sha256"), str) for f in fights
    ):
        return "unknown"
    if len({f["name"] for f in fights}) != len(fights):
        return "unknown"
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, str) for s in seeds) or len(set(seeds)) != len(seeds):
        return "unknown"
    if not isinstance(execution, dict) or not all(k in execution for k in (
        "policy", "max_turns", "start", "reuse_map", "multi_pick"
    )):
        return "unknown"
    if not isinstance(execution["max_turns"], int) or execution["max_turns"] < 1 or not isinstance(execution["reuse_map"], bool):
        return "unknown"
    if not all(isinstance(execution[k], str) and execution[k] for k in ("policy", "start", "multi_pick")):
        return "unknown"
    required_build = {"game/sts2.dll", "worker/CombatWorker.dll", "worker/GodotSharp.dll", "python/episode.py", "python/worker.py"}
    if not isinstance(build, dict) or not required_build.issubset(build) or not all(isinstance(v, str) and v for v in build.values()):
        return "unknown"
    try:
        actual = digest({k: v for k, v in conditions.items() if k != "sha256"})
    except (TypeError, ValueError):
        return "unknown"
    return actual if conditions.get("sha256") == actual else "unknown"


def compare_conditions(a, b):
    if "unknown" in (condition_id(a), condition_id(b)):
        return dict(status="unknown", reason="Evaluation conditions are missing or incomplete; paired results are unavailable.")
    different = [label for key, label in (("fights", "fights"), ("seeds", "seeds"),
                                         ("execution", "evaluation method"), ("build", "game or bridge build"))
                 if a[key] != b[key]]
    if different:
        return dict(status="different", reason="Different " + ", ".join(different) + "; paired results are unavailable.")
    return dict(status="matching", reason="Same fights, seeds, game/bridge build, and evaluation method.")


def character(fight):
    """Unprefixed character key for reports, sourced from the canonical spec ID."""
    return fight["spec"]["character"].removeprefix("CHARACTER.")


def heldout(evaluation):
    """Read canonical summaries, or adapt old logs with flattened heldout_* fields.

    Nested values win when both are present. New writers only emit nested summaries.
    """
    if "heldout" in evaluation:
        return evaluation["heldout"]
    return {key.removeprefix("heldout_"): value for key, value in evaluation.items()
            if key.startswith("heldout_") and key not in {"heldout_cards", "heldout_vocab"}}
