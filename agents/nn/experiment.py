"""Resolve small experiment definitions through the training CLI's existing parser."""

import argparse
import hashlib
import json
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent / "experiments"


def resolve(parser, argv):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--experiment")
    selected, _ = selector.parse_known_args(argv)
    metadata = None
    flags = []
    if selected.experiment:
        path = Path(selected.experiment)
        if not path.is_file():
            path = EXPERIMENTS / (selected.experiment + ".json")
        try:
            raw = path.read_bytes()
            definition = json.loads(raw)
            if definition["format_version"] != 1:
                raise ValueError("unsupported experiment format")
            if not isinstance(definition["id"], str) or not isinstance(definition["hypothesis"], str):
                raise ValueError("id and hypothesis must be strings")
            training = definition["training"]
            inputs = definition.get("required_inputs", [])
            if not isinstance(training, dict) or not isinstance(inputs, list):
                raise ValueError("training must be an object and required_inputs a list")
            actions = {a.dest: a for a in parser._actions}
            forbidden = {"name", "resume", "experiment", "dry_run", "help", "init_from", "pool_file"}
            for key, value in training.items():
                if key not in actions or key in forbidden:
                    raise ValueError(f"unsupported training setting: {key}")
                action = actions[key]
                flag = action.option_strings[0]
                if isinstance(action, argparse.BooleanOptionalAction):
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} must be boolean")
                    flags.append(flag if value else action.option_strings[1])
                elif isinstance(action, argparse._StoreTrueAction):
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} must be boolean")
                    if value:
                        flags.append(flag)
                else:
                    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                        raise ValueError(f"{key} must be a scalar")
                    flags.extend([flag, str(value)])
            if any(key not in {"init_from", "pool_file"} for key in inputs):
                raise ValueError("required_inputs supports only init_from and pool_file")
            metadata = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(), definition=definition)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"invalid experiment {selected.experiment}: {exc}")
    args = parser.parse_args(flags + list(argv))
    if metadata:
        if args.resume:
            if "--extend-vocab" not in argv:
                args.extend_vocab = False
        for key in inputs:
            if key == "init_from" and args.resume:
                continue
            if getattr(args, key) is None:
                parser.error(f"experiment {definition['id']} requires --{key.replace('_', '-')}")
        metadata["resolved"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        metadata["selection_metric"] = "win" if args.pool == "insatiable" else "reward"
    return args, metadata


def record(out, metadata, resume):
    """Keep the original conditions; reject attaching a different definition on resume."""
    path = out / "experiment.json"
    if resume and path.exists():
        original = json.loads(path.read_text())
        if metadata and original["sha256"] != metadata["sha256"]:
            raise ValueError("resume needs the original experiment definition; branch with --init-from")
        return metadata or original
    if metadata:
        if resume:
            raise ValueError("cannot attach an experiment to an ad hoc run on resume; branch with --init-from")
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
