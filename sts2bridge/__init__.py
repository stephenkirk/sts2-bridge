"""Slay the Spire 2, headless, at game parity: step combats and whole runs from Python."""

from .worker import GAME_DRIVEN, CombatWorker, WorkerError, card_name, recorded_action, spec_from_run

__all__ = [
    "GAME_DRIVEN", "CombatWorker", "RunFormatError", "WorkerError",
    "card_name", "reconstruct_run", "recorded_action", "spec_from_run",
]


def __getattr__(name):
    # Lazy, so `python3 -m sts2bridge.chronology` doesn't find itself already imported by the package.
    if name in ("RunFormatError", "reconstruct_run"):
        from . import chronology
        return getattr(chronology, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
