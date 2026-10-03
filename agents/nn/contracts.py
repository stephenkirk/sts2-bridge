"""Identity and the sole legacy evaluation-log adapter. See contracts.md."""


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
