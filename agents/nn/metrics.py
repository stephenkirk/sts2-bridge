"""Episode aggregation shared by training and paired evaluation. See contracts.md."""


def summary(results, fights=False):
    """Aggregate episode metrics overall, by room type, and by character.

    Character comes from episode metadata, independently of the fight display name. `fights=True` adds per-fight metrics.
    """
    def of(rs):
        lengths = sorted(r["turn"] for r in rs)
        return dict(win=mean(r["won"] for r in rs), kept=mean(r["kept"] for r in rs), reward=mean(r["reward"] for r in rs),
                    boss=mean(r["boss"] for r in rs), turn=mean(r["turn"] for r in rs), n=len(rs),
                    timeouts=sum(r.get("timeout", False) for r in rs),
                    hp_change=mean(r.get("hp_change", 0) for r in rs),
                    hp_change_wins=mean(r.get("hp_change", 0) for r in rs if r["won"]),
                    hp_lost_wins=mean(r.get("hp_lost", 0) for r in rs if r["won"]),
                    turn_p95=lengths[max(0, (95 * len(lengths) + 99) // 100 - 1)] if lengths else None,
                    turn_max=max(lengths) if lengths else None)

    def by(key):
        groups = {}
        for r in results:
            groups.setdefault(key(r), []).append(r)
        return {k: of(groups[k]) for k in sorted(groups)}
    out = dict(of(results), by_kind=by(lambda r: r["kind"]), by_char=by(lambda r: r["character"]))
    if all("encounter_pool" in r for r in results):
        out["by_pool"] = by(lambda r: r["encounter_pool"])
        out["by_floor_band"] = by(lambda r: r["floor_band"])
    if fights:
        out["by_fight"] = {k: dict(win=v["win"], kept=v["kept"], hp_change=v["hp_change"], timeouts=v["timeouts"],
                                  turn_p95=v["turn_p95"]) for k, v in by(lambda r: r["fight"]).items()}
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def acceptance_summary(results):
    """Counts by outcome and length, including episodes excluded by the lag filter."""
    counts = {}
    for accepted, res in results:
        outcome_name = "timeout" if res.get("timeout") else "win" if res["won"] else "loss"
        length = "1-5" if res["turn"] <= 5 else "6-10" if res["turn"] <= 10 else "11-20" if res["turn"] <= 20 else "21+"
        cell = counts.setdefault(f"{outcome_name}/{length}", dict(accepted=0, dropped=0))
        cell["accepted" if accepted else "dropped"] += 1
    return counts
