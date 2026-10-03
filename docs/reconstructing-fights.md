# Reconstructing a fight from a real run

Tape reconstruction notes from 2026-09-29, game v0.111.0. Fidelity depends on the source.

| Source | What you get |
|---|---|
| Hand-written spec | A representative fight: any deck, HP and seed you choose (`worker.catalog()` has starting HP). Not one you played. |
| `.run` + `spec_from_run` | Your deck, relics and potions at the **end** of the run, and the HP you entered the last room with. Relic props (Fishing Rod's `CombatsSeen`, …) are end-of-run values too. |
| `.run` + library reconstruction | Inferred decks from dated mutations and relic ownership before Act 1 fights. Entry HP/gold come from the prior node; relic props reset and the potion belt is empty. See the [library curriculum](../agents/nn/README.md#library-curriculum). |
| `.run` + Spirebird tape | **The fight you played.** Same monsters, same draws, same state at every checkpoint. |
| `.mcr` | Exact and loadable, but the game only keeps the latest one. |

## Rebuilding a fight from a tape

A tape records a full-state text dump at every checkpoint. The dump at the first "After player
turn start" gives HP, max HP, gold, the whole deck, relics with their props at that moment, and
the RNG counters. The spec is then:

- `seed`: the run's seed. At floor 2 the opening shuffle matches from the seed alone.
- `player`: HP, max HP, gold, and the relics, with props taken from the dump rather than the `.run`.
- `run.map_point_history`: the `.run`'s history entries before the fight. An encounter seeds its
  monsters with run seed + `TotalFloor` + encounter ID, and `TotalFloor` is the length of that
  history. Without it, Slimes spawn different slimes with different HP.
- Leave the RNG counters alone. The tape's `combat_start` counters come after the opening
  shuffle, so overlaying them shuffles twice. At floor 2 the fresh counters already match.

## Replaying your inputs

A tape's play records name the **combat card index** and the target's **combat id**. The
observation's `combat_card` and `enemies[].combat_id` fields map straight onto them. A choice
record's payload isn't decoded here. The pick shows up as the card that leaves the hand between
the dumps on either side of it.

For five floor-2 fights (one per character), every checkpoint's state matches the tape: 167 of
167. Three dump lines are excluded because combat never reads them:

- `Last executed action ID`: the real run had already used IDs before the fight.
- `Reward IDs` and the relic grab bag: they differ when a Neow relic came out of the pool.

Because of these lines, the hashes match only where no action has run yet.

## Open

- Later-floor tape reconstruction was not tested in this work. RNG counters (`niche`,
  `monster_ai`, …) require pre-shuffle values.
- The fixture seed `7TA07BQT5BSJ` gives the same weak-pool monsters on every floor. Five
  other seeds don't. The cause is unresolved.
