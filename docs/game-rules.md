# The game, for people who haven't played it

Slay the Spire 2's rules and vocabulary, plus how a completed `.run` save records a run.
These notes come from observed saves and play, not an official spec. Details that change between builds
(names, numbers, card text) belong in the catalog linked at the bottom.

## The one-minute model

Slay the Spire 2 is a run-based deck-building game. A player chooses a
**character**, starts with that character's small deck and starting relic, and
tries to survive a route through three **acts**. Each position on the route is
a selectable **node** on a connected map; the save format calls that position
a **map point**. It may contain combat, an event, a shop, a rest site, treasure,
or an Ancient choice. Resolving it advances the run by one **floor** and can
change health, gold, cards, relics, and potions.

**Combat** is played by drawing and playing cards from the current deck. Surviving
combat commonly produces rewards, including an offer of cards from which the
player may take one or skip. Elites are harder combats with relic rewards;
bosses end acts. Deck-building happens across the entire route through card
rewards, shops, events, Ancients, and card mutations such as upgrades,
removals, transforms, and enchantments.

A **run** is the whole attempt, not one combat. It ends in a win, death, or
abandonment. A completed `.run` file is a terminal history of that attempt. It
contains a sparse, post-node chronology and final inventory, not a replay of
every combat action or every state that appeared on screen.

In compact form:

```text
character + starting inventory
  -> Act 1 route -> boss
  -> Act 2 route -> boss
  -> Act 3 route -> final boss(es)
  -> win / death / abandonment

each route node
  -> one or more rooms
  -> choices, rewards, and mutations
  -> post-node HP/gold plus sparse inventory actions
```

Everything here comes from single-player, standard-mode Steam runs. It doesn't
cover multiplayer, daily, custom-mode, live-checkpoint, or replay behavior.

## The anatomy of a run

### Character and starting state

The five playable characters represented here are **Ironclad**, **Silent**,
**Defect**, **Regent**, and **Necrobinder**. A character determines the starter
deck, starting relic, base health, and native card pool. Characters can still
acquire cards outside their native pool through game effects.

The logical starting state exists before the first recorded map point. **Neow**
is the Act 1 Ancient and fills the role of the opening Neow choice from Slay the
Spire 1. Acts 2 and 3 begin with a randomly selected eligible **Ancient**, which
replaces the old post-boss relic-choice system. Thus every normal act begins on
an Ancient floor: Neow is fixed for Act 1, while the later Ancient varies.

An Ancient heals some or all missing HP and presents three **Ancient Relics**,
of which the player must take one. The visible three are drawn from a larger
Ancient-specific offering pool. Pools can be divided into slots and eligibility
can depend on current state, so “three random relics” is only the screen shape,
not a sufficient generation algorithm. The chosen relic can immediately alter
cards, relics, HP, maximum HP, gold, potion capacity, or even the map. Floor 1
is therefore post-Neow state, not an untouched copy of the character defaults.

**Ascension** is the run's cumulative difficulty level, tracked per character
in single-player and abbreviated A0 through A10. It changes map generation,
healing and economy, inventory capacity and composition, reward odds, enemy
strength, and at A10 the number of final bosses. Models must retain it. The
current modifier list lives in the versioned
`ascensions.json` in the [Spire Codex](https://github.com/ptrlrd/spire-codex)
catalog because it is build-sensitive.

### Acts, map points, floors, and rooms

An **act** is one major route segment. The observed standard-mode route plans
use one of two Act 1 locations, followed by fixed Act 2 and Act 3 locations:

| Act | Observed model ID | Normal observed length | Ending |
|---|---|---:|---|
| 1 | `ACT.OVERGROWTH` or `ACT.UNDERDOCKS` | 17 map points | boss |
| 2 | `ACT.HIVE` | 16 map points | boss |
| 3, A0-A9 | `ACT.GLORY` | 15 map points | boss |
| 3, A10 | `ACT.GLORY` | 16 map points | two consecutive bosses |

Rare and legacy route exceptions exist. The table is grounding, not a parser
constraint. In particular, observed alternate Act 2 routes can be longer.

The player navigates a **map** of connected **nodes** and thinks of progress in
**floors**. “Map point” is a useful name for a selected position and is also the
save-format term. Here `floor` means for the 1-based global ordinal of
a reached map point across all acts. It also retains `act_floor`, the 1-based
ordinal within an act. Thus the normal first point of Act 2 is global floor 18
and act floor 1. An ordinary observed win ends on global floor 48; an A10 win
normally ends on 49 because of the extra Act 3 boss.

### Encounter pools

Each act's hallway fights come from two pools: **weak** encounters (IDs ending
`_WEAK`, `IsWeak` in the model) and **regular** ones. Elites and bosses have
pools of their own. When an act is generated, the game fills a queue of
normal encounters in advance. The first `NumberOfWeakEncounters` slots are
drawn from the weak pool, and the remaining slots up to `BaseNumberOfRooms`
come from the regular pool:

| Act | Weak slots | Queue length |
|---|---:|---:|
| `ACT.OVERGROWTH`, `ACT.UNDERDOCKS` | 3 | 15 |
| `ACT.HIVE` | 2 | 14 |
| `ACT.GLORY` | 2 | 13 |

(Multiplayer queues are one slot shorter. If the queue runs out, it wraps
around to the start.) Each pool is a grab bag: an encounter isn't redrawn
until the bag is empty, and no draw may repeat, or share a tag with, the
encounter queued just before it.

The weak window counts **normal combats, not floors**. The queue position
advances only when a monster room is the first room at a map point, which
means an Enemy node, or an Unknown node that rolls combat. An event that leads
into combat starts its fight as the point's second room, or names its own
encounter. That fight neither draws from the queue nor advances it. So "the
first three fights of Act 1 are weak" holds whether they fall on floors 2–4 or
floors 2, 5, and 9.

On an account's first-ever run, Overgrowth replaces random draws with a fixed
tutorial order. `agents/nn/fights.py`'s `starter` pool is every Act 1 weak
encounter, played from a fresh A10 entry state.

The in-game map legend uses the player-facing labels **Unknown**, **Merchant**,
**Treasure**, **Rest**, **Enemy**, and **Elite**. Save data uses related but not
identical identifiers such as `unknown`, `shop`, `rest_site`, and `monster`.
Do not force the UI words and serialization values into one vocabulary.

A **map point** and a **room** are not interchangeable:

- the map point is the route position and owns the per-player resource totals
  and action arrays;
- a room is an activity resolved at that point; and
- one map point can contain multiple rooms, such as an event followed by
  combat.

The map-point type `unknown` corresponds to the map's question-mark node and is
not a parse failure. It commonly resolves to an event, but it can roll a normal
combat or an event choice can lead into combat. Either route can produce an
event room followed by a monster room at the same point. Relics such as Juzu
Bracelet can change random Unknown-node combat behavior without necessarily
removing combats explicitly caused by event choices. Determine whether a point
actually involved combat from its child room types (`monster`, `elite`, or
`boss`), never from `map_point_type`, damage, or event-option wording.

### Cards, decks, relics, and potions

A **card** has a model identity such as `CARD.BASH`. A **deck** is the player's
current multiset of card copies. Two copies of the same model are distinct in
play, but completed saves do not give them durable instance UUIDs. This makes
some upgrades and removals ambiguous when a deck contains duplicates.

Keep these card concepts separate:

- an **upgrade** changes `current_upgrade_level`;
- an **enchantment** is a separate card mutation with its own ID and amount;
- a boolean "enchanted" flag doesn't name the enchantment: named game
  enchantments are separate `enchantment_id` facts; and
- a card's **native pool** says which character or special pool owns the card,
  not which character was being played when it appeared.

Ordinary post-combat card rewards draw from the played character's native pool.
Specific effects widen or replace that pool: Neow and Orobas have off-character
offerings, Prismatic Gem changes later reward pools, Dingy Rug admits Colorless
cards, shops have Colorless inventory, and some events grant special or
off-character rewards. These are explicit exceptions, not evidence that native
pool and played character are the same dimension.

A **relic** is a persistent run item that can change route, reward, combat, or
economy rules. Relic effects are stateful: reasoning about a floor often
requires knowing which relics were held before that floor, not merely which
relics appear in the terminal inventory.

A **potion** is a slotted consumable. Offered, picked, skipped, bought, used,
and discarded potions are different events. Capacity can depend on ascension,
relics, and other game effects.

### Offers, choices, gains, and mutations

An item being **offered**, **chosen**, and **gained** are related but distinct
facts. For an ordinary combat card reward, the completed save generally
records a flattened `card_choices` sequence plus the selected card in
`cards_gained`. The ordinary group has three options, and the player may choose
at most one or skip it. Relics can enlarge or duplicate reward groups.

The baseline reward shape depends on the combat:

- a survived normal-enemy combat awards gold, a skippable card reward, and can
  offer a potion;
- a survived elite combat also awards a relic, which may be skipped;
- treasure normally offers a relic, which may also be skipped; and
- defeating the Act 1 or Act 2 boss awards gold, a rare-card reward, and can
  offer a potion. The next floor starts the next act with an Ancient instead of
  showing a Slay the Spire 1-style boss-relic screen.

Potion odds are stateful rather than a fixed per-floor schedule. Some relics
also alter these bundles: for example, Neow's Lava Rock makes the Act 1 boss
drop two relics. Treat the list above as a baseline causal shape, not fixed
cardinality.

The save does not preserve clean screen boundaries in every flattened card
sequence. A chosen group is reordered with its pick first, and a skipped group
has no explicit boundary marker. Exact reward grouping can therefore be an
inference even when every raw option is known.

Context changes meaning. At a shop, `card_choices` normally describes recorded
inventory remaining after the shopping session, not an ordinary reward offer
and not necessarily the initial stock. Bought cards instead appear as gains.
Do not feed shop choice arrays into combat reward statistics.

Inventory changes include gain, removal, upgrade, downgrade, transform, and
enchantment. Their floor and surrounding room/action supply causal context.
Avoid generating mutations on arbitrary floors and merely labeling a source
afterward.

### Outcome and terminal state

Use the explicit outcome fields: `win`, `was_abandoned`,
`killed_by_encounter`, and `killed_by_event`. Do not infer death solely from
zero HP. Some terminal or abandoned records force HP to zero without a matching
damage delta.

Likewise, the terminal `players` entry is a final snapshot used for validation;
it is not a substitute for the chronology. Questions such as “which relics
were held entering this elite?” or “when did this card join the deck?” require
ordered map-point actions and mutations.

## Game concept to `.run` shape

This table is for orientation. It isn't a full parser spec.

| Concept | Completed `.run` representation | Important caveat |
|---|---|---|
| Whole attempt | one top-level JSON object | filename/seed alone is not a globally safe identity |
| Planned route | `acts` | lists all three planned acts even after an early loss |
| Reached chronology | `map_point_history[act][point]` | contains only reached acts and points |
| Global floor | derived across nested history | 1-based here; source array indexes are 0-based |
| Activity | `rooms[]` within a map point | a point can contain more than one room |
| Combat | room type `monster`, `elite`, or `boss` | do not infer it from the parent point |
| Post-node resources | `player_stats[]` | totals and deltas belong to the whole point |
| Card reward/shop record | `card_choices` | sparse, flattened, and context-dependent |
| Acquisitions and mutations | sparse action arrays in `player_stats` | absent normally means no recorded action |
| Final deck/relics/potions | top-level `players[]` | terminal state, not historical state |
| Result | explicit top-level outcome/killer fields | HP alone is insufficient |
| Game release | `build_id` | includes a leading `v` in `.run` files |
| Save contract revision | `schema_version` | distinct from the game build |

Model IDs are stable-looking identifiers such as `CHARACTER.DEFECT`,
`CARD.ZAP`, `RELIC.CRACKED_CORE`, or `ACT.HIVE`. Human-facing labels and
localized event-choice text are a separate layer. Do not assume a localized
title is a canonical model ID.

## Things the save doesn't tell you

### Post-node is not moment-by-moment

The completed format records state after a map point and sparse actions at that
point. It does not provide combat card plays, per-floor timestamps, exact
per-item shop prices, every intermediate inventory, or stable IDs for card
copies. Preserve unknowns instead of filling those gaps with convenient values.

### Build version is not schema version

`build_id` identifies a game release such as `v0.108.0`.
`schema_version` identifies the save shape. Some sources, such as Spirebird,
write the build without the leading `v`. Normalize at an explicit boundary;
do not silently conflate the forms.

## Where build-sensitive facts live

Use [Spire Codex](https://github.com/ptrlrd/spire-codex) for card, relic, potion, character, encounter, event,
Ancient-pool, and Ascension data. The fixtures here were recorded on `v0.111.0`. Spire Codex is a community
extraction of the game's C# models and localization. A field there can be missing or parser-derived, and a
`null` Ancient-pool condition doesn't prove the game has no runtime eligibility check. For behavior, the game's own `sts2.dll` in `lib/` has the final word.

Catalog IDs are bare (`BASH`, `CRACKED_CORE`, `HIVE`). Saves and the game use typed IDs (`CARD.BASH`,
`RELIC.CRACKED_CORE`, `ACT.HIVE`). Stripping the prefix is a join normalization, not a rule every ID family
follows. Localized names are never IDs.

Neither the catalog nor the saves fully describe:

- reward generation, including potion-drop odds and when they reset;
- Ancient slot selection, eligibility, weights, and nested follow-up choices;
- route generation and special-map floor counts;
- multiplayer differences; and
- event branches or combat behavior the parser only partly resolved.

A known trap: on `v0.111.0`, Ascension 6 is **Inflation**, not **Gloom**.
