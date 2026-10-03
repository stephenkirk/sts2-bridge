# Running fights from Python

The bridge keeps the game running in a .NET worker. Python sends an action, the game resolves it,
and you get control back when there's another decision to make.

## Step a combat

After [setup](../README.md#setup), run this from the repository root:

```python
from examples.basic_policy import policy
from sts2bridge import CombatWorker
from sts2bridge.fixtures import MCR

with CombatWorker() as worker:
    state = worker.load(MCR)
    while state["boundary"] != "terminal":
        state = worker.step(policy(state))
    print(state["obs"]["player"]["hp"])
```

`load` starts at the recording's opening position. You're free to play a different line.
Load it again to reset the fight; keep the worker open to avoid paying for another boot.

A step returns at `awaiting_input`, `awaiting_choice`, or `terminal`. Ending your turn runs
through the enemy turn and into your next decision, including any card choice along the way.

| Action | Shape |
|---|---|
| Play a card | `{"type": "play", "hand": i, "target": slot}` |
| End turn | `{"type": "end_turn"}` |
| Use a potion | `{"type": "potion", "slot": i, "target": slot}` |
| Choose cards | `{"type": "choose", "picks": [i, ...]}` |

Pick actions from `state["legal"]`; targets are enemy slots, or `None` when no target is needed.
For multi-card choices, `legal` is empty: choose distinct indices from `choice["options"]`,
between `choice["min"]` and `choice["max"]`. The example policy handles both cases.

Most of what you need is in three reply fields:

| Field | What's there |
|---|---|
| `obs` | Player, cards, piles, orbs, relics, enemies, and their intents |
| `legal` | Available actions, including single-card choices |
| `choice` | Options and pick counts when a card choice is pending, and what it is for: `screen` (the game's selection method), `prompt` (its prompt key, such as `TO_DISCARD`), and `source` (the card, relic or power that asked) |

The [observation notes](what-we-give-it.md) describe the fields in more detail.
For debugging, replies also include `state_hash`, `checkpoints`, `enqueued_by_game`, and
`game_errors`. The Python client raises `WorkerError` when a request fails;
`worker.log_path` points to the worker's stderr.

## Start any combat

You can take the deck from a run history and put it against another encounter:

```python
import json
from pathlib import Path
from examples.basic_policy import policy
from sts2bridge import CombatWorker, spec_from_run

run = json.loads(Path("fixtures/7TA07BQT5BSJ.run").read_text())
spec = spec_from_run(run, "ENCOUNTER.CEREMONIAL_BEAST_BOSS", seed="ANY", current_hp=77)

with CombatWorker() as worker:
    state = worker.start(spec)
    while state["boundary"] != "terminal":
        state = worker.step(policy(state))
```

`spec_from_run` takes the final deck, relics, and potions, with HP from entry to the last room.
Keyword arguments override those values. Earlier decks require forward reconstruction of
recorded mutations; see the [library curriculum](../agents/nn/README.md#library-curriculum)
for its assumptions and rejected cases.

You can also build a spec yourself: `character`, `ascension`, `seed`, `encounter`, an optional
`act` (zero-based, inferred from the encounter when omitted), and `player` fields in the
game's save format. Unspecified player fields keep their fresh-run defaults. Inventory is
loaded directly, without repeating relic pickup effects. `worker.catalog()` lists starting HP
and deck by character, the ascension that adds Ascender's Bane, and each encounter's room
type, acts, and weak-pool membership. Unknown IDs are rejected.
The same spec and seed reproduce the same fight. An optional `run` overlays run-level save fields
the same way, such as `map_point_history`, which sets the floor an encounter seeds its monsters
with; [Reconstructing a fight](reconstructing-fights.md) uses it to re-enter fights from real runs.
With `STS2_DUMP` set in the worker's environment, each checkpoint also carries the game's text
dump of the state it hashed.

For training, `worker.start(spec, hashes=False, reuse_map=True)` skips hashing and caches the act map.
`hashes=False` (also on `load`) leaves `state_hash` null and `checkpoints` empty, skipping a full-state
serialisation per reply and per action; `reuse_map=True` generates the act's map once per spec rather
than once per seed. To keep several workers busy from one thread, `send` to each, then `receive` from each
in the same order.

For buffs or other experimental changes, see [sandbox changes](sandbox-manipulations.md).

## Whole runs

`worker.start_run("CHARACTER.IRONCLAD", "MYSEED", ascension=10)` starts a run;
`worker.run_step(action)` advances it through maps, events, fights, rewards, shops, and rest sites.
Ascension defaults to 0. `unlocks="all"` is the default; use `"none"` for a fresh profile.

Check `state["boundary"]` for the current decision and `state["legal"]` for available actions.
During combat, the combat reply is nested under `state["combat"]`. A run policy needs to handle
room decisions as well as fights; the basic example only handles combat.

The existing [run agent](../agents/run/play_run.py) does both:

```sh
python3 -m agents.run.play_run --seed MYSEED --sims 0
```

With `--sims 0` it uses a rollout policy. Search mode uses snapshot copies of the fight, which
reveal future draws and rolls. Outside combat it follows an optional preferences file passed with
`--priors`, rating cards, relics, and event options; the format is in `agents/run/priors.py`.
Without one, it skips card rewards and buys no cards.

`worker.run_trace_path` holds a JSONL trace of decisions and replies.
`worker.run_combat_snapshot(path)` saves an `.mcr` that another worker can load from the fight's
opening position. It isn't a mid-fight save.

## Checking against the game

The parity reference is the recorded Insatiable fight: Defect A10, seed `7TA07BQT5BSJ`, game
v0.111.0. Both replaying the tape and sending its actions through `step` match all 49 of the
game's checksums and the final state. Dropping one action makes the test diverge.

```sh
python3 -m unittest discover -s tests -t . -v
```

The suite also replays a hand-played win and tries a random policy across all 85 encounters in
that build. Checksum parity is tested on the one recorded fight; the encounter sweep checks
that fights can run to completion. Whole-run victory is still untested, and the act-transition
test is skipped until there's a suitable seed or policy.

For a mismatch, `ReplayCheck` writes a report and both states at the first divergence.
Run it in a scratch directory, since the Godot stubs resolve `user://` against the working directory:

```sh
bridge_root="$PWD"  # run from the repository root
replay_dir="$(mktemp -d)"
(cd "$replay_dir" && dotnet "$bridge_root/dotnet/ReplayCheck/bin/Debug/net9.0/ReplayCheck.dll" \
  "$bridge_root/fixtures/7TA07BQT5BSJ-f33-the-insatiable.mcr" out --strict-steps)
```

Look in `$replay_dir/out/` for `replay_report.json` and `diverge_<id>_{local,replay}.txt`.
The [initial findings](findings-2026-09-26.md) cover checkpoint contents and the adapter work.

## Vendored from sts2-cli

The Godot stubs, DLL patcher, setup's DLL-copy routine, and headless bootstrap come from
[sts2-cli](https://github.com/wuhao21/sts2-cli) at `084d1aa`, under
[Hao Wu's MIT license](../dotnet/GodotStubs/LICENSE-sts2-cli).
The bridge adds missing stubs, deferred-call handling, and guards around scene and audio calls.
It uses empty localization tables and omits the upstream Task.Yield and bundle-screen patches.

One inherited patch still needs attention: `Neutralize.OnPlay` replaces the card's body with
damage and Weak. The recorded fixture doesn't exercise it.

After a game update, rerun `./setup.sh` and the tests. If new Godot calls fail, use
[`StubAudit`](../dotnet/StubAudit/Program.cs) to find missing members and their callers.
