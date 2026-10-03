# sts2-bridge

A Python interface to Slay the Spire 2, built for my own game-agent experiments.

The bridge runs the game's own `sts2.dll` headless against stubbed Godot. A .NET worker handles
game execution; Python receives observations and available actions, supplies a choice, and advances
to the next decision. You can control individual combats or a whole run, from map choices to fights
and rewards.

The repository includes a [greedy combat baseline](examples/basic_policy.py), an experimental
[PPO combat agent](agents/nn/README.md) with training and evaluation tools and a live dashboard,
and a [whole-run agent](agents/run/play_run.py) that combines combat search with preferences for
choices outside combat. You can use these or bring your own policy.

Ways in:

- **`.mcr`**: the game's combat recording. Replay it, or load it and step from the first turn.
- **`.run`**: the game's run history. Build a combat spec from any fight in it, or read it floor by floor
  with `python3 -m sts2bridge.chronology x.run` (`--format json|jsonl`; pure Python, no setup).
- **New run**: start a whole run under Python control.

Parity is pinned on one recorded Insatiable boss fight: replay and stepping both match all 49 of
the game's state checksums. Other encounters have smoke coverage. The [test notes](docs/combat-parity.md#checking-against-the-game)
and [neural design notes](docs/nn-design.md) describe what's tested and what's experimental.

Spirebird `.spgn` tapes are read-only here. `sts2bridge.spgn` excerpts a tape's last combat to JSON, and the
tests use that excerpt as a second witness: Spirebird's 49 checksums must equal the game's. A tape can't be
loaded, replayed, or handed to a policy.

```python
from examples.basic_policy import policy
from sts2bridge import CombatWorker
from sts2bridge.fixtures import MCR

with CombatWorker() as w:
    s = w.load(MCR)
    while s["boundary"] != "terminal":
        s = w.step(policy(s))
```

The [example policy](examples/basic_policy.py) scores damage, needed block, and early powers.
It's a small baseline to replace with your own. After setup, run it from the repository root:

```sh
python3 -m examples.basic_policy
```

On an M-series Mac: boot ≈ 0.5 s, combat start ≈ 8 ms, step ≈ 5 ms (21 ms for an end turn).

## Setup

You'll need an installed copy of Slay the Spire 2 (the fixtures were recorded on v0.111.0), a .NET 9 runtime
plus any SDK ≥ 9, and Python ≥ 3.11. The first build also needs network access for NuGet.

```sh
./setup.sh                                        # game DLLs into lib/, IL-patch sts2.dll, build the drivers
python3 -m unittest discover -s tests -t . -v     # parity suites; they skip without setup.sh
```

If the game isn't in a default Steam location, pass its directory to `./setup.sh /path/to/game`.
The script patches a local copy of the DLL; it only reads from the install. The tests also check that
nothing in the game's save directory changes.

Optional extras: `.[nn]` (torch, numpy) for `agents/nn`, `.[spgn]` (cbor2) for reading raw tapes.
The committed excerpt needs neither.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[nn]'
```

## Layout

```text
setup.sh             game DLLs → lib/ (gitignored), IL patch, build
dotnet/
  GodotStubs/        the engine stand-in sts2.dll links against          (vendored from sts2-cli, MIT)
  Patcher/           the two IL patches to sts2.dll                      (vendored from sts2-cli, MIT)
  Substrate/         HeadlessInit (vendored boot) + the parity fixes and headless guards
  CombatWorker/      the JSON-lines worker: load / start / step, start_run / run_step
  ReplayCheck/       replays an .mcr and diffs every checksum
  StubAudit/         which Godot members sts2.dll needs that the stubs lack; rerun per game version
sts2bridge/          Python client (worker.py), pinned fixtures, .run chronology, .spgn excerpter
tests/               the contract: replay 49/49, step 49/49, hand-played win, specs, runs
fixtures/            the Insatiable fight: .mcr, .run, hand-win, .spgn excerpt
agents/nn/           PPO on starter, Insatiable, or reconstructed library fights
agents/run/          whole-run player: search in combat, priors outside it
docs/                how parity was established, what the policy is given, findings
```

For combat specs, whole runs, and debugging, see [Running fights from Python](docs/combat-parity.md).
For the inputs available to an agent, see [what a combat policy sees](docs/what-we-give-it.md).

## License

[MIT](LICENSE).

## Thanks

Thanks to [Mega Crit](https://www.megacrit.com/) for the game,
[Hao Wu's sts2-cli](https://github.com/wuhao21/sts2-cli) for the headless foundation,
[jorbs' Spirebird](https://spirebird.com/) for the recorder and community stats,
and [ptrlrd's Spire Codex](https://github.com/ptrlrd/spire-codex) for the game data.
