# sts2-bridge

Run Slay the Spire 2 headless and step through combats and whole runs from Python.

The game's own `sts2.dll` runs against stubbed Godot, with Python supplying an action at each decision point.
The rules stay; the clicking goes. This started as plumbing for my own game-agent experiments.

Ways in:

- **`.mcr`**: the game's combat recording. Replay it, or load it and step from the first turn.
- **`.run`**: the game's run history. Build a combat spec from any fight in it.
- **New run**: start a whole run under Python control.

Parity is pinned on one recorded Insatiable boss fight. Replay and stepping both match all 49 of the game's
state checksums, and the tests keep checking.

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

Youl'll need Python ≥ 3.11 to run the code in this project. Optional extras: `.[nn]` (torch, numpy) for `agents/nn`,
`.[spgn]` (cbor2) for reading raw tapes. The committed excerpt needs neither.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[nn]'
```
### Local 

If you have an installed copy of Slay the Spire 2 (the fixtures were recorded on v0.111.0) available on the
same system, you can set up a .NET 9 runtime plus any SDK ≥ 9 and network access for NuGet for the first build
for a straight-forward local setup:

```sh
./setup.sh                                        # game DLLs into lib/, IL-patch sts2.dll, build the drivers
python3 -m unittest discover -s tests -t . -v     # parity suites; they skip without setup.sh
```

If the game isn't in a default Steam location, pass its directory to `./setup.sh /path/to/game`.
The script patches a local copy of the DLL; it only reads from the install. The tests also check that
nothing in the game's save directory changes.

### Virtualized

If you do not have Slay the Spire 2 available, you can also run this code against a docker container. The
`CombatWorkerContainer` instance you get can, for the most part, be used interchangeably with the `CombatWorker`
the documentation uses.

Start the worker container with its input files mounted, then connect to it
from Python. The path passed to the worker is a path inside the container:

```sh
docker run --rm -p 18888:18888 \
  --mount type=bind,source=/path/to/fixtures,target=/fixtures,readonly \
  headless-sts2:0.111.0
```

```python
from sts2bridge.fixtures import MCR
from sts2bridge import CombatWorkerContainer

with CombatWorkerContainer(address=("127.0.0.1", 18888)) as worker:
    tape = worker.tape(f"/fixtures/{MCR.name}")
    print(f"{tape['version']}: {len(tape['checkpoints'])} recorded checksums")
```
---

For a self-contained setup, a Python wrapper for docker such as `testcontainers` can be used:

```python
from sts2bridge.fixtures import MCR, FIXTURES
from sts2bridge.worker import CombatWorkerContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

with DockerContainer("headless-sts2:0.111.0", ports=[18888]).with_volume_mapping(
    str(FIXTURES), "/fixtures", mode="ro"
) as container:
    container.waiting_for(
        LogMessageWaitStrategy("STS2 bridge worker listening on").with_startup_timeout(180)
    )
    address = (
        container.get_container_host_ip(),
        int(container.get_exposed_port(18888)),
    )
    with CombatWorkerContainer(address=address) as worker:
        tape = worker.tape(f"/fixtures/{MCR.name}")
        print(f"{tape['version']}: {len(tape['checkpoints'])} recorded checksums")
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
sts2bridge/          Python client (worker.py), pinned fixtures, .spgn excerpter
tests/               the contract: replay 49/49, step 49/49, hand-played win, specs, runs
fixtures/            the Insatiable fight: .mcr, .run, hand-win, .spgn excerpt
agents/nn/           PPO on the Insatiable fight
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
