# AGENTS.md

Headless Slay the Spire 2: the game's own `sts2.dll` runs against stubbed Godot, and Python steps combats and runs.

- New to the game? Read [docs/game-rules.md](docs/game-rules.md) before touching anything game-facing.
- How to drive it: [docs/combat-parity.md](docs/combat-parity.md). What a policy sees: [docs/what-we-give-it.md](docs/what-we-give-it.md).
- Setup: `./setup.sh`. Tests: `python3 -m unittest discover -s tests -t . -v`. They skip without setup, so a skip is not a pass.
- Parity is the contract. Replay and stepping must match all 49 checksums of the pinned Insatiable fight. Don't loosen a test to get green.
- `lib/` is generated and gitignored. Never write to the game install or its save directory.
