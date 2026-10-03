# Training a combat network

PPO training on starter encounters, the pinned Insatiable fight, and inferred scenarios
from completed runs. Training artifacts stay under ignored `runs/`.

For guarantees, research choices, and why the code is shaped this way, read
[design and evidence boundaries](../../docs/nn-design.md). For fight, episode, log, and
checkpoint fields, read [data contracts](contracts.md). Dated results are in
[September 30](../../docs/nn-experiments-2026-09-30.md) and
[October 1](../../docs/nn-experiments-2026-10-01.md) notes; their local artifacts are not
required to use the native pools.

## Experiments and runs

[`experiments/`](experiments/README.md) holds four readable condition files for the current
research branches. The same trainer runs all of them:

```sh
.venv/bin/python -m agents.nn.train --experiment starter-hp --name hp-trial --dry-run
.venv/bin/python -m agents.nn.launch --experiment starter-hp --name hp-trial
```

`--dry-run` prints resolved conditions without starting workers. Actual runs save
`experiment.json`; CLI overrides appear in session provenance. Turn-cost and library
branches require explicit local checkpoint/manifest paths. Direct CLI training remains
available for ad hoc work. See the [experiment guide](experiments/README.md) for inputs,
resume behavior, and independent comparisons.

## Setup and training

From the repository root, after `./setup.sh`:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e '.[nn]'
.venv/bin/python -m agents.nn.train --name starter --hours 2
.venv/bin/python -m agents.nn.train --name seeds --pool insatiable --hours 2
```

The default device is `mps`; use `--device cpu` or `--device cuda` elsewhere.
Five actors (`--actors`) each operate two workers (`--envs`). Each episode samples a fight
and a fresh `TRAIN-*` seed, unless `--train-on recorded` uses the recording.
Evaluation uses fixed `EVAL-*` seeds: 10 per starter or library
fight, 100 for the Insatiable by default.

A run writes `log.jsonl`, `fights.json`, `ckpt.pt`, `best.pt`, and `snapshots/it*.pt`
under `agents/nn/runs/<name>/`. `best.pt` contains the exact evaluated snapshot with the
highest held-out reward, or win rate for the Insatiable. The initial weights are evaluated
before subsequent snapshots.

`--resume` restores weights, optimizer, vocabulary, and counters for another `--hours`.
It requires the same pool and reward/evaluation settings; legacy HP runs need `--turn-cost-hp 0`.
`--init-from CHECKPOINT` retains weights, architecture, and vocabulary with a fresh optimizer
and counters. Use it when changing scoring. Existing run names require `--resume`.

## Dashboard and playback

Launch training and its dashboard together:

```sh
.venv/bin/python -m agents.nn.launch --name starter --hours 0.333333
```

The printed URL selects the named run. Training flags pass through; `--watch-port` defaults
to 8765. Ctrl-C stops both processes and their workers; the dashboard also stops when training
exits. Run `python -m agents.nn.watch` separately to keep it open afterwards.

The dashboard shows held-out HP change, win rates, per-fight results, and training metrics.
It also shows initialization mode, source checkpoint, optimizer status, and scoring settings.
Its fixed starter baselines come from the [2026-09-29 measurements](../../docs/nn-experiments-2026-09-30.md#starter-baselines--2026-09-29).

`--watch-rematch FILE` replays fixed fights with each run's newest available snapshot.
The file is a JSON list of `{"name", "spec", "you": {"lost", "turns"}}`; see
[reconstructing fights](../../docs/reconstructing-fights.md) for spec provenance.
A standalone watcher accepts the same file through `--rematch`.

Watch a checkpoint play:

```sh
.venv/bin/python -m agents.nn.show agents/nn/runs/seeds/best.pt
.venv/bin/python -m agents.nn.show agents/nn/runs/starter/best.pt --fight IRONCLAD/NIBBITS_WEAK
```

Add `--sample` to sample actions or `--seed MYSEED` to change the shuffle.
`python -m agents.nn.shift CHECKPOINT CHECKPOINT ...` compares greedy trajectories and
scores action probabilities along the pinned hand-played win.

## Fight pools

A fight contains a spec without a seed. [`fights.py`](fights.py) builds native pools from
`worker.catalog()` and reads library pools from a manifest.

- **`starter`** (default): five characters against eight Act 1 weak encounters in v0.111.0,
  for 40 fights. Each starts at A10 with the starter deck and relic, 80% of max HP, and an
  empty potion belt. Weak encounters are drawn for the first normal combats, regardless
  of floor; see [encounter pools](../../docs/game-rules.md#encounter-pools).
- **`insatiable`**: the boss with the terminal inventory and last-room entry HP from run
  `7TA07BQT5BSJ`. The recording is evaluated greedily and sampled. `--train-on recorded`
  trains on its fixed shuffle as a pipeline check; results then measure memorization.
- **`library`**: Act 1 weak, regular, and elite scenarios inferred from completed runs.
  Requires `--pool-file`; construction and assumptions are described below.

Two random baselines are available: `spend` picks among non-end-turn legal inputs when
available; `random` picks among all legal inputs.

```sh
.venv/bin/python -m agents.nn.fights starter
.venv/bin/python -m agents.nn.fights starter --policy random
```

### Library curriculum

`agents.nn.library` builds the pool from your own completed runs: the game's
`saves/history/*.run` files, read with [`sts2bridge.chronology`](../../sts2bridge/chronology.py).
No runs ship with this repo. Starter decks come from the worker's catalog.

```sh
.venv/bin/python -m agents.nn.library --output /tmp/library-act1.json --validate
```

By default it reads `~/Library/Application Support/SlayTheSpire2/steam/*/profile*/saves/history`
(macOS), provided exactly one profile has runs, and selects unmodified solo standard A10
runs from `v0.111.0`. Use `--history PATH` to point elsewhere. The builder only reads the
saves; it writes nothing but the output manifest.

Card mutations are applied forward from the starter deck; runs are rejected if the final
deck does not reconcile. Relic ownership uses the before-node chronology. Entry HP and gold
come from the prior node. Relic props reset to defaults, and potions are omitted because
`encode.py` cannot encode potion actions. Mixed event/combat nodes and bosses are excluded.
Card-copy ambiguity and inferred mutation ordering are recorded in each scenario.
These are training scenarios with fresh RNG, rather than historical replays.

Selection keeps at most two distinct loadout/encounter pairs per source run, floor band,
and encounter pool. Floor bands (`2–5`, `6–10`, `11+`) are sampling bins. Whole runs are split
deterministically by character and outcome, approximately 70/15/15 for train/validation/test;
small strata may have no holdouts. Exact inventory overlaps are removed across splits.
Starter pretraining can still overlap held-out loadouts.

Training balances characters, then targets 20% weak, 50% regular, and 30% elite encounters,
renormalized over available pools. Within each pool it balances floor bands, source runs,
and scenarios. Validation selects checkpoints; test fights are excluded from training and
vocabulary warm-up. `--validate` checks scenario starts and encoding without policy scoring.
The manifest retains reconstruction rejections and split counts.

Branch from an existing checkpoint:

```sh
.venv/bin/python -m agents.nn.launch \
  --name library-act1 --pool library --pool-file /tmp/library-act1.json \
  --init-from agents/nn/runs/starter/best.pt --extend-vocab --hours 0.333333
```

`--extend-vocab` preserves existing IDs and embedding rows, initializing only appended rows.
Library vocabulary warm-up covers every training scenario at least three times with different seeds. Afterwards,
unseen IDs share the unknown embedding. Training logs cumulative `unknown_id_counts` and
`id_lookups` across actors, including dropped episodes; evaluations record `heldout_vocab`.
These are occurrence counts. Older `unknown_ids` logs used the highest actor counter.

Each run copies the manifest and stores its digest in checkpoints. Scenario `source.you`
records the original run's HP change, win, damage taken, and turns for comparison; that run
had its own shuffle and potions. The dashboard compares validation outcomes and displays
card coverage. `heldout_cards` counts decisions where a card was playable or played.

## Rewards and metrics

Episodes stop at terminal state or when the observed turn exceeds 30. Multi-card choices
use the first `min` options and are not learned.

For `--reward hp` (starter/library default), let `taken` be the fraction of initial enemy HP
removed and `max0` the player's entry max HP:

- Win: `1 + (final_hp - entry_hp) / max0`.
- Loss or timeout: `-entry_hp / max0 + 0.5 * taken`.
- Both subtract `turn_cost_hp * (turn - 1) / max0`; the default turn cost is 0.05 HP.

The turn cost charges delay even when HP is unchanged. Its coefficient is a research
choice prompted by stalling; see [design limits](../../docs/nn-design.md) and the dated
[experiment results](../../docs/nn-experiments-2026-10-01.md).

HP changes and turn costs are paid per decision; the last decision receives the remaining
reward. This lets the value head predict remaining costs from visible state.
`--reward damage` (Insatiable default) pays 1 for a win or `0.5 * taken` otherwise.
`--reward survival` pays 1 for a win or `0.25 * taken + 0.04 * turn` otherwise.

Logs include accepted/dropped episodes by outcome and turn band, vocabulary counts, and
PPO metrics. Evaluation reports timeouts, HP change, HP loss among wins, and p95/maximum
turns. HP change includes healing; HP loss sums positive drops between observations, so
damage and healing within a single step can cancel.

## Checkpoint comparison

Compare two checkpoints on identical seeds. A fresh seed prefix provides a separate
comparison from the validation seeds used to select checkpoints:

```sh
.venv/bin/python -m agents.nn.evaluate \
  agents/nn/runs/starter/snapshots/it00000.pt agents/nn/runs/starter/best.pt \
  --seed-prefix FINAL-1 --output /tmp/comparison.json
```

The report includes per-seed outcomes, summaries, paired HP/turn differences, and wins
gained/lost. For library checkpoints, `--split test` is the default; `--pool-file` can
explicitly override the saved manifest. For HP scoring, `--turn-cost-hp` defaults to 0.05.

## Implementation and benchmark

[`encode.py`](encode.py) encodes the [visible observation](../../docs/what-we-give-it.md).
[`model.py`](model.py) uses a set transformer to score legal-action pointers and estimate value.
Vocabulary warm-up precedes training and uses training fights only.

Actors batch inference across workers and overlap game steps. Starts disable hashes and
reuse act maps; worker tests check those options against normal starts. The learner buckets
samples by length and trims padding per minibatch. Episodes whose starting weights are more
than `--max-lag` updates old are dropped. `--no-bucket`, `--envs`, `--actors`, `--max-lag`,
`--epochs`, and `--amp bf16` configure these choices.

[`bench.py`](bench.py) trains on the Insatiable for eight minutes, excluding warm-up, then
evaluates the last checkpoint on 200 fixed seeds and the recording. Results go to ignored
`agents/nn/bench/results.jsonl`; `ratchet.json` stores an accepted result. Lower mean enemy HP
is better; throughput is reported separately.

```sh
.venv/bin/python -m agents.nn.bench --note "trial"
.venv/bin/python -m agents.nn.bench --note "1 epoch" -- --epochs 1
.venv/bin/python -m agents.nn.bench --accept "baseline A"
```
