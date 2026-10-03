# Experiment conditions

An experiment is a hypothesis and a set of conditions. A run is one execution of those
conditions. A comparison is evidence about two resulting checkpoints. Shared execution,
encoding, models, and PPO stay in the parent directory; these files only declare conditions.

| Definition | Question | Required local inputs |
|---|---|---|
| `insatiable.json` | Can damage reward learn the pinned boss loadout across fresh shuffles? | None |
| `starter-hp.json` | Can HP reward improve starter-deck play? | None |
| `starter-turn-cost.json` | Can a turn cost prevent stalling from pre-stall weights? | `--init-from` |
| `library-act1.json` | Does a starter-deck policy improve on real Act 1 loadouts, regulars, and elites without losing weak fights? | `--init-from`, `--pool-file` |

These are runnable protocols for the research branches in the dated notes. They are not
complete reproductions of historical runs: source artifacts, hardware, exact stopping time,
and earlier code versions differ. Comparison prefixes are new namespaces; use a new one
for each independent final comparison. Existing test outcomes have already been inspected.

## Execute and inspect

From the repository root:

```sh
# Inspect conditions and all CLI defaults without workers or output files.
.venv/bin/python -m agents.nn.train --experiment starter-hp --name hp-trial --device cpu --dry-run

# Execute with the same shared runner used by direct training commands.
.venv/bin/python -m agents.nn.train --experiment starter-hp --name hp-trial --device cpu

# The dashboard launcher passes experiment settings through to training.
.venv/bin/python -m agents.nn.launch --experiment starter-turn-cost --name turn-cost-trial \
  --init-from /path/to/pre-stall.pt

.venv/bin/python -m agents.nn.train --experiment library-act1 --name library-trial \
  --init-from /path/to/starter-best.pt --pool-file /path/to/library.json --dry-run
```

`--experiment` accepts a built-in name or a JSON file path. Paths supplied on the command
line are relative to the working directory. CLI options override file conditions; changes
are visible in the resolved configuration. Runtime device and thread settings use the
normal CLI defaults unless overridden. Branches inherit model architecture and vocabulary
from their source checkpoint; `d` and `layers` configure fresh networks only.

A run writes `experiment.json` with the definition, its SHA-256, resolved CLI arguments,
and checkpoint-selection metric. Every session header in `log.jsonl` records experiment
provenance; checkpoints carry the experiment ID and definition digest. The original
`experiment.json` stays fixed on resume, while session headers record new CLI overrides.
An ordinary `--resume` without `--experiment` retains the run's original experiment identity.
Resume keeps the existing reward/evaluation compatibility checks. A changed definition
requires a new branch. Attaching a definition to an existing ad hoc run also requires
a branch. Library resumes still need the manifest, but do not re-extend the
vocabulary or require the original initialization checkpoint.

Compare checkpoints explicitly with the shared tool:

```sh
.venv/bin/python -m agents.nn.evaluate \
  /path/to/baseline.pt /path/to/candidate.pt \
  --seed-prefix TURN-COST-FINAL-NEW --seeds 25 --output /tmp/comparison.json
```

The `comparison` block states the intended baseline, seed namespace, seed count, and
library split. It is a reviewable plan; training does not automatically run or interpret it.
For turn-cost work, compare to the supplied source checkpoint. For library transfer,
compare to `snapshots/it00000.pt`, which includes newly initialized embedding rows.
Evaluation scores both checkpoints with the same objective; HP comparisons default to
0.05 turn cost, so use `--turn-cost-hp 0` when assessing the original HP-only objective.

## Definition shape

Version 1 requires `format_version`, `id`, `hypothesis`, and `training`.
`training` contains CLI setting names with underscores. Unknown settings and invalid CLI
values fail before execution. No configuration inheritance or executable Python is involved.
`name`, `resume`, runtime paths (`init_from`, `pool_file`), and experiment-runner flags
belong on the CLI, rather than in the definition. `required_inputs` declares either or both
runtime paths. Optional `comparison` documents the independent evaluation plan.

When adding a research condition, copy a definition, change its ID and hypothesis, then
edit the relevant values. Repetition here lets a reader inspect the conditions in one file.
Put new algorithms in shared modules when needed, and record findings in dated docs with
links to the condition file and run artifacts.
