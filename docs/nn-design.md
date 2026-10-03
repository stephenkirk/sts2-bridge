# What is guaranteed, and what is an experiment?

The neural agent is an experimental policy on a tested bridge. Its reusable episode and
artifact interfaces live in [contracts.md](../agents/nn/contracts.md); operating commands
live in the [agent README](../agents/nn/README.md). Dated measurements belong in
[September 30 notes](nn-experiments-2026-09-30.md) and
[October 1 notes](nn-experiments-2026-10-01.md).

## Evidence boundaries

| Area | Dependable contract | Provisional choice or limit |
|---|---|---|
| Game execution | Replay and stepping match all 49 pinned Insatiable checksums | One recording is the parity witness; other encounters have smoke coverage |
| Episode execution | Explicit scoring; consistent termination and metrics across training/comparison | Turn cap 30; multi-pick takes the first minimum options |
| Policy input | Visible observations and legal-action pointers | ID embeddings and set transformer; no potion encoding; unknown IDs share an embedding |
| Checkpoints | Best weights correspond to the evaluated version; vocabulary transfer preserves old rows | Validation objective chooses best: reward for starter/library, win rate for Insatiable |
| Evaluation | Paired checkpoints use identical specs and seeds | Seed counts and namespaces; validation reused for selection is not independent test evidence |
| Curriculum | Deterministic whole-run splits and removal of exact cross-split inventory overlap | A10, 80% starter HP, 20/50/30 encounter weighting, floor bins, two scenarios per selection bucket |

Tests protect execution, metrics, transfer, reconstruction, and split invariants. They do
not establish that the reward is strategically correct, that inferred fights recreate
history, or that a policy generalizes to full runs. Turn cost is a research response to
observed stalling; its coefficient and success claims are scoped to the dated experiments.

## Experiments, runs, and comparisons

[Condition files](../agents/nn/experiments/README.md) define the hypothesis and settings for
each research branch. The shared trainer resolves them through its existing CLI parser;
CLI overrides remain possible and are recorded with each session. One definition can
produce many runs, and the comparison tool assesses checkpoints independently of training.
Dated notes interpret the evidence. Runtime checkpoints and manifests are explicit inputs
so the definitions work without private artifact paths baked into the repository.

## Why this shape

The bridge delegates combat rules to the game DLL. The policy code owns only action
selection and experimental objectives. `episode.py` shares execution/scoring without
importing the learner; `metrics.py` lets reports and dashboards agree on metric meanings.
`contracts.py` keeps identity and old-log compatibility at one boundary. The learner
retains its batched actor loop because asynchronous sends overlap game work; it calls the
same start, termination, scoring, and context helpers as serial evaluation.

Actors can receive weights during an episode, so a trajectory can contain multiple policy
versions. Lag is measured from its starting version and old episodes are dropped. This is
an approximation in the asynchronous PPO experiment, not a guarantee of on-policy data.
Accepted/dropped outcome counts expose that selection. Bucketing trims padding, disabled
hashes avoid serialization, and reused maps reduce startup overhead; worker tests check
optimized starts against normal starts. These speed choices must preserve bridge behavior.

## Inferred library scenarios

The builder reads the player's own completed `.run` saves with `sts2bridge.chronology`;
starter decks come from the worker catalog. No runs ship here. Without them the starter and
Insatiable pools still work. The library measurements came from one local run history, so
a fresh checkout cannot reproduce them.

Deck mutations advance from the starter deck and must reconcile with final inventory.
Relics come from before-node chronology; entry HP and gold come from the previous node.
Relic props reset, potions are omitted, and mixed event/combat nodes and bosses are excluded.
Card-copy ambiguity and mutation-order inference remain in each scenario's provenance.
Fresh training RNG makes these scenarios rather than historical replays. Human comparison
figures used different shuffles and potions.

The library balances characters, encounter pools, floor bands, runs, and scenarios. Small
strata may lack holdouts, and starter pretraining can overlap held-out loadouts. Test is
excluded from training and vocabulary warm-up; once inspected for tuning, it is no longer
an untouched holdout. Change these choices on a new branch with recorded configuration
and a fresh evaluation namespace. Keep results and local artifact paths in dated notes.
