# Neural-agent data contracts

These are repo-local interfaces. Update their owners and consumers together when changing a field.
The worker's combat spec and observation are described in [combat-parity](../../docs/combat-parity.md)
and [policy inputs](../../docs/what-we-give-it.md).

## Ownership

| Module | Responsibility |
|---|---|
| `fights.py` | Seed-independent fight pools and seeded combat specs |
| `contracts.py` | Character identity and compatibility with old evaluation logs |
| `episode.py` | Start/stop, choices, policy execution, scoring, and episode results |
| `metrics.py` | Summary and acceptance aggregation |
| `train.py` | Actor/evaluator processes, PPO updates, checkpoint selection |
| `experiment.py`, `experiments/` | Resolve and record declarative research conditions |
| `evaluate.py` | Paired comparison on identical fights and seeds |
| `watch.py` | Read artifacts and adapt them for display |

## Fight

```json
{
  "name": "DEFECT/SLIMES_WEAK",
  "kind": "monster",
  "spec": {
    "character": "CHARACTER.DEFECT",
    "ascension": 10,
    "encounter": "ENCOUNTER.SLIMES_WEAK",
    "player": {"current_hp": 60}
  }
}
```

`name` is a unique key within a pool, also used as a display label. Its spelling has no
semantic role. `spec.character` owns character identity; `contracts.character` removes
only the `CHARACTER.` prefix for report keys. `kind` is the native lower-case room type
(`monster`, `elite`, `boss`); synthetic consumers may use other labels.
The spec omits its seed; `fights.spec(fight, seed)` adds it at episode start.

Optional fields: `recording` (path to an `.mcr`, used when the seed is `None`),
`sample_weight` (training only), and library provenance: `encounter_pool`, `floor_band`,
`source` (including `copy_ambiguities` and reconstruction notes). Provenance never enters the policy observation.
Library manifests retain `format_version: 1`, `pool: "library"`, and
`splits: {train: [...], validation: [...], test: [...]}`. The loader rejects unsupported
versions and empty requested splits. Library manifests record `chronology_reader_sha256`,
the hash of the `.run` reader that built them. [Reconstruction assumptions](../../docs/nn-design.md)
explain the meaning of these scenarios.

## Scoring and episode result

`episode.Scoring(reward="hp", turn_cost_hp=0.05)` is immutable and passed explicitly to
`play`, `outcome`, and `step_rewards`. Reward modes are `hp`, `damage`, and `survival`.
Negative or nonfinite costs are rejected. Turn cost applies only to HP scoring.
The CLI defaults and [formulas](README.md#rewards-and-metrics) are unchanged.

`play` returns `(steps, result)`. A recorded step contains tokens, legal-action pointers,
chosen index, and log probability. Results contain:

| Fields | Meaning |
|---|---|
| `fight`, `kind`, `character` | Pool key, room type, unprefixed character key |
| `won`, `timeout` | Terminal victory; nonterminal state beyond turn 30 |
| `reward` | Configured episode score |
| `hp`, `boss`, `turn` | Final player HP, summed living enemy HP, observed turn |
| `kept` | Final/entry HP on wins; zero on losses |
| `hp_change` | **Entry minus final HP**: positive means HP lost, negative means healing |
| `hp_lost` | Sum of positive HP drops between replies; same-step healing can mask damage |
| `encounter_pool`, `floor_band` | Optional library grouping keys, supplied together |

Actors additionally attach decision `rewards`; they sum to `reward`. This is training
bookkeeping, not part of a comparison report. The historical `hp_change` sign is retained
for existing consumers. Paired `hp_change_delta` is candidate minus baseline, so negative
means the candidate lost less HP.

## Summary and evaluation artifacts

`metrics.summary` owns means (`win`, `kept`, `reward`, `boss`, `turn`, `hp_change`),
`n`, `timeouts`, `hp_change_wins`, `hp_lost_wins`, `turn_p95`, `turn_max`, and grouped
`by_kind`/`by_char` summaries. Library summaries add `by_pool`/`by_floor_band`.
`fights=True` adds `by_fight` with win, kept, HP change, timeout count, and p95 turns.
P95 uses nearest rank. Empty means are NaN; empty turn tails are null. JSON artifacts
currently use Python's NaN serialization, so they are not strict JSON for empty groups.

New training evaluations write one `heldout` summary under `eval`, alongside `version`,
`episodes`, `eval_s`, `heldout_cards`, and `heldout_vocab`. Recording evaluations add
`recorded_greedy` and `recorded_sampled_win`. `heldout_cards` maps character to card ID to
`[decisions playable, decisions played]`; duplicate copies count once per decision.
`heldout_vocab` contains `lookups`, `unknown`, and per-ID `counts` for that evaluation.

Old logs may contain flattened `heldout_*` metrics, sometimes alongside `heldout`.
`contracts.heldout` is the single read adapter; nested summaries take precedence.
New writers do not duplicate metrics. Existing checkpoints and library manifests need
no migration; external log consumers should read `eval.heldout` through this adapter.

Comparison reports contain checkpoint paths, pool/split, seed prefix, turn cost,
`summaries: {baseline, candidate}`, `paired` aggregate differences and win changes,
and `pairs: [{fight, seed, baseline: result, candidate: result}]`. They score both policies
under the same explicit configuration, rather than either checkpoint's training objective.

Checkpoint architecture, vocabulary IDs, weights, optimizer, counters, scoring settings,
and library digest retain their existing shape. `best.pt` copies the exact evaluated
snapshot and attaches `eval`; the current learner weights may already be newer.

## Experiment provenance

Definitions and resolution rules are documented in [the experiment guide](experiments/README.md).
`experiment.json` stores `path`, `sha256`, `definition`, `resolved`, and `selection_metric`.
Session headers optionally include the same `experiment` record. Checkpoints optionally
include `experiment: {id, sha256}`; older checkpoints and ad hoc runs need neither field.
The definition digest identifies the condition file; resolved session arguments identify
CLI overrides. Model architecture inherited on branch/resume lives in checkpoint `cfg`.
