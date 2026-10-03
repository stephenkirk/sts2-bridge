# Combat network results, 2026-09-30–10-01

These are dated observations, not operating prerequisites. Checkpoints, run logs, and
`.context` comparison files mentioned below are ignored local artifacts and are not
available in a fresh checkout. See the [usage guide](../agents/nn/README.md) for reproducible
commands and [design limits](nn-design.md) for the scope of the evidence.

[Earlier runs](nn-experiments-2026-09-30.md); [commands](../agents/nn/README.md).
Figures checked against local logs and paired comparisons.

## Turn cost stops the Slimes stall — 2026-09-30

`starter-turn-cost`: pre-stall iteration 340, fresh optimizer, 0.05 HP charged per advanced
turn. Other settings unchanged. The Slimes stall did not recur.

34 validation evaluations, zero timeouts; all training episodes, including dropped ones,
won within 20 turns. Stopped at 15 minutes / 81,360 accepted fights. Selected iteration 280;
continuation checkpoint 339.

A fresh paired check used 40 fights × 25 seeds (`FINAL-2026-09-30-TURN-COST`):

| Metric | Original iteration 340 | Turn-cost iteration 280 |
|---|---:|---:|
| Wins | 1,000/1,000 | 1,000/1,000 |
| Net HP lost | 4.154 | 3.867 |
| HP loss before healing* | 5.354 | 5.067 |
| Mean turns | 5.233 | 5.079 |
| p95 turns | 8 | 7 |

The stall fix mattered most; 0.287 HP saved was secondary. Five human rematches regressed
from 16 to 17 HP lost. Starting before the stall tests prevention, not unlearning; additional
training and optimizer reset also prevent isolating the penalty's effect.

Lag dropped 43.4% of episodes; no failures occurred to test failure-specific dropping.

*Sum of observed HP drops; same-step healing can mask damage.

## Broader loadouts and harder fights — 2026-10-01

`library-act1`: turn-cost iteration 280, fresh optimizer, vocabulary extended on training
fights. Inferred real Act 1 loadouts, now including regular fights and elites.
Train 240 scenarios / 37 runs; validation 83 / 12; test 83 / 12. Twenty minutes / 71,280
accepted fights; validation selected iteration 200, continuation checkpoint 297.

Initial versus selected weights: 83 test scenarios × ten fresh `LIBRARY-FINAL-1` seeds.

| Test metric | Initial | Selected |
|---|---:|---:|
| Wins | 613/830 (73.9%) | 719/830 (86.6%) |
| Net HP lost | 19.978 | 15.919 |
| Regular wins | 73.6% | 92.0% |
| Elite wins | 47.9% | 68.6% |
| Weak wins | 100% | 100% |
| Weak net HP lost | 3.717 | 4.090 |
| Timeouts | 0 | 0 |
| Longest fight | 12 turns | 28 turns |

125 wins gained, 19 lost; 2.426 HP saved on 594 shared wins. Weak-fight preservation
regressed; Regent barely improved. Elites remained difficult.

Evidence covers reconstructed Act 1 fights; no full-run or from-scratch comparison.
830 outcomes share 83 scenarios and 12 runs. Test is inspected; future tuning needs a new holdout.

Unknown IDs were 0.0718% of training lookups and 2.128% in the selected validation evaluation.
Next: inspect train/validation failure traces for missing observations, IDs, and tactical errors.
