# Combat network experiments, 2026-09-27–30

These are dated observations, not operating prerequisites. Checkpoints, run logs, and
`.context` comparison files mentioned below are ignored local artifacts and are not
available in a fresh checkout. See the [usage guide](../agents/nn/README.md) for reproducible
commands and [design limits](nn-design.md) for the scope of the evidence.

[Later results](nn-experiments-2026-10-01.md); [commands](../agents/nn/README.md). Raw artifacts remain local.

## Insatiable only — 2026-09-27

`seeds`: the Insatiable boss, one Defect A10 loadout, fresh seeds each episode to vary
draws and rolls. Two hours / 123k fights reached 9% held-out wins. The original recording
still killed it on turn 5, leaving 52–96 boss HP.

First 4 turns, it played identical to the hand-played win.
At 5 HP it used Hologram+ for Turbo+ instead of Boot Sequence+: energy over block.
Damage reward appeared to discourage surviving to the turn-7 win; survival reward did not fix it.

Two fixed-recording checks, `smoke` (14k fights) and `smoke-survival` (9k), reached 0%
held-out wins. Both died on turn 5 of the recording, leaving 60 boss HP.

## Faster training — 2026-09-28

Faster workers, batched inference, and bucketing roughly doubled throughput: 499–502
to 884–1,077 decisions/s. Eight-minute Insatiable scores stayed within baseline variation
(151.2–155.7 boss HP before; 150.8–156.3 after). `baseline A` remained accepted.

## Starter baselines — 2026-09-29

Five characters × eight Act 1 weak encounters, A10 starter loadouts, fresh seeds.
Random card spending won all 400 validation fights; HP preservation mattered more.

| Character | Spend HP lost | Random HP lost (losses/80) | User's first-fight HP lost |
|---|---:|---:|---:|
| Defect | 12.1 | 28.5 (0) | 3.6 |
| Ironclad | 15.1 | 35.2 (24) | 7.4 |
| Necrobinder | 12.3 | 31.8 (28) | 1.6 |
| Regent | 14.0 | 34.3 (19) | 4.6 |
| Silent | 15.1 | 34.5 (31) | 2.7 |

HP loss excludes losses. Human baseline: first fights of 210 A10 runs, Neow gift, 11-card deck.

## `starter-20m`: learns the fights, then stalls on Slimes — 2026-09-29–30

A fresh network trained across those 40 fights with HP reward and no turn cost:
445 updates / 106,800 accepted fights in 20.1 minutes. Iteration 340 won all 400 validation
fights and retained 93.5% of entry HP on average.

Later greedy policies stalled against Slimes. At iteration 430, Necrobinder won 7/10
Slimes seeds and Silent 8/10, both down from 10/10 at iteration 340. Accepted training
batches still reported 100% wins, hiding the evaluation regression.

Reward hacking: HP loss cost reward; delaying the kill was free.
[The fix](nn-experiments-2026-10-01.md#turn-cost-stops-the-slimes-stall--2026-09-30):
restart from iteration 340 and charge 0.05 HP per advanced turn.

Source: `agents/nn/runs/starter-20m/log.jsonl`.
