# Insatiable revisited, 2026-10-05–09

Dated observations; checkpoints, logs and `.context` scripts are local. [Earlier results](nn-experiments-2026-10-01.md).

## Backdrop

This began as a question about my own loss: what did I get wrong on the turn where the old agent went for Turbo?
Tracing the orbs answered it, then turned into a question about the reward that taught the agent that move.
Under `damage` a loss pays up to 0.5 against 1 for a win; maybe that gap is too small. So: try paying 1 for a win and 0 for a loss.
The plan was to fine-tune the 9% checkpoint under a win-only reward. That checkpoint turned out to be deleted.
A fresh win-only run sampled no wins; the session ended before its first evaluation.
Rejected: survival reward, a small damage consolation, and a "a surviving line existed" metric, which needs a search per death.
Rejected: training from turn 5 of the recorded fight. The original run shuffled so it couldn't learn the draw order; a pinned deal undoes that.
Couldn't do: fresh shuffles at turn 5 (no reseed command), or a 20-minute damage control (no evaluation logs survive).
The library checkpoint was meant as a baseline, with its training history as an admitted confounder.
Untrained on this boss, with the boss itself an unknown ID, it won 7 of 100 fresh fights. That was the surprise of the session.
On my deal it plays turns 1–4 like my win, better than my loss, then fails turn 5 differently from both.

## Turn 5 of the recorded fight

The lost damage-reward agent matched the hand-played win for four turns, then used Hologram+ for Turbo+.
At 5 HP against Thrash 12×2 no line kills the boss, so that line dies. Synchronize+ gives +8 Focus (four orb types).
Win: Boot Sequence+ for block, then Quadcast evokes the starting Lightning 4×16; with passives, 97 damage.
Recorded loss: Coolheaded+ evicted that Lightning once for 16, leaving no energy for Quadcast; 49 damage.
Its second Frost still blocked 25 of 24. The player skipped Boot Sequence+ because it exhausts and Frost was enough; the cost was damage.

## Rewards

`damage` ranks every loss below every win, yet prefers dying at 80% damage (0.40) to 9% wins otherwise at 60% (0.363).
`survival` keeps that consolation and pays per turn, the opposite of the Slimes turn cost; Sandpit caps survival anyway.
GAE λ 0.95 is per decision: the terminal TD error keeps 0.60 of its weight at 10 decisions, 0.21 at 30.
Added `--reward win`: 1 for a win, 0 otherwise.

## Fresh `win` run — stopped

`insatiable-no-consolation` ended after 3 minutes with the session: 61 updates, 14,640 fights, no sampled wins.
Value loss fell to ~1e-6; with no wins there is nothing to learn from. No evaluation completed.
The 9% checkpoint is deleted. The 09-28 eight-minute damage runs sampled 0.4–1.1% wins.

## Turn-5 starts — dropped

`--start-turn 5` replays the hand-win prefix. A random policy won 0/400 from there; 360 died on turn 5.
The deal is pinned, so training there memorises one tree. Fresh shuffles need a reshuffle/reseed command first.

## Library checkpoint, never trained on the boss

`library-act1` iteration 200 on 100 fresh seeds: greedy 7/100 (≈3–14%), sampled 4/100. 8.9% of lookups
unknown, including the boss, Sandpit, Quadcast and Synchronize. Wins ended on 2–19 HP; LIBPROBE-27 won on turn 8 with 19.
On the recorded deal: 0/100 sampled; greedy matched the hand-played win for four turns, then died on turn 5.
It played Quadcast before Synchronize+ and fetched Refract+ instead of blocking. p(Boot Sequence+) 0.005; p(Synchronize+ first) 0.05.

Planned: `damage` and `win` arms from these weights with `--extend-vocab`, 20 minutes, compared at matched fights.
The shared start cancels between arms; results describe refining a generalist, not the deleted checkpoint.

## Two arms, one hour (2026-10-06)

Both arms started from `library-act1-it200-insatiable.pt`, with new IDs copied from the unknown row, so iteration 0 is identical.
`agents.nn.pair`, `--actors 3`, `--hours 1`: about 64k fights each, 27 evaluations each on the 100 `EVAL` seeds.
Iteration 0 won 3/100 `EVAL` seeds (the 7/100 above was on `LIBPROBE` seeds).
For 30 minutes the arms were indistinguishable. After about 40k fights, `win` evaluations scored 8–12, `damage` 4–8.
Last five evaluations pooled: `win` 49/500, `damage` 30/500. These replay the same seeds, so they aren't independent.
Sandpit doesn't deal damage: its removal kills before the attack, so a final observation without it marks a Sandpit loss.
`damage` ended with 32–39 Sandpit losses per 100 and 62% of boss HP removed in losses: damage dealt until the clock runs out.
`win` cut Sandpit losses from 36 to 16–21 and survived more lethal turns (58% vs 50%), dealing less damage in losses.
Recorded deal: `win` survived turn 5 from 38k fights on and lost on turn 7 in its last six evaluations, boss on 22, 16, 15, 16, 1, 24.
`damage` died on turn 5 in every evaluation of the run.
Turn-5 probe, after both swapped back and forth early on: `win` took Hologram+ → Boot Sequence+, `damage` Quadcast after Synchronize+ (0.94).
Each arm learned half of the hand-played turn 5: the block, or the evoke. `win`'s line skipped Quadcast and dealt 44 to the win's 97.
Synchronize+ order is irrelevant there: Hologram+ and Boot Sequence+ don't touch orbs. The missed decision is Quadcast.
Watching p(Synchronize+ first) was a mistake I only saw post hoc; the probe now records p(Quadcast after Synchronize+).

## Untouched seeds and the winnable bank (2026-10-08)

200 `INSATIABLE-LIBRARY-FINAL-NEW` seeds, greedy, played once by five checkpoints:
start 5, `win` iteration 260 14, `damage` iteration 260 9, `win` best (iteration 230) 17, `damage` best (iteration 220) 14.
Exact McNemar: `win` vs start p 0.035 and 0.004; `damage` vs start 0.34 and 0.049; `win` vs `damage` 0.33 and 0.66.
Lethal turns survived: start 45%, `win` 58–59%, `damage` 51–53%. The arms share 3 of their 20 wins at iteration 260.
Bank: the 54 greedy evaluations won 30 of the 100 `EVAL` seeds at least once; any one late snapshot won 6–12.
Twenty sampled attempts per bank seed: start 42/600, `win` 144/600, `damage` 109/600; on the 17 seeds both arms found, 38, 124, 82.
Each arm stopped winning about 8 seeds it had won earlier in training. Sampling wins far more of the bank than greedy play.
The bank is selected by these arms' own wins, and the `EVAL` seeds are now curriculum. `FINAL-NEW` has been used once.
