# Insatiable revisited, 2026-10-05–06

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

Next: `damage` and `win` arms from these weights with `--extend-vocab`, 20 minutes, compared at matched fights.
The shared start cancels between arms; results describe refining a generalist, not the deleted checkpoint.
