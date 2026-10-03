"""Shared combat execution and explicit research scoring; no training process state."""

from dataclasses import dataclass
import math

import torch

from . import fights as pools
from .contracts import character
from .encode import collate, encode

MAX_TURNS = 30


@dataclass(frozen=True)
class Scoring:
    reward: str = "hp"
    turn_cost_hp: float = 0.05

    def __post_init__(self):
        if self.reward not in {"hp", "damage", "survival"}:
            raise ValueError("unknown reward mode")
        if not math.isfinite(self.turn_cost_hp) or self.turn_cost_hp < 0:
            raise ValueError("turn_cost_hp must be finite and nonnegative")


def send_start(worker, fight, seed):
    """Load the recording when seed is None; otherwise start the spec without hashes and reuse its map."""
    if seed is None:
        worker.send("load", mcr=fight["recording"], hashes=False)
    else:
        worker.send("start", spec=pools.spec(fight, seed), hashes=False, reuse_map=True)


def start(worker, fight, seed):
    send_start(worker, fight, seed)
    return worker.receive()


def first(state):
    """Capture entry HP for reward normalization and outcome metrics."""
    o = state["obs"]
    return dict(boss0=sum(e["hp"] for e in o["enemies"]), hp0=o["player"]["hp"], max0=o["player"]["max_hp"])


def over(state):
    return state["boundary"] == "terminal" or state["obs"]["turn"] > MAX_TURNS


def multi_pick(state):
    """A multi-pick choice is not enumerated by the worker and not learned here: take the first `min` options."""
    return {"type": "choose", "picks": list(range(state["choice"]["min"]))}


def outcome(state, boss0, hp0, max0, *, scoring):
    """Score terminal or capped episodes under explicit scoring; see README.md for the formulas.

    HP reward uses max HP for normalization and charges scoring.turn_cost_hp per advanced turn.
    `kept` is relative to entry HP and zero for losses; `hp_change` includes healing.
    """
    o = state["obs"]
    won = state["boundary"] == "terminal" and o["player"]["hp"] > 0 and not any(e["alive"] for e in o["enemies"])
    boss = sum(e["hp"] for e in o["enemies"] if e["alive"])
    taken = (boss0 - boss) / boss0
    kept = o["player"]["hp"] / hp0 if won else 0.0
    if scoring.reward == "hp":
        reward = (o["player"]["hp"] - hp0) / max0 + (1.0 if won else -o["player"]["hp"] / max0 + 0.5 * taken)
        reward -= scoring.turn_cost_hp * max(0, o["turn"] - 1) / max0
    elif won:
        reward = 1.0
    elif scoring.reward == "survival":
        reward = 0.25 * taken + 0.04 * o["turn"]
    else:
        reward = 0.5 * taken
    return dict(won=won, reward=reward, kept=kept, hp=o["player"]["hp"], boss=boss, turn=o["turn"],
                timeout=state["boundary"] != "terminal" and o["turn"] > MAX_TURNS,
                hp_change=hp0 - o["player"]["hp"])


def step_rewards(hps, res, max0, turns=None, *, scoring):
    """Distribute HP and turn costs across decisions, with the remainder at the final decision.

    Rewards sum to the episode outcome. Other reward modes pay only at the final decision.
    """
    if scoring.reward != "hp":
        return [0.0] * (len(hps) - 1) + [res["reward"]]
    r = [(b - a) / max0 for a, b in zip(hps, hps[1:] + [res["hp"]])]
    if turns is not None:
        for i, (a, b) in enumerate(zip(turns, turns[1:] + [res["turn"]])):
            r[i] -= scoring.turn_cost_hp * max(0, b - a) / max0
    r[-1] += res["reward"] - sum(r)
    return r


def fight_context(fight):
    """Scenario provenance is for metrics, never part of the policy observation."""
    return dict(character=character(fight), **{k: fight[k] for k in ("encounter_pool", "floor_band") if k in fight})


def count_cards(cards, state, chosen):
    """For each card id, the decisions where some copy was playable and the ones where a copy was played."""
    hand, legal = state["obs"]["hand"], state["legal"]
    for cid in {hand[a["hand"]]["id"] for a in legal if a["type"] == "play"}:
        cards.setdefault(cid, [0, 0])[0] += 1
    if legal[chosen]["type"] == "play":
        cards[hand[legal[chosen]["hand"]]["id"]][1] += 1


def play(worker, net, vocab, fight, seed, greedy=False, record=True, cards=None, *, scoring):
    """Return (steps, result) for one episode.

    Steps contain (tokens, actions, chosen index, log probability). `cards` accumulates
    playable/played counts by card ID; `record=False` omits training steps.
    """
    state = start(worker, fight, seed)
    at_start = first(state)
    steps = []
    hp_lost = 0
    while not over(state):
        hp_before = state["obs"]["player"]["hp"]
        if not state["legal"]:
            state = worker.step(multi_pick(state))
            hp_lost += max(0, hp_before - state["obs"]["player"]["hp"])
            continue
        toks, acts = encode(state, vocab)
        with torch.no_grad():
            logits, _ = net(collate([(toks, acts)]))
        dist = torch.distributions.Categorical(logits=logits[0])
        i = int(logits[0].argmax()) if greedy else int(dist.sample())
        if record:
            steps.append((toks, acts, i, float(dist.log_prob(torch.tensor(i)))))
        if cards is not None:
            count_cards(cards, state, i)
        state = worker.step(state["legal"][i])
        hp_lost += max(0, hp_before - state["obs"]["player"]["hp"])
    return steps, dict(outcome(state, **at_start, scoring=scoring), kind=fight["kind"], fight=fight["name"], hp_lost=hp_lost,
                       **fight_context(fight))
