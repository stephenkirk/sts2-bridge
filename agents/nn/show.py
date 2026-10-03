"""Print a checkpoint's line on the recorded fight, turn by turn, in the hand-win fixture's format.

    .venv/bin/python -m agents.nn.show runs/seeds/best.pt [--seed EVAL-3] [--sample]
    .venv/bin/python -m agents.nn.show runs/starter/best.pt --fight IRONCLAD/NIBBITS_WEAK
"""

import argparse

import torch

from sts2bridge import CombatWorker, card_name

from . import fights as pools
from .encode import Vocab, collate, encode
from .model import Net
from .episode import multi_pick, over, start


def load(path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    net = Net(**ck["cfg"]).eval()
    net.load_state_dict(ck["state"])
    return net, Vocab(ck["vocab"], frozen=True), ck


def policy(net, vocab, state):
    """(probabilities over state["legal"], value)."""
    toks, acts = encode(state, vocab)
    with torch.no_grad():
        logits, value = net(collate([(toks, acts)]))
    return torch.softmax(logits[0], -1), float(value[0])


def label(state, action):
    o = state["obs"]
    if action["type"] == "play":
        return card_name(o["hand"][action["hand"]])
    if action["type"] == "choose":
        return "-> " + (card_name(state["choice"]["options"][action["picks"][0]]) if action["picks"] else "none")
    return "end"


def line(worker, net, vocab, seed=None, sample=False, fight=None):
    """Play one episode, of the Insatiable unless `fight` is given. Returns (turns, end): turns are
    {turn, hp, boss, intent, moves: [(label, p)], value}."""
    state = start(worker, fight or pools.insatiable()[0], seed)
    turns = []
    while not over(state):
        o = state["obs"]
        if not turns or turns[-1]["turn"] != o["turn"]:
            boss = o["enemies"][0]
            turns.append(dict(turn=o["turn"], hp=o["player"]["hp"], boss=boss["hp"],
                              intent=boss["intent"] and boss["intent"]["move"], moves=[]))
        if not state["legal"]:
            state = worker.step(multi_pick(state))
            continue
        p, value = policy(net, vocab, state)
        i = int(torch.multinomial(p, 1)) if sample else int(p.argmax())
        turns[-1]["moves"].append((label(state, state["legal"][i]), float(p[i])))
        turns[-1]["value"] = value
        state = worker.step(state["legal"][i])
    o = state["obs"]
    alive = [e for e in o["enemies"] if e["alive"]]
    return turns, dict(hp=o["player"]["hp"], boss=sum(e["hp"] for e in alive), won=not alive and o["player"]["hp"] > 0)


def short(name):
    return name.replace("CARD.", "").replace("_DEFECT", "").replace("_", " ").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--fight", help="a starter-pool fight, e.g. IRONCLAD/NIBBITS_WEAK; default the Insatiable")
    ap.add_argument("--seed", default=None, help="a spec seed instead of the recorded fight")
    ap.add_argument("--sample", action="store_true", help="sample instead of taking the top-scored input")
    args = ap.parse_args()
    net, vocab, _ = load(args.ckpt)
    with CombatWorker() as w:
        fight = None
        if args.fight:
            fight = next((f for f in pools.pool("starter", w.catalog()) if f["name"] == args.fight), None)
            if fight is None:
                ap.error(f"no starter fight {args.fight!r}")
            args.seed = args.seed or "EVAL-0"
        turns, end = line(w, net, vocab, args.seed, args.sample, fight)
    for t in turns:
        print(f"turn {t['turn']}: hp {t['hp']}, boss {t['boss']}, intent {t['intent']}, value {t.get('value', 0):.2f}")
        print("   ", " · ".join(f"{short(m)} ({p:.2f})" for m, p in t["moves"]))
    print(f"end: {'WIN' if end['won'] else 'loss'}, hp {end['hp']}, boss {end['boss'] or 'dead'}")


if __name__ == "__main__":
    main()
