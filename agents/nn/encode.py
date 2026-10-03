"""Encode visible combat state as tokens and legal actions as token pointers.

Tokens carry kind, namespaced IDs, categorical fields, and numbers. Orb positions and
enemy ownership are explicit; piles have no positional encoding. All processes share
one frozen vocabulary: 0 is absent, 1 is unknown. Combat instance IDs are not encoded.
"""

import math
from collections import Counter

import numpy as np

import torch

KINDS = ["global", "card", "orb", "enemy", "power", "relic"]
N_SUB = 4
N_NUM = 16
MAX_VARS = 6


def sl(x):
    """Symmetric log, for amounts that range from 1 to hundreds."""
    return math.copysign(math.log1p(abs(x)), x)


class Vocab:
    """Namespaced strings to indexes. 0 is none, 1 is unknown (only once frozen)."""

    def __init__(self, table=None, frozen=False):
        self.table = dict(table or {})
        self.frozen = frozen
        self.unknown = 0
        self.unknown_counts = Counter()
        self.lookups = 0

    def __call__(self, s):
        if s is None:
            return 0
        self.lookups += 1
        i = self.table.get(s)
        if i is None:
            if self.frozen:
                self.unknown += 1
                self.unknown_counts[s] += 1
                return 1
            i = self.table[s] = len(self.table) + 2
        return i

    def __len__(self):
        return len(self.table) + 2

    def diagnostics(self):
        return dict(lookups=self.lookups, unknown=self.unknown, counts=dict(self.unknown_counts))


def encode(state, v):
    """-> (tokens, actions). tokens: dict of per-token lists. actions: (a, b) token indexes per legal input in
    order, b = -1 when the input names one token."""
    o, p = state["obs"], state["obs"]["player"]
    kind, ident, sub, nums, varn, varv = [], [], [], [], [], []

    def add(k, i, subs=(), xs=(), vars_=None):
        kind.append(KINDS.index(k)); ident.append(v(i))
        s = [v(x) for x in subs][:N_SUB]; sub.append(s + [0] * (N_SUB - len(s)))
        xs = [float(x) for x in xs][:N_NUM]; nums.append(xs + [0.0] * (N_NUM - len(xs)))
        items = list((vars_ or {}).items())[:MAX_VARS]
        varn.append([v("var:" + n) for n, _ in items] + [0] * (MAX_VARS - len(items)))
        varv.append([sl(x) for _, x in items] + [0.0] * (MAX_VARS - len(items)))
        return len(kind) - 1

    orbs = p["orbs"] or []
    osty = p.get("osty")
    cap = p.get("orb_capacity") or 0
    # Choice context distinguishes discard, fetch, and other selections with similar options.
    ctx = state["choice"] or {}
    glob = add("global", "global", ["char:" + str(p.get("character")), ctx.get("screen") and "screen:" + ctx["screen"],
                                    ctx.get("prompt") and "prompt:" + ctx["prompt"],
                                    ctx.get("source") and "source:" + ctx["source"]], xs=[
        p["hp"] / 50, sl(p["hp"]), p["max_hp"] / 50, p["block"] / 20, sl(p["block"]), p["energy"] or 0,
        p["max_energy"] or 0, (o["turn"] or 0) / 10, cap / 5, len(orbs) / 5, len(o["draw"]) / 20, len(o["discard"]) / 20,
        # Regent stars and Necrobinder's Osty HP.
        (p.get("stars") or 0) / 5, osty is not None and osty["alive"], (osty or {}).get("hp", 0) / 20,
        (osty or {}).get("max_hp", 0) / 20])

    def card(c, zone):
        e, cost = c.get("enchantment"), c["cost"]
        return add("card", "card:" + c["id"],
                   ["zone:" + zone, "type:" + c["type"], "target:" + c["target"], e and "ench:" + e["id"]],
                   [c["upgrades"], -1 if cost is None else cost, c["x_cost"], c["playable"], e["amount"] if e else 0],
                   c.get("vars"))

    hand = [card(c, "hand") for c in o["hand"]]
    for zone in ("draw", "discard", "exhaust"):
        for c in o[zone]:
            card(c, zone)
    for zone in ("played", "drawn"):
        for c in o["this_turn"][zone]:
            card(c, zone)
    choice = [card(c, "choice") for c in (state["choice"] or {}).get("options", [])]

    for i, orb in enumerate(orbs):
        add("orb", "orb:" + orb["id"], [f"pos:{i}"], [
            orb["passive"] / 10, sl(orb["passive"]), orb["evoke"] / 20, sl(orb["evoke"]), i == 0, len(orbs) >= cap])

    enemy_tok = {}
    for e in o["enemies"]:
        it = e["intent"] or {"move": None, "intents": []}
        dmg = sum(x.get("damage", 0) * x.get("hits", 1) for x in it["intents"])
        hits = sum(x.get("hits", 0) for x in it["intents"] if "damage" in x)
        types = ["intent:" + x["type"] for x in it["intents"]][:2]
        # The slot is shared with the enemy's powers, so attention can tell which enemy has which.
        slot = f"slot:{e['slot']}"
        enemy_tok[e["slot"]] = add("enemy", "monster:" + str(e["id"]), ["move:" + str(it["move"]), *types, slot], [
            e["hp"] / 100, sl(e["hp"]), e["max_hp"] / 100, e["block"] / 20, sl(e["block"]), dmg / 20, sl(dmg), hits,
            e["alive"]])
        for pw in e["powers"]:
            add("power", "power:" + pw["id"], ["owner:enemy", slot], [pw["amount"] / 10, sl(pw["amount"])])
    for pw in p["powers"]:
        add("power", "power:" + pw["id"], ["owner:player"], [pw["amount"] / 10, sl(pw["amount"])])
    for r in p["relics"]:
        add("relic", "relic:" + r["id"], (), [(r["counter"] or 0) / 10, r["counter"] is not None, r["used_up"]])

    actions = []
    for a in state["legal"]:
        if a["type"] == "play":  # a target is an index into obs.enemies, i.e. an enemy's slot
            actions.append((hand[a["hand"]], -1 if a["target"] is None else enemy_tok[a["target"]]))
        elif a["type"] == "end_turn":
            actions.append((glob, -1))
        elif a["type"] == "choose":
            actions.append((choice[a["picks"][0]], -1) if a["picks"] else (glob, glob))
        else:
            raise ValueError(f"no encoding for legal input {a}")
    return dict(kind=kind, ident=ident, sub=sub, nums=nums, varn=varn, varv=varv), actions


def collate(batch):
    """Pad token/action batches; True masks mark real tokens and legal actions.

    Flatten fields before scattering into padded arrays to avoid per-row tensor writes.
    """
    B = len(batch)
    lens = np.array([len(t["kind"]) for t, _ in batch])
    alens = np.array([len(a) for _, a in batch])
    T, A = int(lens.max()), int(alens.max())
    mask = np.arange(T) < lens[:, None]
    act_mask = np.arange(A) < alens[:, None]
    out = dict(mask=torch.from_numpy(mask), act_mask=torch.from_numpy(act_mask))
    for k, width, dtype in (("kind", 0, np.int64), ("ident", 0, np.int64), ("sub", N_SUB, np.int64),
                            ("nums", N_NUM, np.float32), ("varn", MAX_VARS, np.int64), ("varv", MAX_VARS, np.float32)):
        arr = np.zeros((B, T, width) if width else (B, T), dtype=dtype)
        arr[mask] = np.array([x for t, _ in batch for x in t[k]], dtype=dtype)
        out[k] = torch.from_numpy(arr)
    act = np.full((B, A, 2), -1, dtype=np.int64)
    act[act_mask] = np.array([x for _, a in batch for x in a], dtype=np.int64).reshape(-1, 2)
    out["act"] = torch.from_numpy(act)
    return out
