"""A small greedy combat policy.

Run from the repository root: python3 -m examples.basic_policy
Uses visible card numbers and enemy intents; ignores most card interactions.
"""

from sts2bridge import CombatWorker
from sts2bridge.fixtures import MCR


def policy(state):
    """Pick a legal play, or answer a card choice with its first required options."""
    if state["boundary"] == "awaiting_choice":
        choice = state["choice"]
        return {"type": "choose", "picks": list(range(choice["min"]))}

    obs = state["obs"]
    enemies = {e["slot"]: e for e in obs["enemies"] if e["alive"]}
    incoming = sum(
        (intent.get("damage") or 0) * (intent.get("hits") or 1)
        for enemy in enemies.values() if enemy["intent"]
        for intent in enemy["intent"]["intents"]
    )
    needed_block = max(0, incoming - obs["player"]["block"])

    def score(action):
        if action["type"] != "play":
            return 0 if action["type"] == "end_turn" else -1  # Save potions.
        card = obs["hand"][action["hand"]]
        numbers = card.get("vars") or {}
        damage = numbers.get("Damage") or numbers.get("CalculatedDamage") or 0
        block = numbers.get("Block") or numbers.get("CalculatedBlock") or 0
        value = 1.3 * min(block, needed_block)
        if action["target"] in enemies:
            enemy = enemies[action["target"]]
            remaining = enemy["hp"] + enemy["block"]
            value += min(damage, remaining) + (10 if damage >= remaining else 0)
        elif card["target"] == "AllEnemies":
            value += sum(min(damage, e["hp"] + e["block"]) for e in enemies.values())
        if card["type"] == "Power":
            value += 10 if obs["turn"] <= 3 else 3
        elif not damage and not block:
            value += 2  # Try utility cards without modelling their effects.
        cost = card["cost"] if card["cost"] is not None else obs["player"]["energy"]
        return value / (1 + max(0, cost) * 0.4)

    return max(state["legal"], key=score)


def main():
    with CombatWorker() as worker:
        state = worker.load(MCR)
        for _ in range(500):
            if state["boundary"] == "terminal":
                break
            action = policy(state)
            print(f"Turn {state['obs']['turn']}: {action}")
            state = worker.step(action)
        else:
            raise RuntimeError("Stopped after 500 actions; the policy may be looping.")
        obs = state["obs"]
        won = obs["player"]["hp"] > 0 and not any(e["alive"] for e in obs["enemies"])
        print(f"{'Won' if won else 'Lost'} on turn {obs['turn']}; HP: {obs['player']['hp']}")


if __name__ == "__main__":
    main()
