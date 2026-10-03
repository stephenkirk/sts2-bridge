# What a combat policy sees

`state["obs"]` is the combat view; `legal` lists actions and `choice` describes a pending card pick.
The [basic example](../examples/basic_policy.py) scores card numbers; the
[neural agent](../agents/nn/README.md) learns from card IDs and instance state.

| Field | Contents |
|---|---|
| Player | Character, HP, block, energy, stars, potions, powers, orb capacity, and Osty's HP (Necrobinder) |
| Hand | Cards in hand order, with cost, playability, upgrades, enchantments, and displayed numbers |
| Draw, discard, exhaust | The same card records, sorted so pile order is not exposed |
| Orbs | Orb state in queue order, front first |
| Enemies | HP, block, powers, current move, and intent damage |
| Relics | IDs, displayed counters, and used-up state |
| `this_turn` | Cards played and drawn this turn, from the game's combat history |
| Turn | Current turn number |

Orb order matters for Quadcast; discard contents matter for Hologram.
The [hand-played win](../tests/test_hand_win.py) is a useful example of both.

`legal` lists available plays, potion uses, end turn, and single-card choices. A choice also says
what it is for (`screen`, `prompt`, `source`): Survivor's discard and Hologram's fetch both offer
cards, and call for opposite picks. Multi-card choices use `choice.options`, `min`, and `max`;
send distinct option indices as `{"type": "choose", "picks": [...]}`. See the [worker interface](combat-parity.md#step-a-combat).

The worker also returns hashes and checkpoints for debugging. Leave those out of policy inputs
to keep the policy working from what a player can see. Draw order and RNG state aren't part of `obs`.
You can derive features such as expected block or lethal damage from the visible fields.

Known top-deck cards aren't tracked yet: sorting loses that information. Card descriptions
or rule features could also help a policy work across decks; the neural example uses IDs
and displayed numbers across its fight pools.
