"""Out-of-combat preferences for ``play_run``: how much the agent wants each card, relic and choice.

Pass a JSON file with ``--priors``, keyed by character (``CHARACTER.IRONCLAD``, ...). Every table is optional and
anything unrated gets a default. The numbers can come from anywhere (community stats, your own runs, a hand-written
list) as long as they use these scales:

    card     {"CARD.X": {"0": score, "1": score}}   by upgrade level, on an Elo-like scale: unrated cards are 1500,
                                                    Strike and Defend 1000, Bash 1300
    skip     {"1": score, ..., "any": score}       by act, numbered from 1, on the card scale: a card reward is taken
                                                    if its best card beats this, a shop card if it beats it by 150
    relic    {"RELIC.X": 0..1}                     unrated relics are 0.3
    ancient  {"OPTION": 0..1}                      an Ancient's options, by the last segment of the option key
    event    {"EVENT.pages.PAGE.options.OPTION": n}   higher wins, unrated is 0; keys are the ``key`` the run worker
                                                    reports
    remove   {"CARD.X": n}                         how readily to remove a card; each unit is worth 3 card points

``build`` and ``upgrade`` are read but unused. With no file every card ties with skip at 1500, so the agent skips
card rewards and buys no cards; other choices fall to the defaults above.

Why a lookup table: nothing here learns card picks yet, so outside combat the agent copies better players' homework.
The format is the shape of the first homework it copied, three unrelated scales and two unread fields included.
"""

import json
from pathlib import Path

DEFAULT_ELO = 1500.0  # an unrated card sits at the Elo scale's starting point


class Priors:
    def __init__(self, character, path=None):
        p = json.loads(Path(path).read_text()).get(character, {}) if path else {}
        self.build = p.get("build")
        self._card = p.get("card", {})
        self._skip = p.get("skip", {})
        self.relic = p.get("relic", {})
        self.ancient = p.get("ancient", {})
        self.event = p.get("event", {})
        self.upgrade = p.get("upgrade", {})
        self.remove = p.get("remove", {})

    def card(self, card_id, upgrades=0):
        by_upg = self._card.get(card_id, {})
        return by_upg.get(str(upgrades), by_upg.get("0", DEFAULT_ELO))

    def skip(self, act):
        """``act`` is the game's 0-based act index; the file numbers acts from 1."""
        return self._skip.get(str(act + 1), self._skip.get("any", DEFAULT_ELO))
