"""Python side of the combat worker: start it once, load combats, step them one chosen input at a time.

    with CombatWorker() as w:
        s = w.load(MCR)                        # first decision boundary of the recorded fight
        # or: s = w.start(spec_from_run(run, "ENCOUNTER.X", seed="ANY"))   any deck against any fight
        while s["boundary"] != "terminal":
            s = w.step(pick(s["legal"]))       # any entry of s["legal"], or a choose with picks

Every reply carries ``boundary`` (awaiting_input, awaiting_choice or terminal), ``obs``, ``legal``, ``choice``
(options when a choice is pending), ``state_hash`` (the game's NetFullCombatState hash at the boundary),
``checkpoints`` (the game's own checksums taken since the previous reply), ``enqueued_by_game`` and
``game_errors`` (error-level lines the game logged since the previous reply). ``load`` and ``start`` take
``hashes=False`` for training: ``state_hash`` comes back null and ``checkpoints`` empty, and nothing else changes.

Also here: ``recorded_action``, which turns a tape's net actions into the same caller vocabulary. It is what
the parity test feeds through ``step``.

``start_run`` / ``run_step`` keep one native RunState across rooms. Each call is recorded as JSONL at
``run_trace_path`` in the worker's scratch directory.
"""

import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "lib" / "sts2.dll"  # present once ./setup.sh has copied and patched the game's DLLs
PROJECT = ROOT / "dotnet" / "CombatWorker" / "CombatWorker.csproj"
# Release: the worker's own code (observation, legal inputs, JSON) is a third of a step in a Debug build.
# STS2_BRIDGE_WORKER points at another build, e.g. to compare two.
BINARY = Path(os.environ.get("STS2_BRIDGE_WORKER") or ROOT / "dotnet" / "CombatWorker" / "bin" / "Release" / "net9.0" / "CombatWorker.dll")

# Tape events the game produces by itself on the singleplayer net service; a caller never sends them.
GAME_DRIVEN = {"ready_to_begin_enemy_turn", "resume"}


class WorkerError(RuntimeError):
    pass


class CombatWorker:
    def __init__(self, workdir=None):
        # GodotStubs resolve user:// against the working directory, so give the worker a scratch one.
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="combat-worker-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.workdir / "worker.stderr"
        self._log = open(self.log_path, "w")
        self._proc = subprocess.Popen(["dotnet", str(BINARY)], cwd=self.workdir, text=True, bufsize=1,
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log)
        self._stdin, self._stdout = self._proc.stdin, self._proc.stdout
        self._quit_command = json.dumps({"cmd": "quit"}) + "\n"
        self._quit_if_running = lambda: self._proc.poll() is None
        self._wait_for_exit = lambda: self._proc.wait(timeout=30)
        self._close_transport = lambda: None
        self._read_eof_error = lambda: f"worker exited (rc={self._proc.poll()}); see {self.log_path}"
        self.boot_ms = self._read()["boot_ms"]
        self.run_trace_path = None
        self._run_trace = None

    def _read(self):
        line = self._stdout.readline()
        if not line:
            raise WorkerError(self._read_eof_error())
        reply = json.loads(line)
        if not reply.get("ok"):
            raise WorkerError(reply.get("error"))
        return reply

    def request(self, cmd, **fields):
        self.send(cmd, **fields)
        return self.receive()

    # A request in two halves, so one caller can keep several workers busy at once: send to each, then receive
    # from each. Every send must be matched by one receive, in order.
    def send(self, cmd, **fields):
        self._stdin.write(json.dumps({"cmd": cmd, **fields}) + "\n")
        self._stdin.flush()

    def receive(self):
        return self._read()

    def _worker_path(self, path):
        return str(Path(path).resolve())

    def load(self, mcr, hashes=True):
        # The worker runs in its own scratch directory, so hand it an absolute path.
        return self.request("load", mcr=self._worker_path(mcr), hashes=hashes)

    def start(self, spec, hashes=True, reuse_map=False):
        """Enter any fight from a spec: character, ascension, seed, encounter and a partial save player (see
        ``spec_from_run`` and CombatWorker/CombatSpec.cs). Returns the first decision boundary.

        For training: ``hashes=False`` leaves ``state_hash`` and ``checkpoints`` out of every reply, and
        ``reuse_map=True`` generates the act's map once per spec instead of once per seed. Neither changes the fight."""
        return self.request("start", spec=spec, hashes=hashes, reuse_map=reuse_map)

    def catalog(self):
        return self.request("catalog")

    def start_run(self, character, seed, ascension=0, unlocks="all"):
        """Start one continuous native run. ``unlocks`` is ``all`` or fresh-profile ``none``."""
        spec = {"character": character, "seed": seed, "ascension": ascension, "unlocks": unlocks}
        reply = self.request("start_run", spec=spec)
        if self._run_trace:
            self._run_trace.close()
        self.run_trace_path = self.workdir / f"run-{reply['run_identity']}.jsonl"
        self._run_trace = open(self.run_trace_path, "w")
        self._record_run({"type": "start_run", **spec}, reply)
        return reply

    def run_step(self, action):
        try:
            reply = self.request("run_step", action=action)
        except WorkerError as error:
            self._record_run(action, {"ok": False, "error": str(error)})
            raise
        self._record_run(action, reply)
        return reply

    def _record_run(self, action, reply):
        if self._run_trace:
            self._run_trace.write(json.dumps({"action": action, "state": reply}, separators=(",", ":")) + "\n")
            self._run_trace.flush()

    def run_combat_snapshot(self, path):
        """Write the game's own recording of the current run combat as an .mcr. Its initial state is the run as it
        entered this room, so another worker's ``load`` re-enters the same fight while this run carries on."""
        return self.request("run_combat_snapshot", path=self._worker_path(path))

    def run_observe(self):
        return self.request("run_observe")

    def step(self, action):
        return self.request("step", action=action)

    def observe(self):
        return self.request("observe")

    def tape(self, mcr):
        return self.request("tape", mcr=self._worker_path(mcr))

    def close(self):
        if self._quit_command and self._quit_if_running():
            self._stdin.write(self._quit_command)
        self._stdin.close()
        self._wait_for_exit()
        self._stdout.close()
        self._close_transport()
        if self._run_trace:
            self._run_trace.close()
        if self._log is not None:
            self._log.close()


class CombatWorkerContainer(CombatWorker):
    def __init__(self, address, *, workdir=None):
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="combat-worker-container-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._socket = socket.create_connection(address)
        self._stdin = self._socket.makefile("w", encoding="utf-8", buffering=1)
        self._stdout = self._socket.makefile("r", encoding="utf-8", buffering=1)
        self._quit_command = None
        self._quit_if_running = lambda: False
        self._wait_for_exit = lambda: None
        self._close_transport = self._socket.close
        self._read_eof_error = lambda: "container worker disconnected; inspect the container logs"
        self._log = None
        self.boot_ms = self._read()["boot_ms"]
        self.run_trace_path = None
        self._run_trace = None

    def _worker_path(self, path):
        # Container paths come from runtime mounts and are already in the worker's namespace.
        return str(path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def card_name(card):
    """A card record from ``obs`` as the game id with one + per upgrade, e.g. CARD.HOLOGRAM+."""
    return card["id"] + "+" * card["upgrades"]


def recorded_action(event, state):
    """Translate one recorded tape event into a caller action at ``state``, the boundary it was played from.

    Cards and targets are recorded by combat id; a caller names them by hand index and enemy slot, so the
    translation goes through the observation exactly as a policy's choice would.
    """
    obs = state["obs"]
    slot_of = {e["combat_id"]: e["slot"] for e in obs["enemies"]}
    kind = event["kind"]
    if kind == "play":
        hand = [c["combat_card"] for c in obs["hand"]].index(event["combat_card"])
        target = None if event["target_combat_id"] is None else slot_of[event["target_combat_id"]]
        return {"type": "play", "hand": hand, "target": target}
    if kind == "end_turn":
        if obs["turn"] != event["turn"]:
            raise ValueError(f"tape ends turn {event['turn']} but the worker is on turn {obs['turn']}")
        return {"type": "end_turn"}
    if kind == "potion":
        target = None if event["target_combat_id"] is None else slot_of[event["target_combat_id"]]
        return {"type": "potion", "slot": event["slot"], "target": target}
    if kind == "choice":
        options = [o["combat_card"] for o in state["choice"]["options"]]
        return {"type": "choose", "picks": [options.index(c) for c in event["combat_cards"]]}
    raise ValueError(f"no caller action for tape event {event}")


def spec_from_run(run, encounter, seed, player=0, **player_fields):
    """A spec for ``encounter`` with the deck, relics and potions a .run history file ends on.

    Works on the game's local ``saves/history/*.run`` and on Spire Codex export records alike: both store each
    player's deck, relics and potions as the game's save JSON, which is what a spec's ``player`` is. HP defaults
    to what the player entered the run's last room with (the ``current_hp`` and ``max_hp`` recorded after the
    room before it). Any other SerializablePlayer field can be given as a keyword, e.g. ``current_hp=40``.

    The deck is the deck at the end of the run. Nothing in a .run says what it was at an earlier floor: cards
    carry the floor they were added on, but removals, transforms and upgrades are not dated.
    """
    p = run["players"][player]
    history = [point for act in run["map_point_history"] for point in act]
    entering = next(s for s in history[-2]["player_stats"] if s["player_id"] == p["id"]) if len(history) > 1 else {}
    fields = {"deck": p["deck"], "relics": p["relics"], "potions": p["potions"],
              "max_potion_slot_count": p["max_potion_slot_count"]}
    fields.update({k: entering[k] for k in ("current_hp", "max_hp") if k in entering})
    fields.update(player_fields)
    return {"character": p["character"], "ascension": run["ascension"], "seed": seed, "encounter": encounter,
            "player": fields}
