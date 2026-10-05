"""Train two experiment arms at once with the comparison dashboard; Ctrl-C stops all three.

    .venv/bin/python -m agents.nn.pair --arm lib-win=insatiable-library-win \\
        --arm lib-damage=insatiable-library-damage --init-from INIT.pt --actors 3

Other arguments are passed to both trainers.
"""

import argparse
import socket
import subprocess
import sys
import time
from urllib.parse import urlencode

from .launch import stop


def commands(args, training_args):
    server = [sys.executable, "-m", "agents.nn.compare", "--port", str(args.port)]
    trainers = [[sys.executable, "-m", "agents.nn.train", "--name", name, "--experiment", experiment, *training_args]
                for name, experiment in args.arm]
    return server, trainers


def arm(value):
    name, sep, experiment = value.partition("=")
    if not sep or not name or not experiment:
        raise argparse.ArgumentTypeError("expected NAME=EXPERIMENT")
    return name, experiment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=arm, action="append", required=True, help="NAME=EXPERIMENT; give exactly two")
    ap.add_argument("--port", type=int, default=8772)
    args, training_args = ap.parse_known_args()
    if len(args.arm) != 2 or args.arm[0][0] == args.arm[1][0]:
        ap.error("give two --arm options with different names")
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as exc:
            ap.error(f"dashboard port {args.port} unavailable ({exc}); choose another --port")
    server_cmd, trainer_cmds = commands(args, training_args)
    processes = []
    try:
        server = subprocess.Popen(server_cmd, start_new_session=True)
        processes.append(server)
        trainers = [subprocess.Popen(cmd, start_new_session=True) for cmd in trainer_cmds]
        processes += trainers
        query = urlencode(dict(a=args.arm[0][0], b=args.arm[1][0]))
        print(f"Dashboard: http://localhost:{args.port}/?{query}", flush=True)
        while any(t.poll() is None for t in trainers):
            if server.poll() is not None:
                raise RuntimeError("dashboard exited; stopping training")
            if any(t.poll() for t in trainers):
                raise RuntimeError("an arm failed; stopping the other so neither runs unpaired")
            time.sleep(0.25)
        return max(t.returncode for t in trainers)
    except KeyboardInterrupt:
        return 130
    finally:
        for process in reversed(processes):
            stop(process)


if __name__ == "__main__":
    sys.exit(main())
