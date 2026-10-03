"""Start training and its dashboard together; Ctrl-C stops both and their workers."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlencode


def commands(args, training_args):
    watcher = [sys.executable, "-m", "agents.nn.watch", "--port", str(args.watch_port)]
    if args.watch_rematch:
        watcher += ["--rematch", args.watch_rematch]
    trainer = [sys.executable, "-m", "agents.nn.train", "--name", args.name, *training_args]
    return watcher, trainer


def stop(process):
    # The learner's multiprocessing actors/evaluator share its process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog="Other arguments are passed to agents.nn.train.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--watch-port", type=int, default=8765)
    ap.add_argument("--watch-rematch", help="optional fixed-fight rematch JSON")
    args, training_args = ap.parse_known_args()
    # Fail before starting anything; don't replace an existing watcher.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.watch_port))
        except OSError as exc:
            ap.error(f"dashboard port {args.watch_port} unavailable ({exc}); choose another --watch-port")
    watcher_cmd, trainer_cmd = commands(args, training_args)
    processes = []
    try:
        watcher = subprocess.Popen(watcher_cmd, start_new_session=True)
        processes.append(watcher)
        trainer = subprocess.Popen(trainer_cmd, start_new_session=True)
        processes.append(trainer)
        print(f"Dashboard: http://localhost:{args.watch_port}/?{urlencode({'run': args.name})}", flush=True)
        while trainer.poll() is None:
            if watcher.poll() is not None:
                raise RuntimeError("dashboard exited; stopping training")
            time.sleep(0.25)
        return trainer.returncode
    except KeyboardInterrupt:
        return 130
    finally:
        for process in reversed(processes):
            stop(process)


if __name__ == "__main__":
    sys.exit(main())
