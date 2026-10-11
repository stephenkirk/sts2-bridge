"""Extend a checkpoint's vocabulary for another pool once, so several arms start from the same weights.

    .venv/bin/python -m agents.nn.branch SOURCE.pt --pool insatiable --out INIT.pt

New IDs start as copies of the unknown row (see train.transfer_weights). Train each arm with
--init-from INIT.pt and without --extend-vocab.
"""

import argparse
import hashlib
from pathlib import Path

import torch

from . import fights as pools
from .model import Net
from .train import transfer_weights, warm_count, warm_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--pool", choices=["starter", "insatiable", "library"], required=True)
    ap.add_argument("--pool-file")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.pool == "library" and not args.pool_file:
        ap.error("--pool library requires --pool-file")
    if args.out.exists():
        ap.error(f"{args.out} exists")
    source = torch.load(args.source, weights_only=False, map_location="cpu")
    fights = pools.pool(args.pool, pool_file=args.pool_file)
    metadata = {}
    if args.pool == "library":
        metadata = dict(pool_file=str(Path(args.pool_file).resolve()),
                        pool_digest=hashlib.sha256(Path(args.pool_file).read_bytes()).hexdigest())
    vocab = warm_vocab(fights, n=warm_count(args.pool, fights), table=source["vocab"])
    cfg = dict(source["cfg"], vocab_size=max(source["cfg"]["vocab_size"], len(vocab) + 8))
    net = Net(**cfg)
    transfer_weights(net, source["state"], inherited=len(source["vocab"]) + 2)
    added = sorted(set(vocab.table) - set(source["vocab"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cfg=cfg, vocab=vocab.table, pool=args.pool, state=net.state_dict(), it=source.get("it"),
                    episodes=source.get("episodes"), **metadata,
                    branched_from=dict(checkpoint=str(args.source.resolve()), pool=source.get("pool"),
                                       sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), added=added)),
               args.out)
    print(f"{args.out}: {len(source['vocab'])} -> {len(vocab.table)} IDs; added {added}")


if __name__ == "__main__":
    main()
