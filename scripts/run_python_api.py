#!/usr/bin/env python3
"""Minimal in-process API example for mathematical token cross-entropy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    parser.add_argument("--task", choices=["token-ce", "vector"], default="token-ce")
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", help="JSON object containing runtime parameters")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    overrides = {"seed": args.seed}
    if args.config:
        overrides.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for assignment in args.set:
        name, value = assignment.split("=", 1)
        overrides[name] = value

    runtime = Runtime(args.library)
    if args.task == "token-ce":
        dataset_context = runtime.generate_math_token_dataset(
            sequence_count=args.sequences,
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            seed=args.seed,
            temperature=0.8,
            interaction_strength=0.2,
        )
    else:
        dataset_context = runtime.generate_dataset(length=args.length, seed=args.seed)

    with dataset_context as dataset:
        with runtime.config(overrides) as config:
            result = dataset.run(config, warmup=args.warmup)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
