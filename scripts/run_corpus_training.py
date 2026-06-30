#!/usr/bin/env python3
"""Run one mapped-corpus training/evaluation job and write its full result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_presets import apply_preset  # noqa: E402
from sbm_runtime import Runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--bucket-bits", type=int, default=14)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--adaptive-topology", action="store_true")
    parser.add_argument(
        "--preset",
        help="Named config preset from python/sbm_presets.py",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional runtime parameter override.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.adaptive_topology and args.preset:
        parser.error("--adaptive-topology is already part of presets; use --preset only")

    runtime = Runtime(args.library)
    config_values: dict[str, Any] = {
        "seed": args.seed,
        "bucket_bits": args.bucket_bits,
        "sparse_token_output": True,
    }
    if args.preset:
        config_values.update(apply_preset(args.preset))
    if args.adaptive_topology:
        config_values["adaptive_topology"] = True
    for override in args.overrides:
        if "=" not in override:
            parser.error(f"--set requires NAME=VALUE, got {override!r}")
        name, value = override.split("=", 1)
        config_values[name] = value
    with runtime.open_token_corpus(args.manifest, eval_split=args.eval_split) as corpus:
        with runtime.config(config_values) as config:
            result = corpus.run(
                config,
                strict_freeze=True,
                max_train_examples=args.max_train_examples,
                max_eval_examples=args.max_eval_examples,
            )
    payload = {
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "eval_split": args.eval_split,
        "config": config_values,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
