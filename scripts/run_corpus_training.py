#!/usr/bin/env python3
"""Run one mapped-corpus training/evaluation job and write its full result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket-bits", type=int, default=14)
    parser.add_argument("--adaptive-topology", action="store_true")
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

    runtime = Runtime(args.library)
    config_values = {
        "seed": args.seed,
        "bucket_bits": args.bucket_bits,
        "sparse_token_output": True,
        "adaptive_topology": args.adaptive_topology,
    }
    for override in args.overrides:
        if "=" not in override:
            parser.error(f"--set requires NAME=VALUE, got {override!r}")
        name, value = override.split("=", 1)
        config_values[name] = value
    with runtime.open_token_corpus(args.manifest) as corpus:
        with runtime.config(config_values) as config:
            result = corpus.run(config, strict_freeze=True)
    payload = {
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
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
