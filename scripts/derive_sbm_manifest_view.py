#!/usr/bin/env python3
"""Derive a smaller whole-shard SBM corpus manifest view without split reassignment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def select_shards(shards: list[dict[str, Any]], target_examples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total = 0
    for shard in shards:
        selected.append(dict(shard))
        total += int(shard.get("examples", 0))
        if total >= target_examples:
            break
    if total < target_examples:
        raise ValueError(f"split has {total} examples, requested {target_examples}")
    return selected


def relpath_for_output(source_manifest: Path, output_manifest: Path, shard_path: str) -> str:
    absolute = (source_manifest.parent / shard_path).resolve()
    relative = os.path.relpath(absolute, output_manifest.parent.resolve())
    return relative.replace(os.sep, "/")


def derive(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema") != "sbm-corpus-manifest":
        raise ValueError("unsupported source manifest schema")
    budgets = {
        "train": args.train_examples,
        "validation": args.validation_examples,
        "test": args.test_examples,
    }
    splits: dict[str, Any] = {}
    for name, budget in budgets.items():
        split = source.get("splits", {}).get(name)
        if not split or not split.get("shards"):
            raise ValueError(f"source manifest has no {name!r} split")
        shards = select_shards(split["shards"], budget)
        rewritten = []
        for shard in shards:
            item = dict(shard)
            item["path"] = relpath_for_output(source_path, output_path, shard["path"])
            rewritten.append(item)
        splits[name] = {"shards": rewritten}
    return {
        "schema": "sbm-corpus-manifest",
        "status": source.get("status"),
        "view_name": args.name,
        "source_manifest": str(source_path),
        "selection_policy": "whole-shard-prefix-by-example-budget",
        "requested_examples": budgets,
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--train-examples", type=int, required=True)
    parser.add_argument("--validation-examples", type=int, required=True)
    parser.add_argument("--test-examples", type=int, required=True)
    args = parser.parse_args()
    payload = derive(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "view_name": args.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
