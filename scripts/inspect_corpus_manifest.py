#!/usr/bin/env python3
"""Inspect an SBM corpus manifest and print compact split totals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def split_totals(split: dict[str, Any]) -> dict[str, int]:
    shards = split.get("shards", [])
    return {
        "shards": len(shards),
        "tokens": sum(int(shard.get("tokens", 0)) for shard in shards),
        "examples": sum(int(shard.get("examples", 0)) for shard in shards),
        "sequences": sum(int(shard.get("sequences", 0)) for shard in shards),
    }


def inspect_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sbm-corpus-manifest":
        raise ValueError("unsupported manifest schema")
    splits = {
        name: split_totals(split)
        for name, split in sorted(manifest.get("splits", {}).items())
    }
    return {
        "path": str(path.resolve()),
        "schema": manifest.get("schema"),
        "status": manifest.get("status"),
        "splits": splits,
        "total_examples": sum(item["examples"] for item in splits.values()),
        "total_tokens": sum(item["tokens"] for item in splits.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_manifest(args.manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
