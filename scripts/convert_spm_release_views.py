#!/usr/bin/env python3
"""Convert SPM data-pipeline tokenized Parquet views into SBM token shards."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow.parquet as pq


NAIME_CONVERTER = Path(__file__).resolve().parent / "convert_naime_dataset.py"
SPEC = importlib.util.spec_from_file_location("convert_naime_dataset", NAIME_CONVERTER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load convert_naime_dataset.py")
SHARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARD)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_token_rows(path: Path) -> Iterator[Sequence[int]]:
    parquet = pq.ParquetFile(path)
    try:
        for batch in parquet.iter_batches(columns=["token_ids"]):
            for row in batch.column(0).to_pylist():
                if len(row) >= 2:
                    yield row
    finally:
        parquet.close()


def convert_view(
    view_path: Path,
    output: Path,
    split_name: str,
    vocab_size: int,
    shard_tokens: int,
    max_tokens: int | None,
) -> list[dict]:
    return SHARD.convert_split(
        iter_token_rows(view_path),
        output,
        split_name,
        vocab_size,
        shard_tokens,
        max_tokens,
    )


def _view_stats(path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    try:
        rows = int(parquet.metadata.num_rows)
    finally:
        parquet.close()
    table = pq.read_table(path, columns=["token_count"])
    tokens = sum(int(value) for value in table.column("token_count").to_pylist())
    return {"rows": rows, "tokens": tokens, "sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-view", default="train-10000000.parquet")
    parser.add_argument("--validation-view", default="validation.parquet")
    parser.add_argument("--test-view", default="test.parquet")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--shard-tokens", type=int, default=4_000_000)
    parser.add_argument("--max-train-tokens", type=int)
    parser.add_argument("--max-validation-tokens", type=int)
    parser.add_argument("--max-test-tokens", type=int)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer_json = args.tokenizer / "tokenizer.json"
    tokenizer_manifest = args.tokenizer / "tokenizer-manifest.json"
    views_manifest = args.views / "views-manifest.json"
    for path in [tokenizer_json, tokenizer_manifest, views_manifest, args.validation_report]:
        if not path.is_file():
            raise FileNotFoundError(path)

    split_specs = {
        "train": (args.views / args.train_view, args.max_train_tokens),
        "validation": (args.views / args.validation_view, args.max_validation_tokens),
        "test": (args.views / args.test_view, args.max_test_tokens),
    }
    manifest = {
        "schema": "sbm-corpus-manifest",
        "schema_version": 1,
        "preprocess_version": 1,
        "status": "admissible_document_level",
        "source": "HuggingFaceFW/fineweb-edu",
        "source_config": "sample-10BT",
        "source_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "license": "odc-by-1.0",
        "boundary_policy": "one parquet row is one source document with explicit bos/eos tokens",
        "split_policy": "document_level_hash_split_from_spm_data_pipeline",
        "tokenizer": {
            "identity": "spm-data-pipeline-byte-bpe-16k",
            "vocab_size": args.vocab_size,
            "artifact_sha256": _sha256(tokenizer_json),
            "manifest_sha256": _sha256(tokenizer_manifest),
            "trained_on_train_only": True,
            "manifest": _load_json(tokenizer_manifest),
        },
        "upstream_validation": {
            "report_sha256": _sha256(args.validation_report),
            "report": _load_json(args.validation_report),
        },
        "source_views_manifest": {
            "sha256": _sha256(views_manifest),
            "manifest": _load_json(views_manifest),
        },
        "source_splits": {},
        "splits": {},
    }
    for split_name, (view_path, max_tokens) in split_specs.items():
        if not view_path.is_file():
            raise FileNotFoundError(view_path)
        manifest["source_splits"][split_name] = {
            "view": view_path.name,
            **_view_stats(view_path),
            "max_tokens": max_tokens,
        }
        shards = convert_view(
            view_path,
            args.output,
            split_name,
            args.vocab_size,
            args.shard_tokens,
            max_tokens,
        )
        manifest["splits"][split_name] = {
            "shards": shards,
            "tokens": sum(item["tokens"] for item in shards),
            "sequences": sum(item["sequences"] for item in shards),
            "examples": sum(item["examples"] for item in shards),
        }
    manifest_path = args.output / "manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "splits": {
                    name: manifest["splits"][name]["tokens"] for name in manifest["splits"]
                },
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
