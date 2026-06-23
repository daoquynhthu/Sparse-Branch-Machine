#!/usr/bin/env python3
"""Convert the existing NAIME FineWeb-Edu token blocks into SBM shards.

The source artifact contains fixed token blocks, not recoverable source-document
boundaries.  Every source row is therefore an independent sequence and the
generated manifest is deliberately marked compatibility_only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from array import array
from pathlib import Path
from typing import Iterable, Iterator, Sequence


MAGIC = b"SBMSHR01"
FORMAT_VERSION = 1
HEADER_SIZE = 128
ENDIAN_MARKER = 0x01020304
ALIGNMENT = 64
MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    value &= MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _rotl64(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & MASK64


def dataset_hash(vocab_size: int, tokens: Sequence[int], offsets: Sequence[int]) -> int:
    result = _mix64(vocab_size) ^ 0x544F4B454E
    for index, token in enumerate(tokens):
        result ^= _mix64((int(token) + index * 0x9E3779B97F4A7C15) & MASK64)
        result = _rotl64(result, 11)
    for index, offset in enumerate(offsets):
        result ^= _mix64((int(offset) + index * 0xC2B2AE3D27D4EB4F) & MASK64)
        result = _rotl64(result, 5)
    return _mix64(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_shard(path: Path, rows: Sequence[Sequence[int]], vocab_size: int) -> dict:
    if sys.byteorder != "little":
        raise RuntimeError("SBM shard conversion currently requires a little-endian host")
    if not rows or vocab_size <= 0:
        raise ValueError("a shard requires rows and a positive vocabulary")
    if path.exists():
        raise FileExistsError(path)

    tokens = array("I")
    offsets = array("Q", [0])
    for row in rows:
        if len(row) < 2:
            raise ValueError("every source row must contain at least two tokens")
        for token in row:
            value = int(token)
            if value < 0 or value >= vocab_size:
                raise ValueError(f"token ID {value} is outside vocabulary {vocab_size}")
            tokens.append(value)
        offsets.append(len(tokens))

    offsets_offset = HEADER_SIZE
    tokens_offset = (offsets_offset + len(offsets) * 8 + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
    file_size = tokens_offset + len(tokens) * 4
    content_hash = dataset_hash(vocab_size, tokens, offsets)
    header = bytearray(HEADER_SIZE)
    header[:8] = MAGIC
    struct.pack_into(
        "<IIIIQQQQQQ",
        header,
        8,
        FORMAT_VERSION,
        HEADER_SIZE,
        ENDIAN_MARKER,
        vocab_size,
        len(tokens),
        len(rows),
        offsets_offset,
        tokens_offset,
        content_hash,
        file_size,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as output:
            output.write(header)
            offsets.tofile(output)
            output.write(b"\0" * (tokens_offset - output.tell()))
            tokens.tofile(output)
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "path": path.name,
        "sha256": _sha256(path),
        "dataset_hash_u64": f"{content_hash:016x}",
        "tokens": len(tokens),
        "sequences": len(rows),
        "examples": len(tokens) - len(rows),
        "bytes": file_size,
    }


def _iter_rows(split) -> Iterator[Sequence[int]]:
    for batch in split.iter(batch_size=1024):
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("source split has no input_ids column")
        for row in input_ids:
            yield row


def convert_split(
    rows: Iterable[Sequence[int]],
    output_dir: Path,
    split_name: str,
    vocab_size: int,
    shard_tokens: int,
    max_tokens: int | None,
) -> list[dict]:
    shards: list[dict] = []
    pending: list[Sequence[int]] = []
    pending_tokens = 0
    accepted_tokens = 0

    def flush() -> None:
        nonlocal pending, pending_tokens
        if not pending:
            return
        path = output_dir / f"{split_name}-{len(shards):05d}.sbt"
        shards.append(write_shard(path, pending, vocab_size))
        pending = []
        pending_tokens = 0

    for row in rows:
        row_tokens = len(row)
        if max_tokens is not None and accepted_tokens + row_tokens > max_tokens:
            break
        if pending and pending_tokens + row_tokens > shard_tokens:
            flush()
        pending.append(row)
        pending_tokens += row_tokens
        accepted_tokens += row_tokens
    flush()
    return shards


def split_token_limit(args: argparse.Namespace, split_name: str) -> int | None:
    if split_name == "train" and args.max_train_tokens is not None:
        return args.max_train_tokens
    if split_name == "validation" and args.max_validation_tokens is not None:
        return args.max_validation_tokens
    return args.max_tokens_per_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--shard-tokens", type=int, default=16_000_000)
    parser.add_argument("--max-tokens-per-split", type=int)
    parser.add_argument("--max-train-tokens", type=int)
    parser.add_argument("--max-validation-tokens", type=int)
    parser.add_argument("--provenance-log", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")

    from datasets import load_from_disk

    dataset = load_from_disk(str(args.dataset))
    split_names = [name.strip() for name in args.splits.split(",") if name.strip()]
    manifest = {
        "schema": "sbm-corpus-manifest",
        "schema_version": 1,
        "preprocess_version": 1,
        "status": "compatibility_only",
        "source": "HuggingFaceFW/fineweb-edu",
        "source_config": "sample-10BT",
        "source_revision": None,
        "license": None,
        "source_artifact": args.dataset.name,
        "boundary_policy": "each_existing_fixed_token_block_is_an_independent_sequence",
        "split_policy": "preexisting_train_and_validation_splits_unverified",
        "tokenizer": {
            "identity": "gpt2_inferred_from_observed_vocabulary",
            "vocab_size": args.vocab_size,
            "artifact_sha256": None,
            "trained_on_train_only": None,
        },
        "limitations": [
            "original_document_ids_and_boundaries_are_not_present",
            "source_revision_and_license_are_not_present_in_the_artifact",
            "tokenizer_artifact_and_checksum_are_not_present",
            "cross_split_content_deduplication_cannot_be_verified",
            "fixed_blocks_may_split_original_documents",
        ],
        "source_splits": {},
        "splits": {},
    }
    if args.provenance_log:
        manifest["provenance_log"] = {
            "name": args.provenance_log.name,
            "sha256": _sha256(args.provenance_log),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    for name in split_names:
        if name not in dataset:
            raise ValueError(f"source dataset has no split {name!r}")
        split = dataset[name]
        manifest["source_splits"][name] = {
            "rows": len(split),
            "fingerprint": getattr(split, "_fingerprint", None),
        }
        shards = convert_split(
            _iter_rows(split), args.output, name, args.vocab_size,
            args.shard_tokens, split_token_limit(args, name),
        )
        manifest["splits"][name] = {
            "shards": shards,
            "tokens": sum(item["tokens"] for item in shards),
            "sequences": sum(item["sequences"] for item in shards),
            "examples": sum(item["examples"] for item in shards),
        }

    manifest_path = args.output / "manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "splits": {name: manifest["splits"][name]["tokens"] for name in split_names},
        "status": manifest["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
