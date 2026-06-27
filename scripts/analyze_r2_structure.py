#!/usr/bin/env python3
"""R2 structural diagnosis for SBM token shards.

The report intentionally analyzes aggregate codelength behavior before any
example inspection.  It does not replay the trained model; it measures whether
the fixed address structures used by the completed R1 run have held-out
statistical value by document position, target frequency and address reuse.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAGIC = b"SBMSHR01"
HEADER_SIZE = 128
ALIGNMENT = 64
LAGS = (1, 2, 4)
EPS = 1.0e-12


@dataclass(frozen=True)
class ShardView:
    path: Path
    vocab_size: int
    tokens: memoryview
    offsets: memoryview
    mapping: mmap.mmap
    handle: object

    def close(self) -> None:
        self.tokens.release()
        self.offsets.release()
        self.mapping.close()
        self.handle.close()


class ConditionalCounts:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.totals: Counter[int] = Counter()
        self.joint: Counter[int] = Counter()

    def update(self, context_key: int, target: int) -> None:
        self.totals[context_key] += 1
        self.joint[context_key * self.vocab_size + target] += 1

    def probability(self, context_key: int, target: int, fallback: float) -> float:
        total = self.totals.get(context_key, 0)
        count = self.joint.get(context_key * self.vocab_size + target, 0)
        # One effective pseudocount, distributed according to the fallback.
        return (count + fallback) / (total + 1.0)


class Accumulator:
    def __init__(self) -> None:
        self.count = 0
        self.nll = defaultdict(float)

    def add(self, values: dict[str, float]) -> None:
        self.count += 1
        for key, value in values.items():
            self.nll[key] += value

    def mean(self) -> dict[str, float | int]:
        output: dict[str, float | int] = {"examples": self.count}
        if self.count == 0:
            return output
        for key, total in sorted(self.nll.items()):
            output[f"{key}_nll"] = total / self.count
        if "current" in self.nll and "interp_lags_1_2_4" in self.nll:
            output["gain_current_minus_interp"] = (
                self.nll["current"] - self.nll["interp_lags_1_2_4"]
            ) / self.count
        for lag in LAGS:
            name = f"lag{lag}"
            if "current" in self.nll and name in self.nll:
                output[f"gain_current_minus_{name}"] = (
                    self.nll["current"] - self.nll[name]
                ) / self.count
        return output


def _read_u64(buffer: memoryview, index: int) -> int:
    return struct.unpack_from("<Q", buffer, index * 8)[0]


def open_shard(path: Path) -> ShardView:
    handle = path.open("rb")
    mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    header = memoryview(mapping)[:HEADER_SIZE]
    if bytes(header[:8]) != MAGIC:
        raise ValueError(f"{path} is not an SBM token shard")
    version, header_size, endian, vocab_size = struct.unpack_from("<IIII", header, 8)
    token_count, sequence_count, offsets_offset, tokens_offset, _, file_size = struct.unpack_from(
        "<QQQQQQ", header, 24
    )
    if version != 1 or header_size != HEADER_SIZE or endian != 0x01020304:
        raise ValueError(f"{path} has an unsupported shard header")
    if tokens_offset % ALIGNMENT != 0 or file_size != mapping.size():
        raise ValueError(f"{path} has an invalid shard layout")
    offsets = memoryview(mapping)[offsets_offset : offsets_offset + (sequence_count + 1) * 8]
    tokens = memoryview(mapping)[tokens_offset : tokens_offset + token_count * 4]
    return ShardView(path, vocab_size, tokens.cast("I"), offsets, mapping, handle)


def split_paths(manifest: dict, manifest_path: Path, split: str) -> list[Path]:
    base = manifest_path.parent
    entries = manifest["splits"][split]["shards"]
    return [base / entry["path"] for entry in entries]


def manifest_vocab_size(manifest: dict, first_shard: Path) -> int:
    tokenizer = manifest.get("tokenizer")
    if isinstance(tokenizer, dict) and "vocab_size" in tokenizer:
        return int(tokenizer["vocab_size"])
    if "vocab_size" in manifest:
        return int(manifest["vocab_size"])
    shard = open_shard(first_shard)
    try:
        return shard.vocab_size
    finally:
        shard.close()


def iter_sequences(paths: Iterable[Path]):
    for path in paths:
        shard = open_shard(path)
        try:
            sequence_count = len(shard.offsets) // 8 - 1
            for index in range(sequence_count):
                start = _read_u64(shard.offsets, index)
                end = _read_u64(shard.offsets, index + 1)
                yield shard.tokens[start:end].tolist()
        finally:
            shard.close()


def position_bin(position: int) -> str:
    if position == 0:
        return "0"
    if position <= 3:
        return "1-3"
    if position <= 15:
        return "4-15"
    if position <= 63:
        return "16-63"
    if position <= 255:
        return "64-255"
    return "256+"


def count_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 15:
        return "5-15"
    if count <= 63:
        return "16-63"
    if count <= 255:
        return "64-255"
    return "256+"


def nll(probability: float) -> float:
    return -math.log(max(probability, EPS))


def train_counts(train_paths: list[Path], vocab_size: int, max_examples: int | None) -> dict:
    target_counts = [0] * vocab_size
    current_counts = ConditionalCounts(vocab_size)
    lag_counts = {lag: ConditionalCounts(vocab_size) for lag in LAGS}
    total = 0
    stop = False
    for sequence in iter_sequences(train_paths):
        for pos in range(len(sequence) - 1):
            current = int(sequence[pos])
            target = int(sequence[pos + 1])
            target_counts[target] += 1
            current_counts.update(current, target)
            for lag, table in lag_counts.items():
                if pos >= lag:
                    context = int(sequence[pos - lag]) * vocab_size + current
                    table.update(context, target)
            total += 1
            if max_examples is not None and total >= max_examples:
                stop = True
                break
        if stop:
            break
    return {
        "examples": total,
        "target_counts": target_counts,
        "current": current_counts,
        "lags": lag_counts,
    }


def unigram_probability(target_counts: list[int], total: int, vocab_size: int, target: int) -> float:
    return (target_counts[target] + 0.5) / (total + 0.5 * vocab_size)


def evaluate(
    validation_paths: list[Path],
    counts: dict,
    vocab_size: int,
    max_examples: int | None,
) -> dict:
    overall = Accumulator()
    by_position: dict[str, Accumulator] = defaultdict(Accumulator)
    by_target_frequency: dict[str, Accumulator] = defaultdict(Accumulator)
    by_address_reuse: dict[str, dict[str, Accumulator]] = {
        f"lag{lag}": defaultdict(Accumulator) for lag in LAGS
    }
    doc_gains: list[float] = []
    doc_positive = 0
    evaluated = 0

    target_counts = counts["target_counts"]
    train_examples = counts["examples"]
    current_counts: ConditionalCounts = counts["current"]
    lag_counts: dict[int, ConditionalCounts] = counts["lags"]
    stop = False

    for sequence in iter_sequences(validation_paths):
        doc_current = 0.0
        doc_interp = 0.0
        doc_examples = 0
        for pos in range(len(sequence) - 1):
            current = int(sequence[pos])
            target = int(sequence[pos + 1])
            p_unigram = unigram_probability(target_counts, train_examples, vocab_size, target)
            p_current = current_counts.probability(current, target, p_unigram)
            lag_probabilities: dict[int, float] = {}
            lag_reuse: dict[int, int] = {}
            for lag, table in lag_counts.items():
                if pos >= lag:
                    context = int(sequence[pos - lag]) * vocab_size + current
                    lag_probabilities[lag] = table.probability(context, target, p_current)
                    lag_reuse[lag] = table.totals.get(context, 0)
                else:
                    lag_probabilities[lag] = p_current
                    lag_reuse[lag] = 0
            p_interp = sum(lag_probabilities.values()) / len(lag_probabilities)
            values = {
                "unigram": nll(p_unigram),
                "current": nll(p_current),
                "interp_lags_1_2_4": nll(p_interp),
            }
            for lag, probability in lag_probabilities.items():
                values[f"lag{lag}"] = nll(probability)

            overall.add(values)
            by_position[position_bin(pos)].add(values)
            by_target_frequency[count_bin(target_counts[target])].add(values)
            for lag, reuse in lag_reuse.items():
                by_address_reuse[f"lag{lag}"][count_bin(reuse)].add(values)

            doc_current += values["current"]
            doc_interp += values["interp_lags_1_2_4"]
            doc_examples += 1
            evaluated += 1
            if max_examples is not None and evaluated >= max_examples:
                stop = True
                break
        if doc_examples > 0:
            gain = (doc_current - doc_interp) / doc_examples
            doc_gains.append(gain)
            if gain > 0.0:
                doc_positive += 1
        if stop:
            break

    doc_gains_sorted = sorted(doc_gains)
    quantiles = {}
    if doc_gains_sorted:
        for name, q in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            idx = min(len(doc_gains_sorted) - 1, max(0, round(q * (len(doc_gains_sorted) - 1))))
            quantiles[name] = doc_gains_sorted[idx]
    return {
        "overall": overall.mean(),
        "by_document_position": {key: acc.mean() for key, acc in sorted(by_position.items())},
        "by_target_train_frequency": {
            key: acc.mean() for key, acc in sorted(by_target_frequency.items())
        },
        "by_address_reuse": {
            lag: {key: acc.mean() for key, acc in sorted(values.items())}
            for lag, values in sorted(by_address_reuse.items())
        },
        "document_gain_current_minus_interp": {
            "documents": len(doc_gains),
            "positive_documents": doc_positive,
            "positive_fraction": doc_positive / len(doc_gains) if doc_gains else 0.0,
            "mean": sum(doc_gains) / len(doc_gains) if doc_gains else 0.0,
            **quantiles,
        },
    }


def load_result_summary(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("result", payload)


def format_table(rows: dict[str, dict], columns: list[str]) -> str:
    lines = ["| bucket | " + " | ".join(columns) + " |"]
    lines.append("|---|" + "|".join("---" for _ in columns) + "|")
    for bucket, values in rows.items():
        rendered = []
        for column in columns:
            value = values.get(column)
            if isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value) if value is not None else "")
        lines.append(f"| {bucket} | " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_markdown(report: dict, path: Path) -> None:
    overall = report["evaluation"]["overall"]
    doc = report["evaluation"]["document_gain_current_minus_interp"]
    lines = [
        "# R2 Structural Diagnosis",
        "",
        "This report is aggregate-only. It analyzes fixed lag structures from the R1 run; "
        "it does not claim the adaptive accepted-structure hard gate is satisfied.",
        "",
        "## Overall",
        "",
        format_table(
            {"validation": overall},
            [
                "examples",
                "unigram_nll",
                "current_nll",
                "lag1_nll",
                "lag2_nll",
                "lag4_nll",
                "interp_lags_1_2_4_nll",
                "gain_current_minus_interp",
            ],
        ),
        "",
        "## Cross-document gain",
        "",
        format_table({"documents": doc}, ["documents", "positive_documents", "positive_fraction", "mean", "p10", "p50", "p90"]),
        "",
        "## By document position",
        "",
        format_table(
            report["evaluation"]["by_document_position"],
            ["examples", "current_nll", "interp_lags_1_2_4_nll", "gain_current_minus_interp"],
        ),
        "",
        "## By target train frequency",
        "",
        format_table(
            report["evaluation"]["by_target_train_frequency"],
            ["examples", "current_nll", "interp_lags_1_2_4_nll", "gain_current_minus_interp"],
        ),
        "",
        "## By address reuse",
        "",
    ]
    for lag, rows in report["evaluation"]["by_address_reuse"].items():
        lines.extend(
            [
                f"### {lag}",
                "",
                format_table(
                    rows,
                    [
                        "examples",
                        "current_nll",
                        f"{lag}_nll",
                        f"gain_current_minus_{lag}",
                        "gain_current_minus_interp",
                    ],
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    assert position_bin(0) == "0"
    assert position_bin(4) == "4-15"
    assert count_bin(0) == "0"
    assert count_bin(64) == "64-255"
    counts = ConditionalCounts(8)
    counts.update(3, 4)
    assert counts.probability(3, 4, 0.1) > counts.probability(3, 5, 0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.manifest is None or args.output_json is None or args.output_md is None:
        parser.error("--manifest, --output-json and --output-md are required unless --self-test is used")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train = split_paths(manifest, args.manifest, "train")
    validation = split_paths(manifest, args.manifest, "validation")
    vocab_size = manifest_vocab_size(manifest, train[0])

    print("[r2] counting train conditionals", file=sys.stderr)
    counts = train_counts(train, vocab_size, args.max_train_examples)
    print("[r2] evaluating validation structure bins", file=sys.stderr)
    evaluation = evaluate(validation, counts, vocab_size, args.max_eval_examples)
    result_summary = load_result_summary(args.result)

    report = {
        "schema": "spm.r2_structure_diagnosis.v1",
        "manifest": str(args.manifest),
        "result": str(args.result) if args.result else None,
        "limits": {
            "max_train_examples": args.max_train_examples,
            "max_eval_examples": args.max_eval_examples,
        },
        "limitations": [
            "This is an offline structural-control analysis, not model replay.",
            "The R1 run used fixed lag programs [1], [2] and [4]; accepted adaptive topology reuse is therefore not established by this report.",
            "Document boundaries are shard row boundaries inherited from the converted corpus artifact.",
        ],
        "r1_result_summary": result_summary,
        "train_examples_counted": counts["examples"],
        "evaluation": evaluation,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_md)
    print(f"[r2] wrote {args.output_json}", file=sys.stderr)
    print(f"[r2] wrote {args.output_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
