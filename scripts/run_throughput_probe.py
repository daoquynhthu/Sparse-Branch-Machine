#!/usr/bin/env python3
"""Find mapped-corpus throughput once short C++ windows have stabilized."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stable(speeds: list[float], window_count: int, tolerance: float) -> bool:
    if len(speeds) < window_count:
        return False
    tail = speeds[-window_count:]
    mean = statistics.mean(tail)
    if mean <= 0.0:
        return False
    return (max(tail) - min(tail)) / mean <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--bucket-bits", type=int, default=14)
    parser.add_argument(
        "--train-examples",
        type=int,
        default=300000,
        help="Maximum train examples to probe before giving up on stability.",
    )
    parser.add_argument("--eval-examples", type=int, default=10000)
    parser.add_argument("--start-examples", type=int, default=50000)
    parser.add_argument("--step-examples", type=int, default=50000)
    parser.add_argument("--stable-windows", type=int, default=3)
    parser.add_argument("--stability-tolerance", type=float, default=0.08)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each window size; keep at 1 for quick stability probes.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adaptive-topology", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    if args.train_examples < args.start_examples:
        parser.error("--train-examples must be >= --start-examples")
    if args.start_examples < 1000 or args.eval_examples < 100:
        parser.error("probe windows are too small to be meaningful")
    if args.step_examples < 1:
        parser.error("--step-examples must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.stable_windows < 2:
        parser.error("--stable-windows must be at least 2")
    if args.stability_tolerance <= 0.0:
        parser.error("--stability-tolerance must be positive")

    runs = []
    window_speeds: list[float] = []
    stopped_reason = "max_examples"
    with tempfile.TemporaryDirectory(prefix="sbm-throughput-") as directory_text:
        directory = Path(directory_text)
        train_examples = args.start_examples
        while train_examples <= args.train_examples:
            repeat_speeds = []
            repeat_runs = []
            for repeat_index in range(args.repeat):
                output = directory / f"probe-{train_examples}-{repeat_index}.json"
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_corpus_training.py"),
                    "--manifest",
                    str(args.manifest.resolve()),
                    "--seed",
                    str(args.seed + repeat_index),
                    "--output",
                    str(output),
                    "--bucket-bits",
                    str(args.bucket_bits),
                    "--max-train-examples",
                    str(train_examples),
                    "--max-eval-examples",
                    str(args.eval_examples),
                ]
                if args.library:
                    command.extend(["--library", str(Path(args.library).resolve())])
                if args.adaptive_topology:
                    command.append("--adaptive-topology")
                for override in args.overrides:
                    if "=" not in override:
                        parser.error(f"--set requires NAME=VALUE, got {override!r}")
                    command.extend(["--set", override])
                subprocess.run(command, cwd=ROOT, check=True)
                payload = json.loads(output.read_text(encoding="utf-8"))
                result = payload["result"]
                repeat_speeds.append(float(result["steps_per_second"]))
                repeat_runs.append({
                    "seed": payload["seed"],
                    "train_examples": result["train_examples"],
                    "eval_examples": result["eval_examples"],
                    "steps_per_second": result["steps_per_second"],
                    "live_nodes": result["live_nodes"],
                    "estimated_bytes": result["estimated_bytes"],
                    "eval_cross_entropy": result["eval_cross_entropy"],
                })
            window_speed = statistics.mean(repeat_speeds)
            window_speeds.append(window_speed)
            is_stable = stable(
                window_speeds,
                args.stable_windows,
                args.stability_tolerance,
            )
            runs.append({
                "train_examples": train_examples,
                "mean_steps_per_second": window_speed,
                "runs": repeat_runs,
            })
            print(
                json.dumps(
                    {
                        "train_examples": train_examples,
                        "mean_steps_per_second": window_speed,
                        "stable": is_stable,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            if is_stable:
                stopped_reason = "stable"
                break
            train_examples += args.step_examples

    summary = {
        "manifest": str(args.manifest.resolve()),
        "max_train_examples": args.train_examples,
        "eval_examples": args.eval_examples,
        "start_examples": args.start_examples,
        "step_examples": args.step_examples,
        "stable_windows": args.stable_windows,
        "stability_tolerance": args.stability_tolerance,
        "repeat": args.repeat,
        "stopped_reason": stopped_reason,
        "final_train_examples": runs[-1]["train_examples"],
        "mean_steps_per_second": runs[-1]["mean_steps_per_second"],
        "min_window_steps_per_second": min(window_speeds),
        "max_window_steps_per_second": max(window_speeds),
        "runs": runs,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
