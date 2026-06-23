#!/usr/bin/env python3
"""Schedule reproducible mapped-corpus runs under live CPU and memory limits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_hardware import ResourcePolicy  # noqa: E402
from sbm_scheduler import ResourceScheduler, RunEstimate, ScheduledRun  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    seeds = [int(value) for value in text.split(",") if value.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("7,11,19"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-fraction", type=float, default=0.9)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--minimum-free-memory-mib", type=int, default=1024)
    parser.add_argument("--worker-memory-mib", type=int, default=768)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--bucket-bits", type=int, default=14)
    parser.add_argument("--adaptive-topology", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    if args.worker_memory_mib < 1:
        parser.error("worker memory must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = ResourcePolicy(
        args.cpu_fraction,
        args.memory_fraction,
        args.minimum_free_memory_mib << 20,
        args.max_workers,
    )
    jobs: list[ScheduledRun] = []
    outputs: dict[str, Path] = {}
    for seed in args.seeds:
        run_id = f"corpus-seed-{seed}"
        output = args.output_dir / f"seed-{seed}.json"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_corpus_training.py"),
            "--manifest",
            str(args.manifest.resolve()),
            "--seed",
            str(seed),
            "--output",
            str(output.resolve()),
            "--bucket-bits",
            str(args.bucket_bits),
        ]
        if args.library:
            command.extend(["--library", str(Path(args.library).resolve())])
        if args.adaptive_topology:
            command.append("--adaptive-topology")
        jobs.append(ScheduledRun(run_id, RunEstimate(1, args.worker_memory_mib << 20), command))
        outputs[run_id] = output

    returncodes = ResourceScheduler(policy=policy).run(jobs)
    completed = []
    for job in jobs:
        output = outputs[job.run_id]
        if returncodes.get(job.run_id) != 0 or not output.exists():
            raise RuntimeError(f"training job failed: {job.run_id}")
        completed.append(json.loads(output.read_text(encoding="utf-8")))
    summary = {
        "manifest": str(args.manifest.resolve()),
        "seeds": args.seeds,
        "runs": [
            {
                "seed": item["seed"],
                "eval_cross_entropy": item["result"]["eval_cross_entropy"],
                "unigram_cross_entropy": item["result"]["unigram_baseline_eval_cross_entropy"],
                "steps_per_second": item["result"]["steps_per_second"],
                "estimated_bytes": item["result"]["estimated_bytes"],
            }
            for item in completed
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
