#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_tune(library: str, directory: Path, mode: list[str], name: str) -> dict:
    output = directory / f"{name}.json"
    best = directory / f"{name}-best.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "tune.py"),
        "--library",
        library,
        "--trials",
        "2",
        "--seeds",
        "7",
        "--sequences",
        "4",
        "--sequence-length",
        "64",
        "--warmup",
        "128",
        "--output",
        str(output),
        "--best-config",
        str(best),
        *mode,
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    library = str(Path(args.library).resolve())
    with tempfile.TemporaryDirectory(prefix="sbm-tune-test-") as directory:
        root = Path(directory)
        automatic = run_tune(library, root, ["--max-workers", "2"], "auto")
        fixed = run_tune(library, root, ["--fixed-jobs", "1"], "fixed")
    assert [item["id"] for item in automatic["candidates"]] == [
        item["id"] for item in fixed["candidates"]
    ]
    for automatic_item, fixed_item in zip(automatic["candidates"], fixed["candidates"]):
        automatic_nll = automatic_item["runs"]["7"]["result"]["eval_cross_entropy"]
        fixed_nll = fixed_item["runs"]["7"]["result"]["eval_cross_entropy"]
        assert abs(automatic_nll - fixed_nll) < 1e-12
    print("Tune scheduler integration passed")


if __name__ == "__main__":
    main()
