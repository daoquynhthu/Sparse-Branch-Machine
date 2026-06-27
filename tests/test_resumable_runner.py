#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sbm_runtime import Runtime  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
args = parser.parse_args()

runtime = Runtime(args.library)

with tempfile.TemporaryDirectory() as directory_text:
    directory = Path(directory_text)
    shard_path = directory / "train.sbt"
    manifest_path = directory / "manifest.json"
    complete_output = directory / "complete.json"
    resumed_output = directory / "resumed.json"
    complete_checkpoint = directory / "complete.ckpt.json"
    resumed_checkpoint = directory / "resumed.ckpt.json"

    tokens = []
    offsets = [0]
    for sequence in range(6):
        for position in range(12):
            tokens.append((sequence * 3 + position) % 16)
        offsets.append(len(tokens))
    with runtime.token_dataset_from_ids(tokens, 16, offsets) as dataset:
        dataset.write_token_shard(shard_path)

    manifest_path.write_text(
        json.dumps(
            {
                "schema": "sbm-corpus-manifest",
                "splits": {"train": {"shards": [{"path": shard_path.name}]}},
            }
        ),
        encoding="utf-8",
    )

    base_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_resumable_corpus_training.py"),
        "--library",
        args.library,
        "--manifest",
        str(manifest_path),
        "--seed",
        "17",
        "--bucket-bits",
        "6",
        "--max-train-examples",
        "48",
        "--checkpoint-every-examples",
        "8",
        "--set",
        "address_lags=1",
    ]

    subprocess.run(
        base_command
        + ["--output", str(complete_output), "--checkpoint", str(complete_checkpoint)],
        check=True,
    )
    subprocess.run(
        base_command
        + [
            "--output",
            str(resumed_output),
            "--checkpoint",
            str(resumed_checkpoint),
            "--stop-after-examples",
            "24",
        ],
        check=True,
    )
    interrupted = json.loads(resumed_output.read_text(encoding="utf-8"))
    assert interrupted["train_examples"] == 24
    subprocess.run(
        base_command
        + [
            "--output",
            str(resumed_output),
            "--checkpoint",
            str(resumed_checkpoint),
            "--resume",
        ],
        check=True,
    )

    complete = json.loads(complete_output.read_text(encoding="utf-8"))
    resumed = json.loads(resumed_output.read_text(encoding="utf-8"))
    assert complete["completed"] is True
    assert resumed["completed"] is True
    assert resumed["train_examples"] == 48
    assert complete["train_examples"] == resumed["train_examples"]
    assert math.isclose(
        complete["train_cross_entropy"],
        resumed["train_cross_entropy"],
        abs_tol=1e-6,
    )
    assert math.isclose(
        complete["train_mean_target_probability"],
        resumed["train_mean_target_probability"],
        abs_tol=1e-6,
    )

print("resumable runner tests passed")
