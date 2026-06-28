#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_manifest(path: Path) -> None:
    payload = {
        "schema": "sbm-corpus-manifest",
        "status": "admissible_document_level",
        "splits": {
            "train": {
                "shards": [
                    {
                        "path": "train-000.sbt",
                        "tokens": 101,
                        "examples": 100,
                        "sequences": 3,
                    }
                ]
            },
            "validation": {
                "shards": [
                    {
                        "path": "validation-000.sbt",
                        "tokens": 51,
                        "examples": 50,
                        "sequences": 2,
                    }
                ]
            },
            "test": {
                "shards": [
                    {
                        "path": "test-000.sbt",
                        "tokens": 41,
                        "examples": 40,
                        "sequences": 2,
                    }
                ]
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    temp = Path(tempfile.mkdtemp())
    manifest = temp / "manifest.json"
    write_manifest(manifest)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "inspect_corpus_manifest.py"),
            str(manifest),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["schema"] == "sbm-corpus-manifest"
    assert summary["splits"]["train"]["examples"] == 100
    assert summary["splits"]["validation"]["tokens"] == 51
    assert summary["total_examples"] == 190
    output = temp / "view" / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "derive_sbm_manifest_view.py"),
            "--source",
            str(manifest),
            "--output",
            str(output),
            "--train-examples",
            "100",
            "--validation-examples",
            "50",
            "--test-examples",
            "40",
            "--name",
            "fixture-view",
        ],
        check=True,
    )
    derived = json.loads(output.read_text(encoding="utf-8"))
    assert derived["schema"] == "sbm-corpus-manifest"
    assert derived["view_name"] == "fixture-view"
    assert derived["source_manifest"].endswith("manifest.json")
    assert derived["splits"]["train"]["shards"][0]["path"] == "../train-000.sbt"


if __name__ == "__main__":
    main()
