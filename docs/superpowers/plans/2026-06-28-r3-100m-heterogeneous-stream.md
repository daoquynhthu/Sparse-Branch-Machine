# R3 100M Heterogeneous Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the first R3 100M-token heterogeneous FineWeb-Edu stream gate for the completed address-semantics framework.

**Architecture:** R3 is split into two contracts: corpus materialization and model validation. The corpus contract produces a document-level `sbm-corpus-manifest` with about 100M train examples plus sealed validation/test slices; the model contract uses the existing resumable/checkpointed runners, structural-value admission and bounded-work diagnostics. No experiment result is accepted unless the manifest provenance and split discipline are explicit.

**Tech Stack:** C++20 SBM runtime, Python corpus runners, `E:\SPM_DATA_PIPELINE` for corpus construction, mapped `.sbt` token shards, CMake/CTest, PowerShell orchestration.

---

## Current State

Completed gates:

- 10M/1M multi-seed validation passed.
- 10M/1M held-out test-transfer passed.
- default description-only structural-value admission passed.
- training ranking decode was removed from the training hot path.

Current local corpus inventory:

```text
E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train10m_eval1m
  train:      9,989,918 examples
  validation:   998,482 examples
  test:         998,429 examples

E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_10m
  train:      9,989,918 examples
  validation: 88,064,143 examples
  test:       87,202,822 examples
```

There is no local 100M training manifest yet. The large validation/test shards
must not be silently repurposed as training data. R3 therefore starts with
manifest construction, not model training.

## Files

- Create: `scripts/inspect_corpus_manifest.py`
- Create: `scripts/derive_sbm_manifest_view.py`
- Create: `tests/test_manifest_tools.py`
- Create: `research_results/r3_100m_manifest.md`
- Create: `research_results/r3_100m_multiseed.md`
- Modify: `ROADMAP_REAL_DATA.md`
- Modify: `RESEARCH_LOG.md`
- Modify: this plan file

## Task 1: Add Manifest Inspection Tool

**Files:**
- Create: `scripts/inspect_corpus_manifest.py`
- Create: `tests/test_manifest_tools.py`

- [x] **Step 1: Write fixture manifest test**

Create `tests/test_manifest_tools.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_manifest(path: Path) -> None:
    payload = {
        "schema": "sbm-corpus-manifest",
        "status": "admissible_document_level",
        "splits": {
            "train": {
                "shards": [
                    {"path": "train-000.sbt", "tokens": 101, "examples": 100, "sequences": 3}
                ]
            },
            "validation": {
                "shards": [
                    {"path": "validation-000.sbt", "tokens": 51, "examples": 50, "sequences": 2}
                ]
            },
            "test": {
                "shards": [
                    {"path": "test-000.sbt", "tokens": 41, "examples": 40, "sequences": 2}
                ]
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "import tempfile, pathlib; print(tempfile.mkdtemp())"],
        stdout=subprocess.PIPE,
        text=True,
    ) as proc:
        temp = Path(proc.communicate()[0].strip())
    manifest = temp / "manifest.json"
    write_manifest(manifest)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_corpus_manifest.py"), str(manifest)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["schema"] == "sbm-corpus-manifest"
    assert summary["splits"]["train"]["examples"] == 100
    assert summary["splits"]["validation"]["tokens"] == 51
    assert summary["total_examples"] == 190


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run failing test**

Run:

```powershell
python tests\test_manifest_tools.py
```

Expected: fail because `scripts/inspect_corpus_manifest.py` does not exist.

- [x] **Step 3: Implement inspector**

Create `scripts/inspect_corpus_manifest.py`:

```python
#!/usr/bin/env python3
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
```

- [x] **Step 4: Verify inspector**

Run:

```powershell
python tests\test_manifest_tools.py
python scripts\inspect_corpus_manifest.py E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train10m_eval1m\manifest.json
git diff --check
```

Expected: test passes and the real manifest reports about 9.99M train examples.

- [x] **Step 5: Commit**

```powershell
git add scripts\inspect_corpus_manifest.py tests\test_manifest_tools.py
git commit -m "feat: inspect corpus manifests"
```

## Task 2: Add Safe Manifest View Derivation

**Files:**
- Create: `scripts/derive_sbm_manifest_view.py`
- Modify: `tests/test_manifest_tools.py`

- [x] **Step 1: Add failing derive-view test**

Append to `tests/test_manifest_tools.py` before `if __name__ == "__main__":`:

```python
def test_derive_view_cli(temp: Path) -> None:
    manifest = temp / "manifest.json"
    write_manifest(manifest)
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
```

Replace the current `main()` body with:

```python
def main() -> None:
    import tempfile

    temp = Path(tempfile.mkdtemp())
    manifest = temp / "manifest.json"
    write_manifest(manifest)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_corpus_manifest.py"), str(manifest)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["schema"] == "sbm-corpus-manifest"
    assert summary["splits"]["train"]["examples"] == 100
    assert summary["splits"]["validation"]["tokens"] == 51
    assert summary["total_examples"] == 190
    test_derive_view_cli(temp)
```

- [x] **Step 2: Run failing test**

Run:

```powershell
python tests\test_manifest_tools.py
```

Expected: fail because `derive_sbm_manifest_view.py` does not exist.

- [x] **Step 3: Implement view derivation**

Create `scripts/derive_sbm_manifest_view.py`:

```python
#!/usr/bin/env python3
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
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "view_name": args.name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4: Verify derive-view**

Run:

```powershell
python tests\test_manifest_tools.py
python scripts\derive_sbm_manifest_view.py --source E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_10m\manifest.json --output E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_eval10m_view\manifest.json --name fineweb_edu_v1_r1_eval10m_view --train-examples 9000000 --validation-examples 10000000 --test-examples 10000000
python scripts\inspect_corpus_manifest.py E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_eval10m_view\manifest.json
git diff --check
```

Expected: test passes. The real derive command succeeds only if the source
split capacities are sufficient; it must fail rather than silently reassigning
validation or test shards to train.

- [x] **Step 5: Commit**

```powershell
git add scripts\derive_sbm_manifest_view.py tests\test_manifest_tools.py
git commit -m "feat: derive safe corpus manifest views"
```

## Task 3: Produce Or Import The 100M Train Manifest

**Files:**
- Modify: `research_results/r3_100m_manifest.md`
- Modify: `RESEARCH_LOG.md`

- [ ] **Step 1: Inspect available release views**

Run locally:

```powershell
python scripts\inspect_corpus_manifest.py E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train10m_eval1m\manifest.json
python scripts\inspect_corpus_manifest.py E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_10m\manifest.json
```

Run in the data pipeline workspace:

```powershell
Set-Location E:\SPM_DATA_PIPELINE
python -m spm_data_pipeline.cli validate-release configs\fineweb_edu_v1.json views manifests tokenizers 2>$null
```

Expected: if no 100M train view exists, the command evidence must be recorded
as a data-availability blocker in `research_results/r3_100m_manifest.md`, not in
`ISSUES.md`.

- [ ] **Step 2: Materialize or import 100M**

If the data pipeline already has a 100M tokenized view, convert it to SPM mapped
shards and write:

```text
E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train100m_eval1m\manifest.json
```

If it does not exist, run the data pipeline materialization stage from
`E:\SPM_DATA_PIPELINE` or the remote release root, preserving document splits.
Do not repurpose validation/test as train.

- [ ] **Step 3: Record manifest evidence**

Create `research_results/r3_100m_manifest.md` with:

```markdown
# R3 100M Manifest

## Source

- source pipeline root:
- source release/view:
- tokenizer manifest:
- split policy:

## SPM Manifest

- path:
- train examples:
- validation examples:
- test examples:
- shard count:
- status:

## Gate

The manifest is admissible only if train, validation and test remain
document-disjoint and the tokenizer provenance is inherited from the release
manifest. Validation/test shards are not used for topology proposal calibration.
```

- [ ] **Step 4: Commit**

```powershell
git add research_results\r3_100m_manifest.md RESEARCH_LOG.md
git commit -m "data: record r3 100m manifest"
```

## Task 4: R3 Smoke And Throughput Admission

**Files:**
- Modify: `research_results/r3_100m_multiseed.md`
- Modify: `RESEARCH_LOG.md`

- [ ] **Step 1: Run short-window throughput probe**

Run:

```powershell
python scripts\run_throughput_probe.py --library E:\SPM\build-fast\libsbm_api.dll --manifest E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train100m_eval1m\manifest.json --train-examples 300000 --eval-examples 10000 --start-examples 50000 --step-examples 50000 --stable-windows 3 --stability-tolerance 0.08 --output E:\SPM_EXPERIMENTS\runs\r3_100m_throughput_probe.json --bucket-bits 14 --adaptive-topology --set address_lags=1 --set topology_enable_delta=false --set max_sparse_decisions_per_node=512 --set classification_learning_rate=0.8 --set classification_mature_learning_rate=0.2 --set record_channel_attribution=true --set topology_accept_uses_structural_value=true
```

Expected: reports stable speed or exits at 300k ceiling. This is a sizing
probe, not a quality claim.

- [ ] **Step 2: Run 5M/0.5M smoke from the 100M manifest**

Run:

```powershell
python scripts\run_corpus_batch.py --library E:\SPM\build-fast\libsbm_api.dll --manifest E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train100m_eval1m\manifest.json --eval-split validation --seeds 7,11,19 --output-dir E:\SPM_EXPERIMENTS\runs\r3_100m_smoke_5m_multiseed --skip-existing --max-workers 2 --worker-memory-mib 2300 --bucket-bits 14 --adaptive-topology --max-train-examples 5000000 --max-eval-examples 500000 --set address_lags=1 --set topology_enable_delta=false --set max_sparse_decisions_per_node=512 --set classification_learning_rate=0.8 --set classification_mature_learning_rate=0.2 --set record_channel_attribution=true --set topology_accept_uses_structural_value=true
```

Expected: all seeds complete, no accepted physical prune, positive program
attribution remains present.

- [ ] **Step 3: Commit smoke evidence**

Update `research_results/r3_100m_multiseed.md` with the probe and smoke table,
then run:

```powershell
git add research_results\r3_100m_multiseed.md RESEARCH_LOG.md
git commit -m "research: run r3 100m smoke gate"
```

## Task 5: Full R3 100M Multi-Seed Gate

**Files:**
- Modify: `research_results/r3_100m_multiseed.md`
- Modify: `RESEARCH_LOG.md`
- Modify: `ROADMAP_REAL_DATA.md`

- [ ] **Step 1: Run full 100M gate**

Run:

```powershell
python scripts\run_corpus_batch.py --library E:\SPM\build-fast\libsbm_api.dll --manifest E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train100m_eval1m\manifest.json --eval-split validation --seeds 7,11,19 --output-dir E:\SPM_EXPERIMENTS\runs\r3_100m_validation_multiseed --skip-existing --max-workers 2 --worker-memory-mib 4096 --bucket-bits 14 --adaptive-topology --set address_lags=1 --set topology_enable_delta=false --set max_sparse_decisions_per_node=512 --set classification_learning_rate=0.8 --set classification_mature_learning_rate=0.2 --set record_channel_attribution=true --set topology_accept_uses_structural_value=true
```

Expected: completion may require remote execution if local memory or wall time
is unacceptable. If moved remote, preserve the exact command parameters and
result paths in the result file.

- [ ] **Step 2: Analyze acceptance boundary**

Record:

- eval NLL per seed;
- current-token and interpolated controls;
- accepted/rejected/pruned counts;
- learned programs and operations;
- positive program-attribution counts;
- `avg_active`, `avg_candidates`, `max_bucket_candidates_inspected`;
- estimated bytes and examples/s;
- comparison to 10M gates.

- [ ] **Step 3: Update roadmap**

If the gate passes, update `ROADMAP_REAL_DATA.md` Phase R3 status with the
result and define the next transition to R4/R5. If it fails, record the failure
as research evidence in `RESEARCH_LOG.md`; add to `ISSUES.md` only if the
failure is a confirmed implementation blocker rather than a research outcome.

- [ ] **Step 4: Verify and commit**

Run:

```powershell
python tests\test_manifest_tools.py
git diff --check
```

Then commit:

```powershell
git add ROADMAP_REAL_DATA.md RESEARCH_LOG.md research_results\r3_100m_multiseed.md docs\superpowers\plans\2026-06-28-r3-100m-heterogeneous-stream.md
git commit -m "research: validate r3 100m gate"
```

## Self-Review

- Spec coverage: the plan covers R3 data availability, manifest safety, smoke
  sizing, full 100M multi-seed validation, bounded-work reporting and roadmap
  update.
- Placeholder scan: every task has concrete files, commands and expected
  outcomes. Data materialization has an explicit fail-safe: do not repurpose
  validation/test as train.
- Type consistency: new scripts use `sbm-corpus-manifest` and match the
  existing `run_corpus_training.py` / `run_corpus_batch.py` manifest contract.
