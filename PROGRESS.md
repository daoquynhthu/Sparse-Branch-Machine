# SPM Project Progress

> This file tracks current project state for any agent resuming work.
> It is a summary, not a log — for full chronological detail see RESEARCH_LOG.md.

## Active objective

持续进行正式的工程推进，保持和计划文件同步，饱和式推进，保持连贯性。
基础设施充分时避免保守增量，允许临时粗糙边缘，但最终状态必须经过验证。

## Branch state

- **Branch:** `theory-alignment-v9` (ahead origin by 15 commits)
- **Working tree:** clean (only `.idea/` untracked)
- **Last commit:** `2c229cf feat: restore dependency closure` (2026-06-29)

## Completed: Neuronal Address Semantics (2026-06-27 plan)

**Plan:** `docs/superpowers/plans/2026-06-27-neuronal-address-semantics.md`
**Status:** All 9 tasks complete. The plan also accumulated extensive
beyond-plan work documented as "completion" notes in the plan file.

### The 15 commits on theory-alignment-v9

| Commit | Description |
|---|---|
| `a962bca` | research: run reuse-aware 1m validation |
| `9c3a497` | feat: condition callers on dependency bindings |
| `f9e89b6` | feat: attribute dependency call usage |
| `f2a4e82` | feat: type address program IO contracts |
| `dd42aa4` | feat: enforce typed dependency calls |
| `f6bd7ca` | feat: version dependency graph edges |
| `a5d0102` | feat: emit dependency edge graph |
| `123031e` | feat: restore recoverable topology channels |
| `4aa1b13` | fix: enforce dependency-aware channel enablement |
| `572c51e` | feat: expose effective channel lifecycle state |
| `7eb7fec` | feat: persist effective channel state in experiments |
| `00aa383` | feat: annotate dependency graph effective state |
| `353691b` | feat: name topology lifecycle decisions |
| `1f95358` | feat: summarize direct caller availability |
| `2c229cf` | feat: restore dependency closure |

### What was built (summary)

The address-semantics framework replaced one-shot sparse address signatures
with an executable, auditable address-program system:

- **Explicit address execution:** AddressProgram runs through an interpreter
  producing frames with operation, source, binding, signature, cost and
  dependency fields (Tuple, DeltaMod, ContentMatch, ContentFollow).
- **Persistent lifecycle:** Accepted structures use quarantine, masking and
  recoverable retirement instead of physical deletion. Channels have explicit
  phases (Seed -> Probe -> Active -> Quarantined/RecoverableRetired/PhysicalErase).
- **Dependency-aware attribution:** Callers and prerequisites can be ablated
  separately. The dependency graph carries edge kinds, caller/dependency
  generations, binding keys, and call evidence.
- **Structural value gates:** Accepted topology decisions report description
  cost, execution cost, and structural value (credit minus costs). An optional
  gate flag switches acceptance to use structural value.
- **Graph-level rollback:** `restore_dependency_closure(channel)` restores a
  producer and its recoverably masked direct callers.
  `effective_direct_caller_count` and `blocked_direct_caller_count` audit how
  many committed consumers are currently routable vs masked.
- **Checkpoint/resume:** Exact model-state and corpus training-run checkpoint
  with resumable evaluation. Multi-seed 10M/1M validation and held-out test
  transfer passed.

### Validation gates passed (2026-06-29)

- 10M/1M multi-seed validation: passed
- 10M/1M held-out test transfer: passed
- Default description-only structural-value admission: passed
- Training ranking decode removed from hot path
- `ctest --test-dir build-fast --output-on-failure`: 15/15 passed
- `python tests\test_python_api.py`: passed
- `git diff --check`: clean

## Next: R3 100M Heterogeneous Stream

**Plan:** `docs/superpowers/plans/2026-06-28-r3-100m-heterogeneous-stream.md`
**Status:** Not started.

### Prerequisites

- Build or import a true document-level 100M training manifest
  (`sbm-corpus-manifest` with ~100M train + sealed validation/test slices)
- Do not repurpose validation or test shards as training data
- Run a short throughput/sizing gate before full 100M multi-seed validation
- Keep 10M/1M results as the R3 admission baseline

### Corpus inventory

    E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train10m_eval1m
      train:      9,989,918 examples
      validation:   998,482 examples
      test:         998,429 examples

    E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_10m
      train:      9,989,918 examples
      validation: 88,064,143 examples
      test:       87,202,822 examples

There is no local 100M training manifest yet. R3 starts with manifest
construction, not model training.

## Open theoretical gates

From `THEORY_ALIGNMENT.md` Section 8, in order:

1. Establish real-corpus baselines and failure diagnostics — **10M done, 100M pending**
2. Express structural value in prequential codelength and explicit complexity — **done**
3. Identify failures not explained by bounded positional programs — **pending R3**
4. Propose one minimal content-conditioned primitive — **partially implemented**
5. Extend lineage and dependency-aware ablation to typed state, reusable caller
   graphs and rollback — **explicit closure restore implemented**
6. Test transfer across documents, shards and seeds — **10M verified, 100M pending**
7. Only then consider composition, calls or deeper program graphs — **not started**

## Build and test commands

    cmake --build build-fast --config Release
    ctest --test-dir build-fast --output-on-failure
    python tests\test_python_api.py --library E:\SPM\build-fast\libsbm_api.dll
    git diff --check -- . ':!.idea'

## Interruption note

The previous session (Codex, gpt-5.5) was interrupted on 2026-06-29 after
commit `2c229cf` when the user lost OpenAI account access. No uncommitted
changes remain. The active objective was carried over from the session's
persistent goal.
