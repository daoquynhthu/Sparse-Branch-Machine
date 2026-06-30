# SPM Project Progress

> This file tracks current project state for any agent resuming work.
> It is a summary, not a log — for full chronological detail see RESEARCH_LOG.md.

## Active objective

持续进行正式的工程推进，保持和计划文件同步，饱和式推进，保持连贯性。
基础设施充分时避免保守增量，允许临时粗糙边缘，但最终状态必须经过验证。

当前重点：在 `adaptive-computation-upgrade` 分支上执行升级实验方案，已完成 U2.1/U1.1/U1.2 并验证 10M 效果。

## Branch state

- **Branch:** `adaptive-computation-upgrade` (created from `theory-alignment-v9`)
- **Status:** ahead of origin, uncommitted changes from baselines/ remain untracked
- **Last commits:**
  - `bfc0af3` research: 10M validation and presets for upgrade-v1
  - `6a89d21` fix: Adam bias correction for sparse decision momentum
  - `3e9b661` feat: config presets with r3-baseline and upgrade-v1
  - `59c1955` feat: iterative routing refinement with expanded neighbor radius
  - `81abd27` feat: adaptive beam width with confidence-based truncation
  - `8094de0` feat: per-entry Adam-like momentum for sparse decision logits

## Completed: Adaptive Computation Upgrade (plan `2026-06-30-adaptive-computation-upgrade.md`)

### Implemented

| Item | Status | Commit |
|---|---|---|
| U2.1 Per-entry Adam-like momentum | Done | `8094de0` |
| U1.1 Adaptive beam width | Done | `81abd27` |
| U1.2 Iterative refinement | Done | `59c1955` |
| Config presets (`r3-baseline`, `upgrade-v1`, `upgrade-v1-adaptive`) | Done | `3e9b661` |
| Adam bias correction fix | Done | `6a89d21` |
| 10M validation and result report | Done | `bfc0af3` |

### 10M FineWeb-Edu Results

| Configuration | eval NLL | Topology accepted | Bytes | Tok/s |
|---|---:|---:|---:|---:|
| r3-baseline | 6.3412 | 5 | 2.16 GB | 15,756 |
| **upgrade-v1 (momentum)** | **6.2305 ± 0.0001** | 1 | 895 MB | 17,000 |
| upgrade-v1-adaptive | 6.2559 ± 0.0002 | 1 | 896 MB | 12,444 |

Key finding: **momentum with bias correction improves NLL by 0.111 nats/token**, reduces model size by 59%, and increases throughput by 8%. Adaptive beam width + refinement are neutral/slightly negative in the current configuration.

Full report: `research_results/adaptive_computation_upgrade_10m_20260630.md`

## Next: 100M Heterogeneous Stream (R3) with upgrade-v1

**Plan:** `docs/superpowers/plans/2026-06-28-r3-100m-heterogeneous-stream.md`

Use the proven `upgrade-v1` preset on the 100M FineWeb-Edu stream gate. Requires manifest construction first.

## Open Theoretical Gates

Same as before, with momentum validated on 10M and transfer to 100M pending.
