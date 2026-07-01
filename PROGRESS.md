# SPM Project Progress

> This file tracks current project state for any agent resuming work.
> It is a summary, not a log — for full chronological detail see RESEARCH_LOG.md.

## Active objective

持续进行正式的工程推进，保持和计划文件同步，饱和式推进，保持连贯性。

当前重点是 **Global Predictive Address Field (GPAF)** 的工程收尾和文档完备化。
GPAF 的 engineering milestone 已完成并通过 10M 真实语料验证。当前是所有 6 个 Task
的收尾提交和 PR 合并准备。

## Branch state

- **Branch:** `codex/explore-research-project-repository-mgvpni` (PR #1)
- **Status:** GPAF 实现已完成，21/21 测试通过，10M 验证完成，文档已更新
- **Key documents:**
  - Spec: `docs/superpowers/specs/2026-06-30-global-predictive-address-field-design.md`
  - Plan: `docs/superpowers/plans/2026-06-30-global-predictive-address-field.md`
  - 10M result: appended in `RESEARCH_LOG.md`

## GPAF Implementation Status (plan `2026-06-30-global-predictive-address-field.md`)

| Task | Status | Commit(s) |
|---|---|---|
| T1: Route-source and token-similarity diagnostics | Done | `392ea7b` |
| T2: Shadow GPAF role-key observation | Done | `392ea7b` |
| T3: Bounded GPAF candidate retrieval | Done | `392ea7b` |
| T4: Slot lifecycle and frozen ablation | Done | `392ea7b`, `d46b731` |
| T5: Structural-call role keys | Done | `d46b731` |
| T6: Experiment gates, presets and documentation | Done | (this commit) |

### 10M FineWeb-Edu Results (2026-07-01)

| Configuration | Seed 7 | Seed 11 | Seed 19 | Mean (7/11) | Topology |
|---|---|---|---|---|---|
| upgrade-v1 (baseline) | 6.21185 | 6.21391 | 6.27662 | **6.2129** | 2,2,5 |
| gpaf-retrieval-v1 | 6.21365 | 6.21517 | 6.27561 | **6.2144** | 2,2,5 |

GPAF retrieval is neutral at the noise level (+0.0015 nats on seeds 7/11).
GPAF internal state (seed 7): 59.9M observations, 21 unique role keys,
19 Active slots, 118M candidates returned. Frozen ablation shows +0.203 nats
codelength gain when GPAF slots are removed, with ~0.100 nats false positive
cost. The architecture works correctly; net benefit is currently within eval
noise due to conservative scoring and no description-cost gating.

### GPAF Config Presets

| Preset | Description |
|---|---|
| `gpaf-shadow-v1` | Role-key observation only; no prediction change. Verified on 10M. |
| `gpaf-retrieval-v1` | Full bounded candidate retrieval + structural call roles. Experimental. |

## Completed: Adaptive Computation Upgrade (plan `2026-06-30-adaptive-computation-upgrade.md`)

| Item | Commit |
|---|---|
| U2.1 Momentum | `8094de0` |
| U1.1 Adaptive beam width | `81abd27` |
| U1.2 Iterative refinement | `59c1955` |
| U3.1 Multi-hop content following | `1b730de` |
| Presets (`r3-baseline`, `upgrade-v1`, `upgrade-v1-adaptive`) | `3e9b661` |
| 10M validation | `bfc0af3` |

Key finding: **momentum with bias correction improves NLL by 0.111 nats** on 10M.
Multi-hop content following adds ~0.018 nats on 10M. Adaptive beam + refinement
is neutral in current configuration.

## Next work queue

1. **GPAF research next steps:** description/execution cost attribution, automatic
   frozen-codelength slot acceptance, richer per-slot ablation breakdown.
2. **R3 100M heterogeneous stream gate:** `docs/superpowers/plans/2026-06-28-r3-100m-heterogeneous-stream.md`
   — requires manifest construction.
3. Do not claim language semantics from GPAF unless real-data provenance,
   frozen validation, multi-seed stability, shard transfer and strong controls pass.

## Open theoretical gates

GPAF is intended to create room for global sparse retrieval to emerge from
predictive role reuse. It does not by itself solve content-conditioned variable
binding, relation-following, task-comparable topology value, long-horizon credit
or stable cross-domain language structure.
