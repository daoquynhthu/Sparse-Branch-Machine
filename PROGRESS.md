# SPM Project Progress

> This file tracks current project state for any agent resuming work.
> It is a summary, not a log — for full chronological detail see RESEARCH_LOG.md.

## Active objective

持续进行正式的工程推进，保持和计划文件同步，饱和式推进，保持连贯性。

当前重点是 **Global Predictive Address Field (GPAF)** 的工程收尾和文档完备化。
GPAF 的 engineering milestone 已完成并通过 10M 真实语料验证。当前是所有 6 个 Task
的收尾提交和 PR 合并准备。

## Branch state

- **Branch:** `theory-alignment-v9`
- **Status:** GPAF 实现已完成，22/22 测试通过，10M 验证完成 (pre-fix) + 4 defects fixed (post-fix), 文档已更新
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
| T6: Experiment gates, presets and documentation | Done | `a76977b` |

## GPAF Honest-Value and Calibrated-Scoring Status (plan `2026-07-02-gpaf-honest-value-and-calibrated-scoring.md`)

| Task | Status | Commit(s) |
|---|---|---|
| HV1: Normalize cost/net-value to per-example means | Done | `2c9e4d7` |
| HV2: Stop attributing GPAF-overlap logits | Done | `3be364f` |
| HV3: Live costed-value writer + evidence-based lifecycle gating | Done | `17d77c2` |
| HV4: Calibrated GPAF-unique candidate scoring | Done | `28814c0` |
| HV5: Docs + real-corpus handoff | Done | (current) |

### 10M FineWeb-Edu Results (2026-07-01)

| Configuration | Seed 7 | Seed 11 | Seed 19 | Mean (7/11) | Topology |
|---|---:|---:|---:|---:|---|
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

## 10M costed-value validation (2026-07-02)

Key finding: **GPAF's positive ablation signal comes entirely from overlapping
locally-reachable candidates.** Unique GPAF-only candidates have negative gain
(-0.0013 nats per example). All three role keys have negative net value after
subtracting execution cost. Full report in `RESEARCH_LOG.md`.

## Honest-value and calibrated-scoring fixes (2026-07-02)

Plan: `docs/superpowers/plans/2026-07-02-gpaf-honest-value-and-calibrated-scoring.md`
Investigation: `docs/superpowers/research/2026-07-02-gpaf-architecture-investigation.md`

Four defects identified in the 10M GPAF measurement surface and fixed:

| Task | Fix | Commit |
|---|---|---|
| HV1 | Normalize GPAF cost/net-value aggregation to per-example means (was raw sums, unit-incoherent with gain) | `2c9e4d7` |
| HV2 | Stop attributing GPAF-overlap logits to GPAF causal ablation value (was +0.2288 artifact) | `3be364f` |
| HV3 | Implement live `gpaf_slot_costed_net_value_` writer + evidence-based Probe→Active/Active→Quarantined gating (was dead map, permanently-blocking no-op gate) | `17d77c2` |
| HV4 | Score GPAF-unique candidates from calibrated slot value, not token-signature Hamming similarity; block on non-positive value | `28814c0` |

Synthetic test evidence (all passing):
- `gpaf_costed_gate`: blocking branch (large `gpaf_execution_cost_weight=5.0F` → no promotions) passes; allowing branch (`gpaf_execution_cost_weight=0.0F`, 4096 steps → promotions within ≈2048 steps) passes.
- `gpaf_retrieval`: prefill-test confirms `gpaf_unique_active_nodes == 0U` when no slot value exists yet; bounded-run test still sees positive unique and overlap active nodes.
- Full CTest: 22/22 pass (including non-GPAF tests), no regression.
- `gpaf_slot_costed_net_value_` now has a live EMA writer per role key; the map is no longer empty after training.
- `gpaf_frozen`: aggregate `gpaf_codelength_gain` now equals `gpaf_unique_codelength_gain` (both are unique-only), `gpaf_overlap_codelength_gain` separated as non-causal diagnostic.

**NOTE:** None of this has been validated on real corpus data (10M FineWeb-Edu)
yet. The synthetic tests prove the fix mechanics work correctly, but whether
they change GPAF's real-corpus NLL requires re-running `upgrade-v1` vs
`gpaf-retrieval-v1` with `gpaf_execution_cost_weight` and
`gpaf_active_requires_positive_net_value` explicitly configured.

## Next work queue

1. **Real-corpus re-validation:** run 10M FineWeb-Edu `upgrade-v1` vs
   `gpaf-retrieval-v1` with the fixed code and explicit cost/admission
   config to see whether honest measurement and calibrated scoring change
   the NLL picture. See
   `docs/superpowers/plans/2026-07-02-gpaf-honest-value-and-calibrated-scoring.md`.
2. **GPAF research next steps (post-validation):** if the fix reveals positive
   unique GPAF value, proceed with binding-conditioned role keys (Proposal B),
   role-transition addressing (Proposal C), or epistemic-state addressing
   (Proposal E) per the investigation doc. If not, focus on enriching local
   address programs instead.
3. **R3 100M heterogeneous stream gate:** `docs/superpowers/plans/2026-06-28-r3-100m-heterogeneous-stream.md`
   — gated on GPAF showing positive unique signal post-fix.
4. Do not claim language semantics from GPAF unless real-data provenance,
   frozen validation, multi-seed stability, shard transfer and strong controls pass.

## Open theoretical gates

GPAF is intended to create room for global sparse retrieval to emerge from
predictive role reuse. Current implementation shows that the role keys are not
yet discriminative enough — GPAF mostly finds locally reachable nodes. This is
a key design challenge, not an invalidation of the concept. The four fixes in
the honest-value plan remove measurement artifacts that obscured this picture;
binding-conditioned keys (Proposal B) and role-transition addressing
(Proposal C) remain the next concrete engineering steps if the fixed
measurement confirms unique positive value.
