# GPAF Causal Attribution and Costed Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether GPAF contributes unique predictive value, instead of only overlapping with local route candidates, before adding any stronger retrieval scoring or running another real-corpus quality gate.

**Architecture:** Add causal attribution counters first: GPAF-unique candidates versus GPAF-overlap candidates, then extend frozen ablation and slot value accounting in later tasks. Keep retrieval behavior unchanged until frozen, cost-adjusted unique value is positive. Real-corpus validation remains external and must be requested from the user.

**Tech Stack:** C++20 (`sbm_machine`, `sbm_experiment`, `sbm_api`), C ABI JSON diagnostics, Python presets, CTest, existing synthetic scaling tests.

---

## File structure

- Modify `include/sbm/types.hpp`: add diagnostics fields for GPAF-unique and GPAF-overlap candidate attribution.
- Modify `include/sbm/machine.hpp`: add machine counters and helper behavior for overlap detection.
- Modify `src/modules/machine_routing.cpp`: count whether GPAF resident insertion is unique or overlaps an already retrieved candidate.
- Modify `src/modules/machine_maintenance.cpp`: expose new counters in `Diagnostics`.
- Modify `src/api/c_api.cpp`, `src/modules/experiment.cpp`, `src/modules/token_experiment.cpp`: emit JSON fields.
- Modify `src/modules/machine_checkpoint.cpp`: persist counters and bump checkpoint magic.
- Modify `tests/test_scaling.cpp` and `tests/test_c_api.cpp`: prove counters exist and are non-empty under bounded synthetic retrieval.
- Update `DESIGN_NOTES.md`, `PROGRESS.md`, and `RESEARCH_LOG.md` only after verified implementation evidence.

## Task 1: GPAF unique-vs-overlap candidate diagnostics

Status: implemented locally in the current branch; real-corpus validation is not required for this diagnostics-only slice.

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_routing.cpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Modify: `src/api/c_api.cpp`
- Modify: `src/modules/experiment.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `src/modules/machine_checkpoint.cpp`
- Test: `tests/test_scaling.cpp`
- Test: `tests/test_c_api.cpp`

- [x] **Step 1: Add failing diagnostics assertions**

Add tests requiring:

```text
gpaf_unique_candidates_returned
gpaf_overlap_candidates_returned
```

Run:

```bash
cmake --build build-fast --target sbm_scaling_tests sbm_c_api_tests -j2
./build-fast/sbm_scaling_tests gpaf_retrieval
./build-fast/sbm_c_api_tests
```

Expected before implementation: FAIL because the fields/counters do not exist.

- [x] **Step 2: Add diagnostics fields and counters**

Add two monotonic counters:

```text
gpaf_unique_candidates_returned: GPAF resident was not already in the candidate set.
gpaf_overlap_candidates_returned: GPAF resident was already present from exact bucket, control edge or neighbor retrieval.
```

Do not change route selection, candidate budgets, scoring, or active beam behavior.

- [x] **Step 3: Emit and persist counters**

Expose both counters through machine diagnostics JSON, vector/token experiment JSON and checkpoints. Bump checkpoint magic because the binary format changes.

- [x] **Step 4: Verify behavior-preserving attribution**

Run:

```bash
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests sbm_scaling_tests -j2
ctest --test-dir build-fast --output-on-failure -R "sbm_c_api|sbm_scaling_gpaf_retrieval|sbm_scaling_gpaf_checkpoint"
python tests/test_presets.py
git diff --check
```

Expected: PASS. Existing GPAF candidate counts remain bounded; new unique + overlap counters explain GPAF resident attempts without changing predictions.

- [x] **Step 5: Commit**

```bash
git add include/sbm/types.hpp include/sbm/machine.hpp src/modules/machine_routing.cpp src/modules/machine_maintenance.cpp src/api/c_api.cpp src/modules/experiment.cpp src/modules/token_experiment.cpp src/modules/machine_checkpoint.cpp tests/test_scaling.cpp tests/test_c_api.cpp docs/superpowers/plans/2026-07-01-gpaf-causal-attribution-and-costed-admission.md
git commit -m "feat: add GPAF unique overlap attribution"
```

## Task 2: GPAF-unique active-route attribution

Status: complete. Selected active nodes retain GPAF unique-vs-overlap provenance, emit monotonic active-route counters, and frozen sparse-output ablation now reports unique versus overlap groups separately.

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_routing.cpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Test: `tests/test_scaling.cpp`

- [x] Track whether a selected active node was GPAF-unique or GPAF-overlap at candidate time.
- [x] Add frozen sparse-output diagnostics separating aggregate GPAF ablation into unique versus overlap groups.
- [x] Verify frozen evaluation remains read-only for active-route counters while emitting per-step unique/overlap frozen ablation diagnostics.

## Task 3: Costed slot value accounting

Status: complete. Frozen GPAF ablation keys now report resident count, reuse count, execution-budget cost and diagnostic net value. This remains diagnostic-only and does not affect Probe->Active promotion.

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_topology.cpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Test: `tests/test_scaling.cpp`

- [x] Add per-slot frozen gain, false-positive cost, resident count and reuse counters.
- [x] Compute a diagnostic-only net value: gain minus false-positive cost minus execution budget cost.
- [x] Do not change Probe->Active promotion until Task 4.

## Task 4: Costed Active admission gate

Status: complete. Added a disabled-by-default gate requiring positive diagnostic costed slot value before Probe slots can become Active. Existing observation/resident preconditions remain mandatory, and default behavior is unchanged.

**Files:**
- Modify: `src/modules/machine_topology.cpp`
- Modify: `tests/test_scaling.cpp`

- [x] Add an optional disabled-by-default gate requiring positive costed slot value before Probe->Active promotion.
- [x] Keep existing `gpaf_probe_min_observations` and `gpaf_probe_min_residents` as engineering preconditions.
- [x] Verify old behavior remains default unless the costed gate flag is enabled.

## Task 5: Real-corpus handoff gate

No code change. Stop and ask the user to run real-corpus experiments when Tasks 1-4 pass synthetic/API gates.

Required external runs:

```text
upgrade-v1
gpaf-shadow-v1
gpaf-retrieval-v1
```

Required result fields:

```text
gpaf_unique_candidates_returned
gpaf_overlap_candidates_returned
gpaf_unique_active_nodes
gpaf_overlap_active_nodes
gpaf_unique_ablation_gain
gpaf_overlap_ablation_gain
slot costed net value summaries
```

## Self-review checklist

- Spec coverage: follows the approved mechanism-audit-first path before score boosts or 100M claims.
- Placeholder scan: no placeholder markers; later tasks are intentionally scoped but concrete.
- Type consistency: counter names use `gpaf_unique_*` and `gpaf_overlap_*` consistently.
