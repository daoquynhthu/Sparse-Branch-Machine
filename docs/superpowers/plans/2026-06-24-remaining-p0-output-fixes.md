# Remaining P0 Output Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve node-wide sparse-decision learning decay, magnitude-only eviction and non-conserved multi-channel output mass.

**Architecture:** Extend each bounded sparse entry with local evidence, replace eviction with a deterministic evidence/probation policy, and normalize represented channel weights globally before existing within-channel routing. Add bounded diagnostics before changing policy so old and new behavior remain comparable.

**Tech Stack:** C++20, existing shared-library diagnostics JSON, CTest scaling fixtures, Python corpus runner.

---

### Task 1: Add bounded saturation and churn diagnostics

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_storage.cpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/modules/experiment.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `tests/test_scaling.cpp`

- [x] Write a failing capacity fixture requiring insertion, eviction,
  probable-reconstruction and saturated-node diagnostics.
- [x] Run `sbm_scaling_sparse_capacity` and confirm missing-field RED.
- [x] Add bounded counters and one 64-bit evicted-decision mask per node; count
  current saturated nodes in `diagnostics()`.
- [x] Verify all scaling tests and commit
  `feat: instrument sparse output capacity pressure`.

### Task 2: Implement per-decision learning and evidence eviction

**Files:**
- Create: `include/sbm/detail/sparse_output.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Modify: `tests/test_scaling.cpp`

- [x] Write failing unit tests for deterministic probation-aware victim
  selection, high-evidence near-zero retention and tie-breaking.
- [x] Write a failing machine fixture showing a new decision in a mature node
  separates from a prior-only control after one update.
- [x] Move `SparseOutputEntry` to the detail header with visits, gain EMA and
  last-update step; implement the specified retention score and selector.
- [x] Change mutable lookup to return the entry, schedule learning from entry
  visits, and update branch coding-gain evidence after prediction.
- [x] Merge matching entry evidence with saturating visits and preserve sorted
  decision order and hard capacity.
- [x] Run full tests, compare fixed 10k smoke quality/bytes/churn, and commit
  `fix: learn and retain sparse decisions by evidence`.

### Task 3: Conserve cross-channel responsibility mass

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_routing.cpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Modify: `src/modules/experiment.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `tests/test_scaling.cpp`

- [x] Write a failing two-channel fixture requiring maximum responsibility-mass
  error at most `1e-6` and a single-channel equality control.
- [x] Normalize represented raw channel weights to one before exact-region mass
  subdivision; do not change channel score calculation.
- [x] Add maximum responsibility-mass error diagnostics and JSON output.
- [x] Run full tests and commit `fix: conserve cross-channel output mass`.

### Task 4: Experimental gates and issue closure

**Files:**
- Modify: `ISSUES.md`
- Modify: `DESIGN_NOTES.md`
- Modify: `API.md`
- Modify: `RESEARCH_LOG.md`
- Modify: this plan

- [x] Run the one-shard fixed-topology smoke and record NLL, unigram gap,
  throughput, bytes, saturation, evictions and probable reconstructions.
- [x] Run the 1M/0.1M fixed-topology seeds 7, 11 and 19. Accept P0-2/P0-3 only
  within the `0.5%` NLL gate and with bounded capacity.
- [x] Run a multi-channel synthetic calibration checking conserved mass and
  finite topology credit; do not claim language improvement from it.
- [x] Run Python compilation, full build, CTest and `git diff --check`.
- [x] Move P0-2/P0-3/P0-5 to resolved only when their respective gates pass;
  record negative results without hiding them.
- [x] Commit `research: validate remaining P0 output fixes`.
