# Adaptive Computation Upgrade Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade SPM's architecture to improve held-out NLL from 6.341 toward Transformer baseline (5.527) through adaptive computation, stronger local optimization, and deeper context modeling.

**Architecture:** CPU-first sparse address-semantic machine. All upgrades must preserve §4.1 constraints (no dense global layers, no full-network backprop, bounded active work).

**Tech Stack:** C++20 SBM runtime, Python corpus runners, `E:\SPM_DATA_PIPELINE` for corpus construction, mapped `.sbt` token shards, CMake/CTest, PowerShell orchestration.

---

## Diagnosis: Why SPM Underperforms

The Transformer small (10.5M params) achieves 5.527 nats on 10M FineWeb-Edu, beating SPM's 6.341 by 0.814 nats. Root cause analysis:

| Bottleneck | Current Implementation | Impact |
|---|---|---|
| **Fixed beam_width=6** | `select_route()` in `machine_routing.cpp:152,168` uses hard cap | Only 6 of 193K nodes contribute per token |
| **Simple 1/√visits learning** | `token_sparse_output.cpp:602-608` uses per-entry gradient | No momentum, no variance estimation |
| **Single-pass processing** | `step_token_sparse()` in `token_sparse_output.cpp:195-652` | No iterative refinement for hard tokens |
| **Positional addressing** | Address programs are primarily lag-based | Limited content-dependent relation modeling |
| **No state-dependent routing** | Routing depends on address signatures only | Execution paths not truly state-dependent |

## Upgrade Phases

### Phase 0: Quick Wins (No Architectural Changes)

- [ ] 0.1 Multi-epoch training
- [ ] 0.2 beam_width sweep
- [ ] 0.3 LR tuning

**Hypothesis:** Multi-epoch training and hyperparameter tuning can immediately improve NLL, validating the optimizer bottleneck hypothesis.

| Experiment | Change | Expected |
|---|---|---|
| 0.1 Multi-epoch training | Loop 3x over data, freeze topology after epoch 1 | +0.15-0.35 nats |
| 0.2 beam_width sweep | beam_width = 8, 10, 12 | +0.05-0.1 nats |
| 0.3 LR tuning | Sweep classification_learning_rate | +0.02-0.05 nats |

**Files:**
- Modify: `scripts/run_corpus_batch.py` (or equivalent training script)
- Modify: `types.hpp` (parameter defaults)
- Modify: `c_api.cpp` (parameter registry)

### Phase U1: Adaptive Computation Budget

**Hypothesis:** Hard tokens benefit from more nodes and more rounds. Average active nodes stay O(1).

#### U1.1: Adaptive Beam Width

- [x] Add `beam_width_min` and `confidence_threshold` to `Config`
- [x] Implement truncation logic in `select_route()` after responsibility assignment
- [x] Register params in C API
- [x] Update checkpoint format
- [x] Write test `verify_adaptive_beam_width`

**Current:** `select_route()` in `machine_routing.cpp` uses `config_.beam_width` as hard cap.

**Change:**
- Select up to `beam_width` nodes as before
- Assign responsibilities
- If `selected.size() > beam_width_min` and max responsibility >= `confidence_threshold`, truncate to top `beam_width_min` nodes
- Re-assign responsibilities

**Files:**
- Modify: `types.hpp` additions: `beam_width_min{6}`, `confidence_threshold{0.8F}`
- Modify: `machine_routing.cpp` truncation logic after `assign_responsibilities()`
- Modify: `c_api.cpp` parameter registry
- Modify: `machine_checkpoint.cpp` format fields

**Status:** Implemented in commit `81abd27`.

**Constraint compliance (§4.2):** Average active nodes remain O(1) because truncation keeps the active set at `beam_width_min` on confident tokens. `beam_width` is still a fixed upper bound.

**Expected:** +0.1-0.2 nats

#### U1.2: Iterative Refinement

- [x] Add `max_refinement_rounds` and `refinement_confidence_threshold` to `Config`
- [x] Add `max_radius` parameter to `candidate_ids()` and `select_route()`
- [x] Implement refinement loop in `step_token_sparse()`
- [x] Register params in C API
- [x] Update checkpoint format
- [x] Write test `verify_refinement_rounds`

**Current:** `step_token_sparse()` is single-pass.

**Change:**
- After first `select_route()`, compute max responsibility
- If max responsibility < `refinement_confidence_threshold` and rounds remain, call `select_route(signatures, radius)` with an expanded neighbor radius
- Repeat up to `max_refinement_rounds` times
- All refinement happens before `output_tree.target_path(target_token, ...)`

**Files:**
- Modify: `include/sbm/machine.hpp`
- Modify: `include/sbm/types.hpp`
- Modify: `src/modules/machine_routing.cpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/api/c_api.cpp`
- Modify: `src/modules/machine_checkpoint.cpp`
- Modify: `tests/test_scaling.cpp`

**Status:** Implemented in commit `59c1955`.

**Constraint compliance (§4.5):** All refinement completes before target is used for prediction or learning.

**Expected:** +0.05-0.15 nats

### Phase U2: Stronger Local Optimization

**Hypothesis:** Local Adam-like updates outperform 1/√visits mean, narrowing the optimizer gap.

#### U2.1: Per-Node Momentum and Adaptive Learning Rate

- [x] Add `momentum` and `variance` to `SparseOutputEntry`
- [x] Add `use_momentum`, `momentum_beta1`, `momentum_beta2`, `momentum_eps` to `Config`
- [x] Implement conditional Adam-like update in `token_sparse_output.cpp`
- [x] Register params in C API
- [x] Bump checkpoint magic `SBMCKPT6` → `SBMCKPT7`
- [x] Write test `verify_momentum_learning`

**Current:** `token_sparse_output.cpp:607-608`:
```cpp
entry.logit = (1-logit_decay)*entry.logit + rate*(target - probability);
```

**Change:**
- Add per-entry `momentum` and `variance` buffers
- Replace update with Adam-like rule when `use_momentum` is true:
  ```cpp
  float grad = target_right - probability_right;
  entry.momentum = beta1 * entry.momentum + (1-beta1) * grad;
  entry.variance = beta2 * entry.variance + (1-beta2) * grad * grad;
  float adaptive_lr = rate / (sqrt(entry.variance) + eps);
  entry.logit += adaptive_lr * entry.momentum;
  ```

**Files:**
- Modify: `include/sbm/detail/sparse_output.hpp`
- Modify: `include/sbm/types.hpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/api/c_api.cpp`
- Modify: `src/modules/machine_checkpoint.cpp`
- Modify: `tests/test_scaling.cpp`

**Status:** Implemented in commit `8094de0`.

**Constraint compliance (§4.1):** Pure local per-entry update, no global backprop. Each sparse entry maintains independent momentum.

**Memory cost:** +8 bytes per sparse entry.

**Expected:** +0.1-0.2 nats

#### U2.2: Counterfactual Routing Credit

**Current:** `apply_trace_credit()` in `machine_learning.cpp:59-75` assigns credit only to selected nodes.

**Change:**
- Record high-scoring but unselected candidates during `select_route()`
- Assign negative credit to nodes that "should have been selected but were not"
- This addresses DESIGN_NOTES.md:37 ("Counterfactual credit for nodes that should have been selected but were not")

**Files:**
- Modify: `machine_routing.cpp`: record near-miss candidates in `select_route()`
- Modify: `machine_learning.cpp`: `apply_trace_credit()` applies negative gradient to near-misses

**Expected:** +0.03-0.08 nats

### Phase U3: Deeper Context Modeling

**Hypothesis:** Multi-hop content following and state-dependent routing capture dependencies that positional programs cannot.

#### U3.1: Multi-Hop Content Following

**Current:** `ContentFollow(pattern_lag, max_lag)` does single-hop matching.

**Change:**
- Add `ContentFollowMulti(pattern_lag, max_lag, hops)`:
  - Hop 1: match current token + pattern_lag, find historical occurrence
  - Hop 2: from successor position, match again
  - Support 2-3 hops maximum
- `machine_topology.cpp`: propose multi-hop variants

**Files:**
- Modify: `address_interpreter.cpp`: multi-hop matching
- Modify: `machine_topology.cpp`: proposal logic for multi-hop programs
- Modify: `types.hpp`: AddressOp enum extension

**Constraint compliance (§4.3):** Generic semantics (content matching, no labels), local proposal (extends existing ContentFollow), bounded search (max_lag per hop).

**Expected:** +0.05-0.1 nats

#### U3.2: State-Dependent Routing Signal

**Current:** `score()` in `machine_routing.cpp:110-123` depends on address signatures (Hamming similarity).

**Change:**
- Add routing term: match between current candidate's output state and previous active set's output state
- Makes execution paths "content- and state-dependent" per THEORY_ALIGNMENT.md §1

**Files:**
- Modify: `machine_routing.cpp`: extend `score()` with output-state term
- Modify: `token_sparse_output.cpp`: pass previous output state to routing

**Constraint compliance (§4.1):** Routing signal bounded by previous active set size, not total nodes.

**Expected:** +0.02-0.05 nats

## Phase U4: Scaling and Validation

### U4.1: 10M FineWeb-Edu Validation

- [x] Establish r3-baseline on 10M/1M: **6.3412 nats/token**
- [x] Validate momentum-only (3 seeds): **6.2305 ± 0.0001 nats/token**
- [x] Validate upgrade-v1-adaptive (3 seeds): **6.2559 ± 0.0002 nats/token**
- [x] Document results in `research_results/adaptive_computation_upgrade_10m_20260630.md`

### Key Findings

| Configuration | eval NLL | Topology accepted | Bytes | Tok/s |
|---|---:|---:|---:|---:|
| r3-baseline | 6.3412 | 5 | 2.16 GB | 15,756 |
| **upgrade-v1** | **6.2305** | 1 | 895 MB | 17,000 |
| upgrade-v1-adaptive | 6.2559 | 1 | 896 MB | 12,444 |

- **Adam bias correction is required.** Without it, momentum suppresses topology growth and degrades NLL to 6.414.
- **Momentum is the dominant improvement** (-0.111 nats, 59% smaller, 8% faster).
- **Adaptive beam width + refinement are neutral/slightly negative** in the current configuration.

### U4.2: Preset Update

Based on validation, `upgrade-v1` preset was simplified to momentum on top of r3-baseline. The full adaptive configuration was moved to `upgrade-v1-adaptive` for further tuning.

### U4.3: Next Step

Run `upgrade-v1` on the 100M heterogeneous stream gate (R3) to test transfer and scaling.

---

## Expected Cumulative Impact

| Phase | Expected Improvement | Actual (10M) |
|---|---|---|
| Baseline | — | 6.341 |
| U2.1 (momentum) | +0.1-0.2 | **+0.111** |
| U1.1 + U1.2 (adaptive) | +0.05-0.15 | -0.026 (when combined) |
| **upgrade-v1** | — | **6.231** |


## Implementation Order and Dependencies

```
Phase 0 (no dependencies) ──┐
                             ├─→ Phase U1 (Adaptive Computation) ──→ Phase U3 (Deeper Context)
Phase U2 (no dependencies) ─┘                                                 │
                                                                              ↓
                                                                    Phase U4 (Scaling Validation)
```

U1 and U2 can be developed in parallel. U3 depends on U1 infrastructure. U4 depends on all prior phases.

## Constraints Compliance Checklist

| Constraint | U1.1 | U1.2 | U2.1 | U2.2 | U3.1 | U3.2 |
|---|---|---|---|---|---|---|
| §4.1 No dense global layers | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| §4.1 No full-network backprop | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| §4.2 Bounded active work | ✅ avg O(1) | ✅ bounded rounds | ✅ local | ✅ local | ✅ bounded hops | ✅ bounded signal |
| §4.3 Generic semantics | — | — | — | — | ✅ content match | ✅ state match |
| §4.5 Causal ordering | ✅ | ✅ before target | ✅ | ✅ | ✅ | ✅ |
| §4.6 Strict eval freeze | ✅ | ✅ no param update | ✅ | ✅ | ✅ | ✅ |

## Self-Review

- Spec coverage: covers adaptive computation, stronger optimization, deeper context, scaling validation
- Placeholder scan: every task has concrete files and expected outcomes
- Type consistency: new parameters follow existing Config struct patterns
- Constraint compliance: all upgrades pass §4.1-4.6 checks
