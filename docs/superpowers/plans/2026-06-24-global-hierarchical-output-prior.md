# Global Hierarchical Output Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact corpus-wide hierarchical count prior so sparse local outputs learn residuals and multi-seed experiments share one fixed output decomposition.

**Architecture:** The implicit output tree remains constant-storage and keyed by a new `output_tree_seed`. Two global count arrays indexed by implicit decision ID provide a Jeffreys-smoothed base logit. Target scoring, beam decoding and counterfactuals use base plus bounded local residuals; counts update only after prediction when learning is enabled.

**Tech Stack:** C++20, existing C ABI parameter registry, Python ctypes smoke tests, CTest.

---

### Task 1: Decouple the output-tree seed

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/experiment.hpp`
- Modify: `src/api/c_api.cpp`
- Modify: `src/modules/machine_storage.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `tests/test_c_api.cpp`
- Modify: `tests/test_python_api.py`
- Modify: `tests/test_scaling.cpp`

- [x] **Step 1: Write failing registry and behavior tests**

Add assertions that the parameter schema contains `output_tree_seed`, Python can
set it, and two sparse machines with different `seed` but the same
`output_tree_seed` return identical first-step target probabilities for a
power-of-two vocabulary.

- [x] **Step 2: Verify RED**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_c_api_tests sbm_scaling_tests sbm_api
ctest --test-dir build-fast -C Release -R "sbm_(c_api|scaling_implicit_output|python_api)" --output-on-failure
```

Expected: compilation or schema assertions fail because `output_tree_seed` does
not exist.

- [x] **Step 3: Implement the additive parameter**

Add `std::uint64_t output_tree_seed{7}` to `Config`, register it as a tunable
`uint64`, handle set/get JSON, construct `ImplicitOutputTree` from it, and emit
it in token experiment JSON. Do not change `rng_state_(config.seed)`.

- [x] **Step 4: Verify GREEN and commit**

Run the Task 1 test command, then:

```powershell
git add include/sbm/types.hpp include/sbm/experiment.hpp src/api/c_api.cpp src/modules/machine_storage.cpp src/modules/token_experiment.cpp tests/test_c_api.cpp tests/test_python_api.py tests/test_scaling.cpp
git commit -m "fix: decouple output tree from model seed"
```

### Task 2: Add the global count prior

**Files:**
- Modify: `include/sbm/machine.hpp`
- Modify: `include/sbm/types.hpp`
- Modify: `src/modules/machine_storage.cpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Modify: `src/modules/experiment.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_scaling.cpp`

- [x] **Step 1: Write failing prior tests**

Add a `prior` scaling-test mode that verifies:

```cpp
// V=4, no local residual updates (exact_region_mass=0).
// First observation is scored before its count update.
assert(std::abs(first.target_probability - 0.25F) < 1e-6F);
// Frozen calls are repeatable and do not change prior update count.
assert(before.global_output_prior_updates == after.global_output_prior_updates);
assert(after.global_output_prior_bytes == 2U * (vocabulary - 1U) * sizeof(std::uint64_t));
```

Train an imbalanced fixture and compare frozen path loss to counts computed in
the test with the same `0.5` branch pseudocount.

Register the mode as:

```cmake
add_test(NAME sbm_scaling_prior COMMAND sbm_scaling_tests prior)
```

- [x] **Step 2: Verify RED**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_scaling_tests
ctest --test-dir build-fast -C Release -R sbm_scaling_prior --output-on-failure
```

Expected: diagnostics fields and global prior are missing.

- [x] **Step 3: Implement count storage and helpers**

Add two `std::vector<std::uint64_t>` members sized `V-1`:

```cpp
std::vector<std::uint64_t> global_output_total_;
std::vector<std::uint64_t> global_output_right_;
std::uint64_t global_output_prior_updates_{};
```

Add helpers:

```cpp
float global_output_logit(std::uint32_t decision) const noexcept;
void observe_global_output_path(std::span<const detail::ImplicitDecision> path);
```

Compute `log((right + 0.5) / (left + 0.5))`. Update counts with overflow checks
after all target-dependent metrics and credits are fixed.

- [x] **Step 4: Use combined logits everywhere**

In target path scoring and beam expansion, replace local-only logits with:

```cpp
global_output_logit(decision) + aggregate_sparse_logit(active, decision)
```

Keep counterfactual removal local-only so the prior remains present. Local SGD
uses the combined branch probability and therefore learns a residual.

- [x] **Step 5: Add diagnostics and memory accounting**

Add `global_output_prior_bytes` and `global_output_prior_updates` to
`Diagnostics`, C++ JSON and token JSON. Include count vector capacities in
`estimated_bytes` but not `output_structure_bytes`.

- [x] **Step 6: Verify GREEN and commit**

Run all scaling, C++, C and Python tests, then:

```powershell
git add CMakeLists.txt include/sbm/machine.hpp include/sbm/types.hpp src/modules/machine_storage.cpp src/modules/token_sparse_output.cpp src/modules/machine_maintenance.cpp src/modules/experiment.cpp src/modules/token_experiment.cpp tests/test_scaling.cpp
git commit -m "feat: add global hierarchical output prior"
```

### Task 3: Correctness and corpus gates

**Files:**
- Modify: `ISSUES.md`
- Modify: `DESIGN_NOTES.md`
- Modify: `API.md`
- Modify: `RESEARCH_LOG.md`

- [ ] **Step 1: Run full static and unit verification**

```powershell
Get-ChildItem python\*.py,scripts\*.py,tests\*.py | ForEach-Object { python -m py_compile $_.FullName }
cmake --build build-fast --config Release
ctest --test-dir build-fast -C Release --output-on-failure
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 2: Run fixed smoke controls**

Run three controls on the existing 10,240/10,240 compatibility smoke:

1. prior-only (`exact_region_mass=0`);
2. full fixed-topology model with global prior;
3. the recorded pre-change commit result.

Record model NLL, unigram NLL, throughput, model bytes and prior bytes. Reject
the implementation if prior-only is materially worse than the computed
hierarchical unigram estimator or frozen evaluation mutates counts.

- [ ] **Step 3: Run the medium gate**

Only after Step 2 passes, execute seeds `7,11,19` on
`E:\SPM_DATA\naime_compat_medium\manifest.json` with fixed
`output_tree_seed=7`. Compare mean/stdev NLL against the recorded 10.6696 model
and 7.6826 unigram controls. Keep the change if it closes the unigram gap without
unbounded active work; otherwise record the negative result before deciding
rollback.

- [ ] **Step 4: Update canonical documents and issue status**

Document the global/local logit decomposition and API parameter. Mark P0-1 and
P0-4 resolved only if all correctness gates pass; experimental quality failure
does not invalidate a semantically correct prior but must remain a blocker.

- [ ] **Step 5: Commit the verified result**

```powershell
git add ISSUES.md DESIGN_NOTES.md API.md RESEARCH_LOG.md docs/superpowers/plans/2026-06-24-global-hierarchical-output-prior.md
git commit -m "research: validate global output prior"
```
