# Bounded Core Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make address lookup latency independent of stored-node count, remove vocabulary-sized output structures, bound local output state, and remove full-vocabulary baseline scans.

**Architecture:** Preserve address programs, persistent nodes, bounded sparse routing and hierarchical softmax. Replace dense address directories with lazy occupied-bucket states, materialized output trees with a keyed implicit tree, unbounded local decision vectors with bounded sorted storage, and dense baseline distributions with direct sparse probability queries.

**Tech Stack:** C++20 shared libraries, CMake/CTest, stable C ABI, Python ctypes smoke tests.

---

## File structure

- Create `include/sbm/detail/implicit_output.hpp`: constant-storage token/rank permutation and implicit path traversal.
- Create `src/modules/implicit_output.cpp`: keyed affine permutation and interval-tree implementation.
- Create `tests/test_scaling.cpp`: deterministic structural scaling gates and bounded-work assertions.
- Modify `include/sbm/machine.hpp`: lazy bucket state, implicit output object and bounded sparse entry helpers.
- Modify `include/sbm/types.hpp`: runtime limits and scaling diagnostics.
- Modify `src/modules/machine_storage.cpp`: lazy index initialization and exact byte accounting.
- Modify `src/modules/machine_routing.cpp`: bounded bucket sampling only.
- Modify `src/modules/machine_learning.cpp`: bounded specialization lookup.
- Modify `src/modules/machine_maintenance.cpp`: lazy index rebuild and diagnostics.
- Modify `src/modules/token_sparse_output.cpp`: implicit paths/decoding and bounded decision lookup.
- Modify `src/modules/token_experiment.cpp`: sparse direct baseline NLL.
- Modify `src/api/c_api.cpp`, `python/sbm_runtime.py`: expose new parameters and diagnostics through existing JSON paths.
- Modify `CMakeLists.txt`: compile new module and scaling tests.
- Modify `DESIGN_NOTES.md`, `API.md`, `RESEARCH_LOG.md`: accepted invariants and measured results.

### Task 1: Reproduce and instrument scaling failures

**Files:**
- Create: `tests/test_scaling.cpp`
- Modify: `include/sbm/types.hpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing structural scaling test**

Create configurations with `bucket_bits` 10, 16 and 20 and token vocabularies
4,096, 50,257 and 250,000. Assert diagnostics expose:

```cpp
assert(high_bits.address_index_bytes <= low_bits.address_index_bytes * 105 / 100 + 4096);
assert(large_vocab.output_structure_bytes <=
       small_vocab.output_structure_bytes + 1024ULL * 1024ULL);
assert(result.diagnostics.max_bucket_candidates_inspected <=
       config.max_address_channels * config.bucket_scan_limit);
```

The test must also prefill colliding distractors and verify one token step never
reports a bucket inspection above the analytical budget.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
cmake -S . -B build-fast -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-fast --target sbm_scaling_tests -j2
ctest --test-dir build-fast -R sbm_scaling --output-on-failure
```

Expected: compilation fails because the diagnostic fields do not exist; after
adding fields initialized from current storage, memory assertions fail.

- [ ] **Step 3: Add measurement-only diagnostics**

Add to `Diagnostics`:

```cpp
std::uint64_t address_index_bytes{};
std::uint64_t output_structure_bytes{};
std::uint64_t max_bucket_candidates_inspected{};
std::uint64_t max_sparse_entries_per_node{};
```

Count allocated bucket containers/capacities, output tree/path capacities and
the maximum actual bucket-loop iterations. Do not alter routing or output yet.

- [ ] **Step 4: Verify the test now fails on the intended growth assertions**

Run the scaling test and record current byte counts in its failure output.

- [ ] **Step 5: Commit instrumentation**

```powershell
git add CMakeLists.txt include/sbm/types.hpp src/modules/machine_maintenance.cpp tests/test_scaling.cpp
git commit -m "test: expose address and output scaling failures"
```

### Task 2: Lazy bounded address buckets

**Files:**
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_storage.cpp`
- Modify: `src/modules/machine_routing.cpp`
- Modify: `src/modules/machine_learning.cpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Test: `tests/test_scaling.cpp`

- [ ] **Step 1: Add failing behavior assertions**

Extend the scaling test to prefill at least 100,000 nodes into colliding buckets,
then assert fixed candidate counts, exact resident counts after prune/rebuild,
and deterministic routes from two same-seed machines.

- [ ] **Step 2: Verify RED against full specialization scans**

Run `sbm_scaling`; expected failure is
`max_bucket_candidates_inspected > analytical_budget`.

- [ ] **Step 3: Introduce the lazy bucket state**

Replace parallel dense arrays with:

```cpp
struct BucketState {
    std::vector<NodeId> residents;
    std::vector<NodeId> hot;
    std::uint64_t last_split_step{};
    std::uint32_t cold_cursor{};
};
std::unordered_map<std::size_t, BucketState> bucket_directory_;
```

Add `find_bucket`, `ensure_bucket`, `index_node`, `unindex_node` and
`bounded_bucket_candidates`. No per-token caller may iterate `residents`
directly.

- [ ] **Step 4: Bound creation and routing**

Read resident count from `residents.size()`. Search at most
`bucket_scan_limit` IDs, hot first and then a deterministic wraparound cold
window. Keep edge and neighboring-region hard limits unchanged.

- [ ] **Step 5: Rebuild and removal correctness**

Rebuild the directory only in explicit maintenance. Swap removal must remove the
logical ID from its bucket before physical slot movement. Stale IDs remain
defensively skipped but may not affect resident counts.

- [ ] **Step 6: Run correctness and scaling tests**

```powershell
cmake --build build-fast --target sbm_machine sbm_tests sbm_scaling_tests -j2
ctest --test-dir build-fast -R "sbm_(cpp_api|scaling)" --output-on-failure
```

Expected: both pass; bucket-bit memory and candidate bounds satisfy Task 1.

- [ ] **Step 7: Commit**

```powershell
git add include/sbm/machine.hpp src/modules/machine_storage.cpp src/modules/machine_routing.cpp src/modules/machine_learning.cpp src/modules/machine_maintenance.cpp tests/test_scaling.cpp
git commit -m "perf: bound sparse address lookup"
```

### Task 3: Constant-storage implicit output tree

**Files:**
- Create: `include/sbm/detail/implicit_output.hpp`
- Create: `src/modules/implicit_output.cpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_storage.cpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/test_scaling.cpp`

- [ ] **Step 1: Write failing permutation/path tests**

For vocabularies 2, 3, 4,096, 50,257 and 250,000, assert every sampled token
round-trips through `rank`/`token`, every path terminates at its rank, decision
IDs are below `V-1`, and object storage is independent of `V`.

- [ ] **Step 2: Verify RED**

Build `sbm_scaling_tests`; expected failure is missing `ImplicitOutputTree`.

- [ ] **Step 3: Implement keyed affine permutation**

`ImplicitOutputTree(vocab, seed)` chooses `a` until `gcd(a,V)==1`, derives `b`,
and computes the modular inverse with extended Euclid. Products use `uint64_t`.

- [ ] **Step 4: Implement interval paths and decoding state**

Expose:

```cpp
struct ImplicitDecision { std::uint32_t id; bool right; };
void target_path(std::uint32_t token, std::vector<ImplicitDecision>& out) const;
ImplicitSplit split(std::uint32_t lo, std::uint32_t hi) const;
std::uint32_t token_from_rank(std::uint32_t rank) const noexcept;
```

Use `decision_id = mid - 1` and never allocate per-vocabulary arrays.

- [ ] **Step 5: Replace materialized tree use**

Delete `output_tree_`, `token_path_offsets_`, `token_path_steps_` and
`build_output_tree`. Generate target decisions into a reusable scratch vector;
beam items carry `[lo,hi)` rather than stored child references.

- [ ] **Step 6: Verify quality fixture and storage gate**

Run all C++ tests and scaling tests. The fixed synthetic token run must remain
finite and strict-freeze clean; output structure bytes must pass the vocabulary
gate.

- [ ] **Step 7: Commit**

```powershell
git add CMakeLists.txt include/sbm/detail/implicit_output.hpp src/modules/implicit_output.cpp include/sbm/machine.hpp src/modules/machine_storage.cpp src/modules/token_sparse_output.cpp tests/test_scaling.cpp
git commit -m "perf: make hierarchical output tree implicit"
```

### Task 4: Bound per-node sparse decisions

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/api/c_api.cpp`
- Test: `tests/test_scaling.cpp`
- Test: `tests/test_c_api.cpp`

- [ ] **Step 1: Add RED tests for capacity and schema**

Set `max_sparse_decisions_per_node=8`, train one broad node across many target
paths, and assert diagnostics never exceed 8. Assert the parameter registry
contains the new bounded integer with range 8..4096.

- [ ] **Step 2: Verify RED**

Expected: config/schema field missing.

- [ ] **Step 3: Implement sorted lookup and deterministic replacement**

Binary-search entries by decision ID. Insert in order while under capacity. At
capacity, replace the entry with smallest absolute logit, breaking ties by
largest decision ID, then restore sorted order.

- [ ] **Step 4: Run capacity ablation**

Use the automated three-seed runner for capacities 64, 128 and 256 plus the
unbounded current control. Select the smallest capacity within 1% held-out NLL
of control; otherwise keep the stage experimental and do not change the default.

- [ ] **Step 5: Commit accepted behavior and result**

```powershell
git add include/sbm/types.hpp include/sbm/machine.hpp src/modules/token_sparse_output.cpp src/api/c_api.cpp tests/test_scaling.cpp tests/test_c_api.cpp RESEARCH_LOG.md
git commit -m "perf: bound local sparse output state"
```

### Task 5: Remove dense baseline ranking from synchronous evaluation

**Files:**
- Modify: `src/modules/token_experiment.cpp`
- Modify: `include/sbm/experiment.hpp`
- Modify: `src/api/c_api.cpp`
- Test: `tests/test_sbm.cpp`
- Test: `tests/test_scaling.cpp`

- [ ] **Step 1: Write formula-equivalence RED tests**

On a fixed small corpus, capture old unigram/current/pair NLL and require the
new direct target-probability accumulators to match within `1e-7`. Add a 50,257
vocabulary test asserting synchronous baseline candidate work is independent of
`V`.

- [ ] **Step 2: Verify RED**

Expected: baseline work diagnostic scales with vocabulary.

- [ ] **Step 3: Implement sparse target-probability queries**

Add `ConditionalTable::target_probability(key,target,fallback_probability)` and
an accumulator that accepts one probability without materializing a vector.
Retain exact NLL/bits/perplexity. Mark baseline ranking metrics unavailable in
the synchronous result rather than fabricating zero accuracy.

- [ ] **Step 4: Add named interpolated multiscale baseline**

Compute a normalized convex interpolation of current/pair/lag2/lag4 target
probabilities with fixed documented weights. Emit it under a new JSON name and
retain the old geometric control only in an offline comparison script until its
quality relation is recorded.

- [ ] **Step 5: Verify real-corpus wall time**

Run the existing 10,240/10,240 FineWeb-Edu compatibility smoke. Require
synchronous baseline time no greater than twice measured model time and record
model and baseline timing separately.

- [ ] **Step 6: Commit**

```powershell
git add include/sbm/experiment.hpp src/modules/token_experiment.cpp src/api/c_api.cpp tests/test_sbm.cpp tests/test_scaling.cpp RESEARCH_LOG.md API.md
git commit -m "perf: evaluate language baselines without vocabulary scans"
```

### Task 6: Final core verification and documentation

**Files:**
- Modify: `DESIGN_NOTES.md`
- Modify: `API.md`
- Modify: `RESEARCH_LOG.md`
- Modify: `ROADMAP_REAL_DATA.md`

- [ ] **Step 1: Run full verification**

```powershell
Get-ChildItem python\*.py,scripts\*.py,tests\*.py | ForEach-Object {
  python -m py_compile $_.FullName
}
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests sbm_scaling_tests -j2
ctest --test-dir build-fast --output-on-failure
git diff --check
```

- [ ] **Step 2: Run structural scaling benchmark**

Record address bytes, output bytes, candidates, p50/p95 latency, model bytes and
train/eval throughput for every acceptance point in the design spec.

- [ ] **Step 3: Run three-seed quality gate**

Compare the unchanged pre-change commit and final commit on identical token
streams. Reject any stage exceeding the declared 1% NLL tolerance.

- [ ] **Step 4: Update canonical documents and commit**

```powershell
git add DESIGN_NOTES.md API.md RESEARCH_LOG.md ROADMAP_REAL_DATA.md
git commit -m "docs: record bounded core scaling results"
```
