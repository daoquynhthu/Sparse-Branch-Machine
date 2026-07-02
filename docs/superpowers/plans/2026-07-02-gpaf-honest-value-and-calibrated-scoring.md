# GPAF Honest Value Accounting and Calibrated Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the measurement and lifecycle defects identified in
`docs/superpowers/research/2026-07-02-gpaf-architecture-investigation.md`
(Proposal A: honest attribution + live value plumbing; Proposal D: GPAF-specific
scoring) so that GPAF's reported value is causally honest and its Probe→Active
gate responds to real evidence instead of being either a no-op or a permanent
block.

**Architecture:** This continues
`docs/superpowers/plans/2026-07-01-gpaf-causal-attribution-and-costed-admission.md`
(Tasks 1-4 of that plan, already complete, built the unique/overlap diagnostic
surface and the *disabled-by-default* costed gate). This plan fixes three
remaining defects in that surface — a units bug in the cross-run aggregation,
an over-attribution bug in the per-step causal ablation, and a dead map that
the costed gate reads but nothing ever writes — then adds calibrated,
value-gated GPAF scoring on top of the corrected signal. No dense layers, no
unbounded work, no leakage, no linguistic labels; the token-signature system
is untouched throughout.

**Tech Stack:** C++20 (`sbm_machine`, `sbm_experiment`, `sbm_api`), CTest,
Windows/MinGW build in `build-fast` (already configured — `cmake --build
build-fast --target <t> -j4`, `ctest --test-dir build-fast`).

---

## File structure

- Modify `src/modules/token_experiment.cpp`: normalize GPAF cost/net-value
  aggregation to per-example means (Task 1).
- Modify `src/modules/token_sparse_output.cpp`: stop crediting GPAF-overlap
  node logits as causal GPAF ablation value (Task 2).
- Modify `include/sbm/types.hpp`: add `gpaf_value_ema_decay`,
  `gpaf_execution_cost_weight` Config fields (Task 3).
- Modify `src/modules/machine_topology.cpp`: implement the
  `gpaf_slot_costed_net_value_` writer and value-driven Active→Quarantined
  demotion (Task 3).
- Modify `src/modules/machine_checkpoint.cpp`: persist the two new Config
  fields (Task 3).
- Modify `src/api/c_api.cpp`: expose the two new Config fields via schema,
  setter and JSON emission (Task 3).
- Modify `src/modules/machine_routing.cpp`: calibrated GPAF-unique candidate
  scoring, conservative admission (Task 4).
- Modify `include/sbm/machine.hpp`: declare any new private helper needed for
  Task 4 scoring.
- Modify `tests/test_scaling.cpp`, `tests/test_c_api.cpp`: update/extend
  assertions for corrected semantics.
- Update `DESIGN_NOTES.md`, `PROGRESS.md`, `RESEARCH_LOG.md` after
  implementation evidence exists (Task 5).

## Task 1: Fix GPAF cost/net-value units bug in cross-run aggregation

**Root cause (RC6 in the investigation doc):** in
`src/modules/token_experiment.cpp`, `gpaf_ablation_mean_gain` and
`gpaf_*_mean_removed_cross_entropy` are correctly normalized to per-example
means (`sum * inverse`), but `gpaf_ablation_false_positive_cost`,
`gpaf_unique/overlap_ablation_false_positive_cost`, and
`gpaf_ablation_key_execution_cost`/`gpaf_ablation_key_net_value` are raw sums
accumulated over the whole eval run with no normalization. Over 62,471 eval
examples with `gpaf_residents_per_slot≈4`, this produces reported execution
costs like 224,308 against gains of ~0.06 nats — a spurious 6-order-of-
magnitude mismatch that makes every costed net value negative regardless of
whether the role key is useful.

**Files:**
- Modify: `src/modules/token_experiment.cpp:891-930`
- Test: `tests/test_c_api.cpp`

- [ ] **Step 1: Confirm current (buggy) values with a build**

Run:

```bash
cmake --build build-fast --target sbm_c_api_tests -j4
./build-fast/sbm_c_api_tests.exe
```

Expected: PASS (existing test only checks JSON keys exist, not their
values — this step is a baseline sanity check, not a red test).

- [ ] **Step 2: Fix the aggregation**

In `src/modules/token_experiment.cpp`, replace lines 891-930 (from
`result.gpaf_ablation_key_execution_cost = gpaf_ablation_key_execution_cost_sum;`
through the end of the per-key normalization loop) with:

```cpp
    result.gpaf_ablation_key_resident_count = gpaf_ablation_key_resident_count;
    result.gpaf_ablation_key_reuse_count = gpaf_ablation_key_reuse_count;
    if (gpaf_ablation_examples != 0U) {
        const double inverse = 1.0 / static_cast<double>(gpaf_ablation_examples);
        result.gpaf_ablation_mean_removed_cross_entropy =
            gpaf_ablation_removed_cross_entropy_sum * inverse;
        result.gpaf_ablation_mean_gain = gpaf_ablation_gain_sum * inverse;
        result.gpaf_ablation_false_positive_cost =
            gpaf_ablation_false_positive_cost_sum * inverse;
    }
    if (gpaf_unique_ablation_examples != 0U) {
        const double inverse =
            1.0 / static_cast<double>(gpaf_unique_ablation_examples);
        result.gpaf_unique_ablation_mean_removed_cross_entropy =
            gpaf_unique_ablation_removed_cross_entropy_sum * inverse;
        result.gpaf_unique_ablation_mean_gain =
            gpaf_unique_ablation_gain_sum * inverse;
        result.gpaf_unique_ablation_false_positive_cost =
            gpaf_unique_ablation_false_positive_cost_sum * inverse;
    }
    if (gpaf_overlap_ablation_examples != 0U) {
        const double inverse =
            1.0 / static_cast<double>(gpaf_overlap_ablation_examples);
        result.gpaf_overlap_ablation_mean_removed_cross_entropy =
            gpaf_overlap_ablation_removed_cross_entropy_sum * inverse;
        result.gpaf_overlap_ablation_mean_gain =
            gpaf_overlap_ablation_gain_sum * inverse;
        result.gpaf_overlap_ablation_false_positive_cost =
            gpaf_overlap_ablation_false_positive_cost_sum * inverse;
    }
    for (std::size_t i = 0; i < gpaf_ablation_key_count; ++i) {
        if (gpaf_ablation_key_examples[i] == 0U) continue;
        const double inverse =
            1.0 / static_cast<double>(gpaf_ablation_key_examples[i]);
        result.gpaf_ablation_key_mean_removed_cross_entropy[i] =
            gpaf_ablation_key_removed_cross_entropy_sum[i] * inverse;
        result.gpaf_ablation_key_mean_gain[i] =
            gpaf_ablation_key_gain_sum[i] * inverse;
        result.gpaf_ablation_key_false_positive_cost[i] =
            gpaf_ablation_key_false_positive_cost_sum[i] * inverse;
        result.gpaf_ablation_key_execution_cost[i] =
            gpaf_ablation_key_execution_cost_sum[i] * inverse;
        result.gpaf_ablation_key_net_value[i] =
            gpaf_ablation_key_net_value_sum[i] * inverse;
    }
```

This is the minimal diff: every `*_sum` field that feeds a reported
value now goes through the same `* inverse` normalization that
`gpaf_ablation_mean_gain` already used, so all reported GPAF cost/gain/value
fields share nats-per-example units. Nothing in `types.hpp`,
`machine.hpp`, `token_sparse_output.cpp` or `machine_topology.cpp` changes
in this task — the per-step computation (`StepStats`) was already internally
consistent; only the cross-run aggregation was broken.

- [ ] **Step 3: Rebuild and verify no regression**

```bash
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests sbm_scaling_tests -j4
ctest --test-dir build-fast --output-on-failure -R "sbm_c_api|sbm_cpp_api|sbm_scaling_gpaf"
```

Expected: PASS (field existence and finiteness/sign assertions are
unaffected by this change; only magnitudes shrink into a sane nats-per-example
range).

- [ ] **Step 4: Commit**

```bash
git add src/modules/token_experiment.cpp docs/superpowers/plans/2026-07-02-gpaf-honest-value-and-calibrated-scoring.md
git commit -m "fix: normalize GPAF cost and net-value aggregation to per-example means"
```

## Task 2: Stop crediting GPAF-overlap logits as causal GPAF ablation value

**Root cause (RC5 in the investigation doc, precise form):** in
`src/modules/token_sparse_output.cpp`, the frozen-eval ablation loop computes
an aggregate `removed` quantity by subtracting the logits of **every**
GPAF-touched active node — both nodes GPAF uniquely introduced and nodes GPAF
also touched that were already reachable via exact bucket, control edge or
neighbor probe (`gpaf_overlap == true`). Removing GPAF cannot make an
overlap node disappear from the route (it would still be selected via its
original source), so subtracting its logit overstates the loss increase
attributed to GPAF. This is the mechanism behind the reported +0.2288
nats/example "overlap gain" — it is a measurement artifact, not a causal
GPAF effect. The existing `removed_unique`/`gpaf_unique_codelength_gain`
computation is already correct; this task makes the *primary/aggregate*
metric equal to it, and does the same for the per-key breakdown, while
keeping the overlap-only figures as an explicitly separate, non-causal
diagnostic.

**Files:**
- Modify: `src/modules/token_sparse_output.cpp:361-409`
- Test: `tests/test_scaling.cpp`

- [ ] **Step 1: Read current test expectations**

`tests/test_scaling.cpp:658-684` (`verify_gpaf_frozen_retrieval_is_read_only`)
already only checks finiteness, non-negativity of costs, and
`net_value <= gain + 1e-6` — no test currently asserts a specific relationship
between the aggregate and unique metrics, so this fix does not require a
pre-emptive failing test; the existing test is the regression guard. Run it
before editing to confirm the current passing baseline:

```bash
cmake --build build-fast --target sbm_scaling_tests -j4
./build-fast/sbm_scaling_tests.exe gpaf_frozen
```

Expected: PASS ("GPAF frozen retrieval read-only passed").

- [ ] **Step 2: Fix the ablation loop**

In `src/modules/token_sparse_output.cpp`, replace the loop body from
`for (const auto& step : token_path_scratch_) {` (starting at line 364)
through the line before `gpaf_codelength_gain = gpaf_removed_cross_entropy - cross_entropy;`
(i.e. replace lines 364-409) with:

```cpp
            for (const auto& step : token_path_scratch_) {
                float removed_unique = 0.0F;
                float removed_overlap = 0.0F;
                std::array<float, kMaxGpafAblationKeys> removed_by_key{};
                for (const auto& node : active) {
                    if (node.source != CandidateSource::GpafRole ||
                        node.gpaf_key == UINT64_MAX) {
                        continue;
                    }
                    const auto slot = slot_of(node.id);
                    if (slot == SIZE_MAX) continue;
                    const float node_logit =
                        node.responsibility * sparse_logit(slot, step.id);
                    if (node.gpaf_overlap) {
                        // Overlap nodes remain reachable via their original
                        // local source (exact bucket, control edge or
                        // neighbor probe) even if GPAF never existed. Their
                        // logits are not causally attributable to GPAF and
                        // must not be subtracted when computing GPAF's
                        // codelength value; tracked as a separate,
                        // non-causal attribution diagnostic only.
                        removed_overlap += node_logit;
                        continue;
                    }
                    removed_unique += node_logit;
                    for (std::size_t i = 0; i < gpaf_ablation_key_count; ++i) {
                        if (gpaf_ablation_keys[i] == node.gpaf_key) {
                            removed_by_key[i] += node_logit;
                            break;
                        }
                    }
                }
                // The primary/aggregate GPAF ablation metric equals the
                // unique-only removal: the honest causal estimate of "what
                // happens if GPAF is removed", since overlap-sourced nodes
                // would still be selected without GPAF.
                gpaf_removed_cross_entropy += branch_loss(
                    (path_logits[path_position] - removed_unique) / temperature,
                    step.right);
                if (gpaf_unique_ablation_nodes != 0U) {
                    gpaf_unique_removed_cross_entropy += branch_loss(
                        (path_logits[path_position] - removed_unique) / temperature,
                        step.right);
                }
                if (gpaf_overlap_ablation_nodes != 0U) {
                    gpaf_overlap_removed_cross_entropy += branch_loss(
                        (path_logits[path_position] - removed_overlap) / temperature,
                        step.right);
                }
                for (std::size_t i = 0; i < gpaf_ablation_key_count; ++i) {
                    gpaf_ablation_key_removed_cross_entropy[i] += branch_loss(
                        (path_logits[path_position] - removed_by_key[i]) / temperature,
                        step.right);
                }
                ++path_position;
            }
```

Note `removed_by_key` now only accumulates unique-node logits, so
`gpaf_ablation_key_gain`/`gpaf_ablation_key_net_value` (computed further down,
unchanged) also become causally honest per role key — this closes the other
half of RC6 (per-key gain no longer includes overlap-inflated removal either).

The rest of the function (the `gpaf_codelength_gain = ...` block onward,
lines 410-443 in the original) is unchanged — it already derives aggregate
and per-key gain/false-positive-cost from `gpaf_removed_cross_entropy` /
`gpaf_ablation_key_removed_cross_entropy`, which are now correctly unique-only.

- [ ] **Step 3: Rebuild and verify**

```bash
cmake --build build-fast --target sbm_scaling_tests sbm_c_api_tests -j4
./build-fast/sbm_scaling_tests.exe gpaf_frozen
./build-fast/sbm_scaling_tests.exe gpaf_retrieval
./build-fast/sbm_scaling_tests.exe gpaf_costed_gate
./build-fast/sbm_scaling_tests.exe gpaf_lifecycle
./build-fast/sbm_c_api_tests.exe
```

Expected: PASS on all five. If `verify_gpaf_frozen_retrieval_is_read_only`
fails on `gpaf_unique_ablation_nodes > 0U || gpaf_overlap_ablation_nodes > 0U`
(line 674-675) because the single frozen step in that test happens to select
only overlap nodes, that assertion already tolerates either case (`||`) — no
change needed. If it fails on anything else, stop and report the exact
assertion before proceeding.

- [ ] **Step 4: Commit**

```bash
git add src/modules/token_sparse_output.cpp
git commit -m "fix: stop attributing GPAF-overlap logits to GPAF causal ablation value"
```

## Task 3: Live costed-value writer and evidence-based Probe/Active/Quarantine gating

**Root cause (RC3 in the investigation doc):** `gpaf_slot_costed_net_value_`
(`include/sbm/machine.hpp:275`) is read at
`src/modules/machine_topology.cpp:346` but never written anywhere in the
codebase. With `gpaf_active_requires_positive_net_value=true`,
`operator[]` default-inserts `0.0`, so `0.0 > 0.0` is always false and
promotion is permanently blocked — the gate has never actually gated on
evidence. There is also no demotion path: nothing ever moves an Active slot
back to Quarantined based on measured value.

Per-node counterfactual `contribution` (how much held-out-style loss would
increase if this node were removed from the current step's route) is already
computed for every active node before `observe_gpaf_shadow_roles` runs
(`src/modules/token_sparse_output.cpp:683-703` then `:719`) — this task wires
that existing value into the previously-dead map.

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `src/modules/machine_topology.cpp`
- Modify: `src/modules/machine_checkpoint.cpp`
- Modify: `src/api/c_api.cpp`
- Test: `tests/test_scaling.cpp`

- [ ] **Step 1: Add the two new Config fields**

In `include/sbm/types.hpp`, immediately after
`bool gpaf_active_requires_positive_net_value{false};` (currently line 328),
add:

```cpp
    float gpaf_value_ema_decay{0.98F};
    float gpaf_execution_cost_weight{0.0F};
```

- [ ] **Step 2: Persist the new fields in checkpoints**

In `src/modules/machine_checkpoint.cpp`, immediately after
`write_scalar(out, config.gpaf_active_requires_positive_net_value);`
(currently line 252), add:

```cpp
    write_scalar(out, config.gpaf_value_ema_decay);
    write_scalar(out, config.gpaf_execution_cost_weight);
```

Immediately after
`config.gpaf_active_requires_positive_net_value = read_scalar<bool>(in);`
(currently line 345), add, in the same order as the writes:

```cpp
    config.gpaf_value_ema_decay = read_scalar<float>(in);
    config.gpaf_execution_cost_weight = read_scalar<float>(in);
```

- [ ] **Step 3: Expose the new fields through the C API**

In `src/api/c_api.cpp`, find the schema table row for
`gpaf_active_requires_positive_net_value` (around line 316) and add two rows
immediately after it, following the same 9-tuple shape used by
`binding_reuse_value_weight` (a float row with real min/max):

```cpp
    {"gpaf_value_ema_decay", "float", "0.98", "0.0", "0.999999", "linear",
     true, false, false,
     "EMA decay for the live GPAF per-slot costed-value estimate; higher "
     "means slower-adapting evidence."},
    {"gpaf_execution_cost_weight", "float", "0.0", "0.0", "10.0", "linear",
     true, false, false,
     "Per-visit nats cost subtracted from a GPAF slot's live value estimate "
     "before the costed Active-admission gate."},
```

Find the `set_parameter` dispatcher's `SBM_SET_FLOAT` calls (around line 418,
next to `gpaf_active_requires_positive_net_value`'s `SBM_SET_BOOL`) and add:

```cpp
    SBM_SET_FLOAT(gpaf_value_ema_decay)
    SBM_SET_FLOAT(gpaf_execution_cost_weight)
```

Find the JSON config emission block (around line 540-541, next to
`gpaf_active_requires_positive_net_value`) and add:

```cpp
        << "  \"gpaf_value_ema_decay\": " << c.gpaf_value_ema_decay << ",\n"
        << "  \"gpaf_execution_cost_weight\": " << c.gpaf_execution_cost_weight << ",\n"
```

Find the `parameter_tasks` `"token-ce"` applicability chain (around line
549-573) and add both new field names to the `||` chain the same way
`gpaf_active_requires_positive_net_value` is listed there.

- [ ] **Step 4: Implement the writer and demotion in `observe_gpaf_shadow_roles`**

In `src/modules/machine_topology.cpp`, inside `observe_gpaf_shadow_roles`
(currently lines 316-356), replace this block:

```cpp
        const bool costed_gate_passes =
            !config_.gpaf_active_requires_positive_net_value ||
            gpaf_slot_costed_net_value_[key] > 0.0;
        if (phase->second == static_cast<std::uint8_t>(GpafSlotPhase::Probe) &&
            config_.gpaf_probe_min_observations > 0U &&
            gpaf_role_observations_[key] >= config_.gpaf_probe_min_observations &&
            residents.size() >= config_.gpaf_probe_min_residents &&
            costed_gate_passes) {
            phase->second = static_cast<std::uint8_t>(GpafSlotPhase::Active);
            ++gpaf_slot_promotions_;
        }
```

with:

```cpp
        // Live costed-value evidence: EMA of this slot's per-visit
        // counterfactual contribution, net of a configurable per-visit
        // execution cost. This is the only writer of
        // gpaf_slot_costed_net_value_; previously the map was never written
        // and the costed gate below permanently blocked all promotions.
        auto& value_ema = gpaf_slot_costed_net_value_[key];
        const double decay = static_cast<double>(
            std::clamp(config_.gpaf_value_ema_decay, 0.0F, 0.999999F));
        const double sample = static_cast<double>(node.contribution) -
            static_cast<double>(config_.gpaf_execution_cost_weight);
        value_ema = decay * value_ema + (1.0 - decay) * sample;

        const bool costed_gate_passes =
            !config_.gpaf_active_requires_positive_net_value ||
            value_ema > 0.0;
        if (phase->second == static_cast<std::uint8_t>(GpafSlotPhase::Probe) &&
            config_.gpaf_probe_min_observations > 0U &&
            gpaf_role_observations_[key] >= config_.gpaf_probe_min_observations &&
            residents.size() >= config_.gpaf_probe_min_residents &&
            costed_gate_passes) {
            phase->second = static_cast<std::uint8_t>(GpafSlotPhase::Active);
            ++gpaf_slot_promotions_;
        } else if (phase->second == static_cast<std::uint8_t>(GpafSlotPhase::Active) &&
                   config_.gpaf_active_requires_positive_net_value &&
                   value_ema <= 0.0) {
            phase->second = static_cast<std::uint8_t>(GpafSlotPhase::Quarantined);
            ++gpaf_slot_quarantines_;
        }
```

Both the writer and the demotion branch are unconditionally reached only when
`gpaf_shadow_observation` is on (guarded at function entry); demotion only
fires when `gpaf_active_requires_positive_net_value` is enabled, so default
behavior (flag off) is unchanged except that `gpaf_slot_costed_net_value_`
now actually accumulates evidence instead of staying empty.

- [ ] **Step 5: Rebuild and observe real behavior of the existing costed-gate test**

```bash
cmake --build build-fast --target sbm_scaling_tests sbm_c_api_tests -j4
./build-fast/sbm_scaling_tests.exe gpaf_costed_gate
```

`verify_gpaf_costed_active_gate_blocks_without_positive_value`
(`tests/test_scaling.cpp:809-841`) asserts `gated_diag.gpaf_slot_promotions
== 0U` and `gated_diag.gpaf_active_slots == 0U` after 320 steps with
`gpaf_active_requires_positive_net_value=true` and default
`gpaf_execution_cost_weight=0.0F`. Two outcomes are possible now that the
gate is evidence-driven instead of a permanent no-op:

- **If it still passes:** the synthetic pattern's average per-visit
  contribution is non-positive in this configuration — record this and move
  to Step 6 unchanged.
- **If it fails** (promotions/active_slots > 0): this is expected —  the
  gate is now real and the deterministic bijection pattern in this test
  produces positive-contribution residents. Update the test in place: rename
  it to `verify_gpaf_costed_active_gate_blocks_under_high_execution_cost`,
  set `gated_config.gpaf_execution_cost_weight = 5.0F` (large enough to
  outweigh any realistic per-visit contribution in this synthetic setup —
  contributions are bounded by the cross-entropy scale, at most a few nats),
  and re-run. Iterate the weight upward only if still not blocking; do not
  disable the gate logic itself to force the old assertion to pass.

- [ ] **Step 6: Add a complementary allow-case test**

Add a new test function in `tests/test_scaling.cpp`, immediately after
`verify_gpaf_costed_active_gate_blocks_without_positive_value`:

```cpp
void verify_gpaf_costed_active_gate_allows_positive_value() {
    auto config = sparse_config(4U, 16U);
    config.adaptive_topology = false;
    config.beam_width = 6U;
    config.gpaf_shadow_observation = true;
    config.gpaf_candidate_retrieval = true;
    config.gpaf_query_keys_per_step = 2U;
    config.gpaf_slots = 64U;
    config.gpaf_residents_per_slot = 3U;
    config.gpaf_probe_min_observations = 8U;
    config.gpaf_probe_min_residents = 2U;
    config.gpaf_active_requires_positive_net_value = true;
    config.gpaf_execution_cost_weight = 0.0F;
    sbm::SparseBranchMachine machine(config);

    for (std::uint32_t step = 0U; step < 2048U; ++step) {
        const std::uint32_t token = step % 16U;
        const std::uint32_t target = (step * 7U + 5U) % 16U;
        (void)machine.step_token(token, target, true);
    }

    const auto diag = machine.diagnostics();
    assert(diag.gpaf_role_observations > 0U);
    assert(diag.gpaf_probe_slots + diag.gpaf_active_slots +
           diag.gpaf_quarantined_slots ==
           diag.gpaf_unique_role_keys);
    // With zero execution cost, at least one role key must accumulate
    // positive live value over enough visits, or the gate can never be
    // exercised at all — this is a coverage check on the writer, not a
    // guarantee about GPAF's real predictive value.
    assert(diag.gpaf_slot_promotions > 0U);
    assert(diag.gpaf_active_slots > 0U);
}
```

Register it in `main()` under the `gpaf_costed_gate` mode (find
`if (mode == "gpaf_costed_gate") { verify_gpaf_costed_active_gate_blocks_...(); ... }`
around line 993-996) so both the blocking and allowing cases run together:

```cpp
    if (mode == "gpaf_costed_gate") {
        verify_gpaf_costed_active_gate_blocks_without_positive_value();
        verify_gpaf_costed_active_gate_allows_positive_value();
        std::cout << "GPAF costed active gate passed\n";
        return 0;
    }
```

If `gpaf_slot_promotions` is still `0U` after 2048 steps, increase steps to
4096 before concluding the writer is broken — with `gpaf_probe_min_residents
= 2U` and `gpaf_residents_per_slot = 3U`, a slot needs at least 2 distinct
resident nodes and 8 observations before the EMA has enough samples to turn
positive; do not lower `gpaf_probe_min_observations` to force a pass, since
that changes an unrelated precondition.

- [ ] **Step 7: Rebuild, run full GPAF suite, verify checkpoint round-trip of new config fields**

```bash
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests sbm_scaling_tests -j4
ctest --test-dir build-fast --output-on-failure -R "sbm_c_api|sbm_cpp_api|sbm_scaling_gpaf"
python tests/test_presets.py
```

Expected: PASS. `test_presets.py` must still pass unchanged since the two new
fields are not added to `gpaf-shadow-v1`/`gpaf-retrieval-v1` preset dicts
(they default to `0.98F`/`0.0F`, preserving current preset behavior).

- [ ] **Step 8: Commit**

```bash
git add include/sbm/types.hpp src/modules/machine_topology.cpp src/modules/machine_checkpoint.cpp src/api/c_api.cpp tests/test_scaling.cpp
git commit -m "feat: implement live GPAF costed-value writer and evidence-based lifecycle gating"
```

## Task 4: Calibrated GPAF-unique candidate scoring

**Root cause (RC4 in the investigation doc):** `score()`
(`src/modules/machine_routing.cpp:208-221`) applies token-signature Hamming
similarity and an exact-bucket bonus uniformly to every candidate regardless
of source. A genuinely global GPAF-unique candidate's prototype signature
comes from a different context, so it scores low on the two dominant terms
(`0.42·exact + 0.72·hamming`) and rarely enters the beam on its own merit;
when it does activate, it is uncalibrated by any measured value. This task
scores GPAF-unique candidates from the slot's live costed value (from Task 3)
instead, and only lets them in when that value is positive — the "reserved
beam quota" from Proposal D, implemented as a conservative override rather
than a fixed slot, since the honest per-slot value signal is now available
to gate on directly.

**Files:**
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_routing.cpp`
- Test: `tests/test_scaling.cpp`

- [ ] **Step 1: Add a private accessor for slot value**

In `include/sbm/machine.hpp`, in the private method section immediately
after the declaration of `gpaf_structural_call_key_for_channel` (around line
184), add:

```cpp
    [[nodiscard]] double gpaf_slot_value(std::uint64_t key) const noexcept;
```

- [ ] **Step 2: Implement the accessor**

In `src/modules/machine_topology.cpp`, immediately after
`gpaf_structural_call_key_for_channel`'s closing brace (currently ending at
line 314), add:

```cpp
double SparseBranchMachine::gpaf_slot_value(std::uint64_t key) const noexcept {
    const auto found = gpaf_slot_costed_net_value_.find(key);
    return found == gpaf_slot_costed_net_value_.end() ? 0.0 : found->second;
}
```

- [ ] **Step 3: Apply calibrated scoring to GPAF-unique candidates only**

In `src/modules/machine_routing.cpp`, in `select_route`
(currently lines 223-243), the scoring loop currently reads:

```cpp
    for (const auto& candidate : candidates) {
        const auto slot = slot_of(candidate.id);
        if (slot == SIZE_MAX) continue;
        const auto channel = channels_[slot];
        const bool exact = bucket(prototypes_[slot]) == bucket(signatures[channel]);
        const double similarity = hamming_similarity(prototypes_[slot],
                                                     signatures[channel]);
        route_score_hamming_sum_ += similarity;
        route_score_exact_sum_ += exact ? 1.0 : 0.0;
        route_score_edge_prior_sum_ += static_cast<double>(candidate.edge_prior);
        scored.push_back({score(slot, signatures, candidate.edge_prior), candidate.id,
                          0.0F, 0.0F, exact, channel, candidate.source,
                          candidate.gpaf_key, candidate.gpaf_overlap});
    }
```

Replace it with:

```cpp
    for (const auto& candidate : candidates) {
        const auto slot = slot_of(candidate.id);
        if (slot == SIZE_MAX) continue;
        const auto channel = channels_[slot];
        const bool exact = bucket(prototypes_[slot]) == bucket(signatures[channel]);
        const double similarity = hamming_similarity(prototypes_[slot],
                                                     signatures[channel]);
        route_score_hamming_sum_ += similarity;
        route_score_exact_sum_ += exact ? 1.0 : 0.0;
        route_score_edge_prior_sum_ += static_cast<double>(candidate.edge_prior);
        double candidate_score;
        if (candidate.source == CandidateSource::GpafRole &&
            !candidate.gpaf_overlap) {
            // A genuinely GPAF-unique candidate's token-signature similarity
            // to the current context is not a meaningful predictive-role
            // signal by design (the whole point of GPAF is to retrieve
            // structure that is NOT locally reachable by signature). Score
            // it from the slot's measured live value instead of Hamming
            // similarity, and require positive evidence before it can win a
            // beam slot at all.
            const double value = gpaf_slot_value(candidate.gpaf_key);
            const double reliability = 1.0 / (1.0 + loss_ema_[slot]);
            const double novelty = 1.0 /
                std::sqrt(static_cast<double>(visits_[slot]) + 1.0);
            candidate_score = value <= 0.0
                ? -1.0
                : std::tanh(value) + 0.10 * reliability + 0.02 * novelty;
        } else {
            candidate_score = score(slot, signatures, candidate.edge_prior);
        }
        scored.push_back({candidate_score, candidate.id,
                          0.0F, 0.0F, exact, channel, candidate.source,
                          candidate.gpaf_key, candidate.gpaf_overlap});
    }
```

A `candidate_score` of `-1.0` for non-positive-value GPAF-unique candidates
keeps them in the scored list (so they still count toward
`candidates_examined` diagnostics) but makes them lose every ranked
comparison against any locally-sourced candidate — `score()`'s minimum
possible value is `phase_bias = -0.18` for a Dormant node with zero
similarity/reliability/novelty, so `-1.0` is reliably below it. They can
still win a beam slot in the degenerate case where *no* other candidates
exist for a channel, which is acceptable (bounded fallback, same as today).
GPAF-overlap candidates and all other sources are scored exactly as before —
this task changes nothing about local retrieval or already-reachable
candidates.

- [ ] **Step 4: Rebuild and verify no regression, then verify the calibration effect**

```bash
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests sbm_scaling_tests -j4
ctest --test-dir build-fast --output-on-failure -R "sbm_c_api|sbm_cpp_api|sbm_scaling_gpaf"
```

Expected: PASS. `verify_gpaf_candidate_retrieval_is_bounded` does not gate
`gpaf_active_requires_positive_net_value`, so its GPAF-unique candidates are
scored via the new branch with whatever `gpaf_slot_value` currently holds
(likely `0.0` early in training since the writer needs visits to accumulate,
per Task 3) — if `gpaf_unique_active_nodes > 0U` (line 570) starts failing
because unique candidates now score `-1.0` and never win a beam slot in this
short 512-step synthetic run, that is the calibration doing exactly what it
should (uncalibrated GPAF candidates should not activate). Fix the test by
running long enough for at least one slot to accumulate positive value, or by
explicitly seeding `gpaf_probe_min_observations`/steps the same way Task 3's
Step 6 did. Do not weaken the scoring change to force the old assertion to
pass unmodified — update the test's step count/config and note in a comment
why (calibration requires evidence before GPAF-unique candidates can win a
slot).

- [ ] **Step 5: Add a regression test for the negative-value block**

Add a new test in `tests/test_scaling.cpp`, after
`verify_gpaf_candidate_retrieval_is_bounded`:

```cpp
void verify_gpaf_unique_candidates_need_positive_value_to_activate() {
    auto config = sparse_config(4U, 16U);
    config.adaptive_topology = false;
    config.beam_width = 6U;
    config.gpaf_shadow_observation = true;
    config.gpaf_candidate_retrieval = true;
    config.gpaf_query_keys_per_step = 2U;
    config.gpaf_slots = 64U;
    config.gpaf_residents_per_slot = 3U;
    config.gpaf_probe_min_observations = 8U;
    config.gpaf_probe_min_residents = 2U;
    sbm::SparseBranchMachine machine(config);

    // Immediately after construction, no slot has any recorded value yet
    // (gpaf_slot_costed_net_value_ is empty), so every GPAF-unique
    // candidate must score below every locally-sourced candidate and
    // therefore never win a beam slot ahead of an available exact resident.
    for (std::uint32_t step = 0U; step < 4U; ++step) {
        const auto stats = machine.step_token(step % 16U, (step * 7U + 5U) % 16U, true);
        assert(stats.active_nodes > 0U);
    }
    const auto diag = machine.diagnostics();
    assert(diag.gpaf_candidates_returned > 0U);
    assert(diag.gpaf_unique_active_nodes == 0U);
}
```

Register it under `gpaf_retrieval` mode alongside
`verify_gpaf_candidate_retrieval_is_bounded`:

```cpp
    if (mode == "gpaf_retrieval") {
        verify_gpaf_candidate_retrieval_is_bounded();
        verify_gpaf_unique_candidates_need_positive_value_to_activate();
        std::cout << "GPAF candidate retrieval passed\n";
        return 0;
    }
```

- [ ] **Step 6: Rebuild and run full GPAF and non-GPAF suites**

```bash
cmake --build build-fast --target sbm_api sbm_tests sbm_c_api_tests sbm_scaling_tests -j4
ctest --test-dir build-fast --output-on-failure
python tests/test_presets.py
```

Expected: PASS across the whole suite, not just GPAF tests — this task
touches shared `select_route` scoring code, so non-GPAF routing
(`candidate.source != GpafRole` always takes the unchanged `score(...)`
path) must be verified unaffected by running the complete CTest set.

- [ ] **Step 7: Commit**

```bash
git add include/sbm/machine.hpp src/modules/machine_topology.cpp src/modules/machine_routing.cpp tests/test_scaling.cpp
git commit -m "feat: score GPAF-unique candidates from calibrated slot value, not token-signature similarity"
```

## Task 5: Documentation and real-corpus handoff

No further code change. Update canonical docs with what was actually fixed
and verified, then hand off real-corpus validation to the user (matching the
pattern of Task 5 in the predecessor plan).

**Files:**
- Modify: `DESIGN_NOTES.md` (§Next-generation global predictive addressing)
- Modify: `PROGRESS.md`
- Append: `RESEARCH_LOG.md`

- [ ] **Step 1: Update `DESIGN_NOTES.md`**

Append a dated paragraph to §Next-generation global predictive addressing
stating: the units bug in cost/net-value aggregation is fixed (Task 1); the
frozen ablation no longer credits GPAF-overlap logits as causal GPAF value,
so the aggregate/per-key gain now reflects unique-only causal removal (Task
2); `gpaf_slot_costed_net_value_` now has a live EMA writer and
`gpaf_active_requires_positive_net_value` is a real evidence-driven
Probe→Active gate with Active→Quarantined demotion (Task 3); GPAF-unique
candidates are now scored from measured slot value with a hard block on
non-positive value instead of token-signature Hamming similarity (Task 4).
State plainly that none of this has been validated on real corpus data yet —
all evidence so far is synthetic CTest coverage.

- [ ] **Step 2: Update `PROGRESS.md`**

Add a status line noting Tasks 1-4 of this plan are implemented and covered
by CTest, and that a fresh 10M FineWeb-Edu validation (repeating the
`upgrade-v1` vs `gpaf-retrieval-v1` comparison from the 2026-07-02 run) is
required before drawing any conclusion about whether the fixes change GPAF's
real-corpus value.

- [ ] **Step 3: Append to `RESEARCH_LOG.md`**

Record what was fixed, why (root causes RC3/RC5/RC6 from the investigation
doc), and the exact synthetic-test evidence obtained while implementing
(e.g. the actual step counts and outcomes discovered in Task 3 Step 5/6 and
Task 4 Step 4 — fill in with what was actually observed during
implementation, not hypothetical numbers).

- [ ] **Step 4: Request real-corpus validation**

Stop and ask the user to run, on the same 10M FineWeb-Edu setup as the prior
validation:

```text
upgrade-v1
gpaf-retrieval-v1
```

with `gpaf_execution_cost_weight` and `gpaf_active_requires_positive_net_value`
set explicitly (rather than left at their conservative defaults) so the
costed gate is actually exercised at scale, and report:

```text
eval NLL (both configs, 2 seeds)
gpaf_unique_ablation_mean_gain / gpaf_overlap_ablation_mean_gain
gpaf_ablation_key_net_value per key (now nats/example, not raw sums)
gpaf_slot_promotions / gpaf_slot_quarantines
gpaf_unique_active_nodes (should now require positive slot value to be > 0)
```

- [ ] **Step 5: Commit**

```bash
git add DESIGN_NOTES.md PROGRESS.md RESEARCH_LOG.md
git commit -m "docs: record GPAF honest-value and calibrated-scoring fixes"
```

## Self-review checklist

- Spec coverage: Task 1 = Proposal A item 5 (cost calibration, RC6); Task 2 =
  Proposal A item 1's causal intent (RC5), implemented via ablation-formula
  correction rather than `push_candidate` mutation once analysis showed the
  source-overwrite is load-bearing for the existing overlap/unique counters
  and not itself the source of the inflation; Task 3 = Proposal A items 3-4
  (live writer, value-gated lifecycle, RC3); Task 4 = Proposal D (calibrated
  GPAF-unique scoring, RC4). Proposal A item 2 (re-route counterfactual
  ablation) and Proposals B/C/E are explicitly out of scope for this plan —
  they are larger, separate research efforts gated on this plan's honest
  measurement landing first, per the investigation doc's recommended order.
- Placeholder scan: no TBD/TODO markers; Task 3/4 steps that depend on
  build-time numeric outcomes give explicit fallback instructions
  (increase steps, adjust `gpaf_execution_cost_weight`) rather than
  hand-waving "adjust as needed" — engineer must record what was actually
  observed.
- Type consistency: `gpaf_value_ema_decay`, `gpaf_execution_cost_weight`,
  `gpaf_slot_value` used consistently across Config, checkpoint, c_api and
  routing call sites.
