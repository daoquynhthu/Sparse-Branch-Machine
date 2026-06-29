# Neuronal Address Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current one-shot sparse address signatures with a mature neuron-like address-operation framework: explicit address execution, persistent committed structures, dependency-aware attribution, and codelength/cost accounting before further scale-up.

**Architecture:** The current `AddressProgram` remains the compatibility unit, but it becomes an executable program with frames, bindings, lineage and measured costs rather than a direct hash recipe. Accepted structures are persistent by default: harmful or stale structures are first quarantined and ablated, not physically erased. Experiments are admitted only after the interpreter, persistence model and attribution gates pass; tuning of an incomplete framework is explicitly out of scope.

**Tech Stack:** C++20 core (`include/sbm`, `src/modules`), CMake/CTest, stable C ABI JSON reporting, Python corpus runner diagnostics, existing mapped corpus manifests.

---

## Why This Replaces The Old Immediate Plan

The previous immediate plan framed the blocker as adaptive content fusion losing
to the current-token baseline. That was true but too narrow. The latest evidence
shows a deeper architectural boundary:

- accepted content structures exist and have held-out document-wide credit;
- the current system still executes address programs as one-shot signatures;
- online physical prune destroyed useful accepted structure before the default
  was changed to persistence;
- current `ContentMatch` and `ContentFollow` are useful prototypes, not a full
  binding/follow/caller system;
- more cap/rate/channel sweeps would optimize an unfinished mechanism.

This plan therefore blocks R3 scale-up and broad hyperparameter search until the
complete address-semantics framework exists.

## Non-Negotiable Invariants

- No accepted high-credit structure is physically deleted by default.
- Probe rejection may erase uncommitted candidate state; committed structures
  use quarantine, masking and recoverable retirement before any physical erase.
- Address execution is explicit and auditable: every frame reports operation,
  source, binding, signature, cost and dependency.
- Attribution is dependency-aware: ablating a caller does not incorrectly assign
  all prerequisite failures to one shared structure.
- Large experiments cannot be used to compensate for missing semantics.
- Performance work is allowed only when it preserves the interpreter contract
  and bounded active work.

## File Structure

- Modify `include/sbm/types.hpp`: add address execution records, binding kinds,
  channel lifecycle states, cost counters and diagnostics fields.
- Modify `include/sbm/machine.hpp`: add interpreter buffers, channel masks,
  lineage storage and quarantine helpers.
- Create `include/sbm/detail/address_interpreter.hpp`: pure address-program
  execution API used by both training and tests.
- Create `src/modules/address_interpreter.cpp`: interpreter implementation for
  Tuple, DeltaMod, ContentMatch and ContentFollow plus reusable frame hashing.
- Modify `src/modules/machine_topology.cpp`: proposal, accept, quarantine and
  rollback lifecycle.
- Modify `src/modules/machine_routing.cpp`: route selection from interpreter
  frames instead of direct signatures.
- Modify `src/modules/token_sparse_output.cpp`: record execution frames and
  dependency-aware channel attribution during frozen evaluation.
- Modify `src/modules/machine_maintenance.cpp`: diagnostics and recoverable
  lifecycle accounting.
- Modify `src/api/c_api.cpp`: expose new schema parameters and diagnostics.
- Modify `src/modules/token_experiment.cpp`: emit program execution, lifecycle
  and structural value summaries.
- Modify `scripts/run_corpus_training.py`: accept structural validation modes
  without changing normal training runs.
- Modify `tests/test_scaling.cpp`: bounded-work and persistence gates.
- Modify `tests/test_sbm.cpp`: interpreter and lifecycle unit coverage.
- Modify `tests/test_c_api.cpp` and `tests/test_python_api.py`: schema and JSON
  compatibility checks.
- Modify `DESIGN_NOTES.md`, `ISSUES.md`, `ROADMAP_REAL_DATA.md`,
  `RESEARCH_LOG.md`: make this plan the current architecture route.

### Task 1: Reframe The Canonical Docs And Block Local Tuning

**Files:**
- Modify: `ISSUES.md`
- Modify: `ROADMAP_REAL_DATA.md`
- Modify: `DESIGN_NOTES.md`
- Modify: `RESEARCH_LOG.md`
- Test: documentation consistency through `rg`

- [x] **Step 1: Replace P0-8 framing**

In `ISSUES.md`, replace the old fusion-only P0-8 title with:

```markdown
### P0-8: Address semantics are incomplete despite positive accepted-structure credit
```

Replace the failure-mode paragraph with:

```markdown
Failure mode: the current system has validated content-conditioned address
primitives, but it still executes them as one-shot signature generators with
weak caller/dependency semantics. Accepted structures can have positive frozen
counterfactual credit while the complete model still loses to a current-token
control. Treating this as a fusion or learning-rate problem would continue
optimizing an incomplete address framework.
```

- [x] **Step 2: Replace the P0-8 fix plan**

Replace the old P0-8 fix list with this exact list:

```markdown
Fix plan:

- implement explicit address-program execution frames for Tuple, DeltaMod,
  ContentMatch and ContentFollow;
- preserve accepted structures by default through quarantine and masking rather
  than physical deletion;
- add dependency-aware attribution so callers and prerequisites can be ablated
  separately;
- report held-out codelength gain, description cost and execution cost for
  every accepted program;
- allow medium/large experiments only after the interpreter, lifecycle and
  attribution tests pass.
```

- [x] **Step 3: Update `ROADMAP_REAL_DATA.md` immediate work**

Replace the R2 immediate work list with:

```markdown
Immediate work before R3:

- execute `docs/superpowers/plans/2026-06-27-neuronal-address-semantics.md`;
- stop cap/rate/fusion sweeps unless they test a completed semantic contract;
- keep the 10M/1M FineWeb-Edu corpus as a validation gate, not as a substitute
  for implementing the address framework;
- move to R3 only after explicit address execution, persistent accepted
  structures, dependency-aware attribution and structural value accounting are
  implemented and verified.
```

- [x] **Step 4: Update `DESIGN_NOTES.md` program-language boundary**

Append this paragraph to the `Program-language boundary` section:

```markdown
The next architecture milestone is not another address-operator sweep. The
machine must first promote address programs from signature recipes to explicit
execution objects with frames, bindings, lineage, dependency-aware ablation and
recoverable lifecycle state. Until that exists, experiments may diagnose but
may not claim that the neuron-like address mechanism is complete.
```

- [x] **Step 5: Run documentation consistency checks**

Run:

```powershell
rg -n "seed-only versus accepted-channel|diagnose seed-only|2026-06-27-neuronal-address-semantics|Address semantics are incomplete" ISSUES.md ROADMAP_REAL_DATA.md DESIGN_NOTES.md
```

Expected:

- no remaining active-plan instruction that frames P0-8 as only a fusion sweep;
- one reference to `2026-06-27-neuronal-address-semantics.md`;
- one P0-8 title using the new address-semantics framing at Task 1 time. After
  Task 8 validation, this item is expected to live under `ISSUES.md` resolved
  history rather than the active queue.

- [x] **Step 6: Commit**

```powershell
git add ISSUES.md ROADMAP_REAL_DATA.md DESIGN_NOTES.md RESEARCH_LOG.md docs/superpowers/plans/2026-06-27-neuronal-address-semantics.md
git commit -m "docs: redirect plan to address semantics"
```

### Task 2: Add Address Execution Types And Diagnostics

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `src/api/c_api.cpp`
- Test: `tests/test_c_api.cpp`

- [x] **Step 1: Add failing schema and diagnostics tests**

In `tests/test_c_api.cpp`, add assertions that the config schema contains:

```cpp
expect_schema_contains(schema, "address_execution_mode");
expect_schema_contains(schema, "accepted_channel_retirement");
```

Add result JSON assertions for these diagnostic fields:

```cpp
expect_json_field(result, "address_execution_frames");
expect_json_field(result, "address_binding_hits");
expect_json_field(result, "address_binding_misses");
expect_json_field(result, "quarantined_channels");
expect_json_field(result, "recoverable_retired_channels");
expect_json_field(result, "structural_value_nats");
expect_json_field(result, "structural_description_cost");
expect_json_field(result, "structural_execution_cost");
```

- [x] **Step 2: Run the failing test**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_c_api_tests sbm_api
build-fast\sbm_c_api_tests.exe
```

Expected: fail because the schema and diagnostics fields do not exist.

- [x] **Step 3: Extend public types**

Add to `include/sbm/types.hpp`:

```cpp
enum class AddressExecutionMode : std::uint8_t {
    LegacySignature = 0U,
    InterpretedFrames = 1U
};

enum class AcceptedChannelRetirement : std::uint8_t {
    Preserve = 0U,
    Quarantine = 1U,
    PhysicalErase = 2U
};

enum class AddressBindingKind : std::uint8_t {
    None = 0U,
    Positional = 1U,
    ContentMatch = 2U,
    ContentFollow = 3U
};

struct AddressExecutionFrame {
    AddressProgram program{};
    AddressBindingKind binding{AddressBindingKind::None};
    std::uint32_t source_index{};
    std::uint32_t matched_index{};
    std::uint32_t successor{};
    std::uint32_t dependency{};
    std::uint64_t signature{};
    float description_cost{};
    float execution_cost{};
    bool matched{};
};
```

Add to `Config`:

```cpp
AddressExecutionMode address_execution_mode{AddressExecutionMode::InterpretedFrames};
AcceptedChannelRetirement accepted_channel_retirement{AcceptedChannelRetirement::Preserve};
float structural_description_cost_weight{1.0F};
float structural_execution_cost_weight{0.0F};
```

Add to `Diagnostics`:

```cpp
std::uint64_t address_execution_frames{};
std::uint64_t address_binding_hits{};
std::uint64_t address_binding_misses{};
std::uint64_t quarantined_channels{};
std::uint64_t recoverable_retired_channels{};
double structural_value_nats{};
double structural_description_cost{};
double structural_execution_cost{};
```

- [x] **Step 4: Emit zero-valued diagnostics**

In `src/modules/machine_maintenance.cpp`, initialize the new diagnostics from
machine counters. Before interpreter implementation, counters may be zero except
channel lifecycle counts.

In `src/modules/token_experiment.cpp`, emit the fields into token JSON with the
existing diagnostics block.

In `src/api/c_api.cpp`, register:

```cpp
{"address_execution_mode", "enum", "InterpretedFrames", "", "", "categorical", false, false, false, "Address execution backend: LegacySignature or InterpretedFrames."},
{"accepted_channel_retirement", "enum", "Preserve", "", "", "categorical", false, false, false, "Lifecycle policy for accepted channels: Preserve, Quarantine, RecoverableRetire or PhysicalErase."},
{"structural_description_cost_weight", "float", "1.0", "0.0", "10.0", "linear", true, false, false, "Weight applied to program description cost."},
{"structural_execution_cost_weight", "float", "0.0", "0.0", "10.0", "linear", true, false, false, "Weight applied to measured address execution cost."},
```

- [x] **Step 5: Run schema tests**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_c_api_tests sbm_api
build-fast\sbm_c_api_tests.exe
```

Expected: pass.

- [x] **Step 6: Commit**

```powershell
git add include/sbm/types.hpp src/modules/machine_maintenance.cpp src/modules/token_experiment.cpp src/api/c_api.cpp tests/test_c_api.cpp
git commit -m "feat: add address execution diagnostics"
```

### Task 3: Implement The Address Interpreter

**Files:**
- Create: `include/sbm/detail/address_interpreter.hpp`
- Create: `src/modules/address_interpreter.cpp`
- Modify: `CMakeLists.txt`
- Modify: `src/modules/machine_topology.cpp`
- Test: `tests/test_sbm.cpp`

- [x] **Step 1: Write failing interpreter tests**

In `tests/test_sbm.cpp`, add tests:

```cpp
TEST(AddressInterpreter, TupleMatchesLegacySignature) {
    Config config;
    config.token_alphabet = 4096;
    const std::uint32_t data[] = {11, 29, 31, 47, 53};
    auto program = singleton_address_program(2);
    AddressExecutionFrame frame{};
    execute_address_program(std::span<const std::uint32_t>(data, 5), config.token_alphabet,
                            program, 7U, frame);
    EXPECT_EQ(frame.binding, AddressBindingKind::Positional);
    EXPECT_TRUE(frame.matched);
    EXPECT_EQ(frame.source_index, 2U);
    EXPECT_EQ(frame.signature, address_program_signature(
        std::span<const std::uint32_t>(data, 5), config.token_alphabet,
        std::span<const std::uint32_t>(program.lags.data(), program.arity),
        program.op, 7U));
}

TEST(AddressInterpreter, ContentFollowReportsBindingAndDependency) {
    Config config;
    config.token_alphabet = 4096;
    const std::uint32_t data[] = {4, 8, 9, 4, 8, 13, 4, 8};
    AddressProgram program;
    program.lags[0] = 1U;
    program.lags[1] = 6U;
    program.arity = 2U;
    program.op = AddressOp::ContentFollow;
    AddressExecutionFrame frame{};
    execute_address_program(std::span<const std::uint32_t>(data, 8), config.token_alphabet,
                            program, 11U, frame);
    EXPECT_EQ(frame.binding, AddressBindingKind::ContentFollow);
    EXPECT_TRUE(frame.matched);
    EXPECT_EQ(frame.successor, 13U);
    EXPECT_NE(frame.dependency, 0U);
    EXPECT_GT(frame.execution_cost, 0.0F);
}
```

- [x] **Step 2: Run the failing tests**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_tests
build-fast\sbm_tests.exe --gtest_filter=*AddressInterpreter*
```

Expected: fail because `execute_address_program` is undefined.

- [x] **Step 3: Add interpreter header**

Create `include/sbm/detail/address_interpreter.hpp`:

```cpp
#pragma once

#include "sbm/types.hpp"

#include <cstdint>
#include <span>

namespace sbm {

[[nodiscard]] bool execute_address_program(
    std::span<const std::uint32_t> window,
    std::uint32_t alphabet,
    const AddressProgram& program,
    std::uint64_t seed,
    AddressExecutionFrame& out) noexcept;

} // namespace sbm
```

- [x] **Step 4: Implement interpreter semantics**

Create `src/modules/address_interpreter.cpp` with these semantics:

```cpp
#include "sbm/detail/address_interpreter.hpp"

#include "sbm/math.hpp"

#include <algorithm>

namespace sbm {
namespace {

std::uint32_t safe_history_index(std::span<const std::uint32_t> window,
                                 std::uint32_t lag) noexcept {
    if (window.empty()) return 0U;
    if (window.size() > lag) {
        return static_cast<std::uint32_t>(window.size() - 1U - lag);
    }
    return 0U;
}

} // namespace

bool execute_address_program(std::span<const std::uint32_t> window,
                             std::uint32_t alphabet,
                             const AddressProgram& program,
                             std::uint64_t seed,
                             AddressExecutionFrame& out) noexcept {
    out = {};
    out.program = program;
    out.signature = address_program_signature(
        window, alphabet,
        std::span<const std::uint32_t>(program.lags.data(), program.arity),
        program.op, seed);
    out.description_cost = 1.0F + static_cast<float>(program.arity);
    out.execution_cost = 1.0F;
    if (window.empty()) return false;

    if (program.op == AddressOp::Tuple || program.op == AddressOp::DeltaMod) {
        out.binding = AddressBindingKind::Positional;
        out.matched = true;
        out.source_index = safe_history_index(window, program.lags[program.arity - 1U]);
        out.dependency = program.lags[program.arity - 1U];
        out.execution_cost += static_cast<float>(program.arity);
        return true;
    }

    const auto current_index = window.size() - 1U;
    const auto current = window[current_index];
    const std::uint32_t max_lag = program.arity == 0U ? 0U : program.lags[program.arity - 1U];
    const auto bounded_lag = std::min<std::size_t>(max_lag, current_index);
    const std::size_t pattern_count =
        program.op == AddressOp::ContentFollow && program.arity > 1U
            ? program.arity - 1U
            : 0U;
    out.binding = program.op == AddressOp::ContentFollow
        ? AddressBindingKind::ContentFollow
        : AddressBindingKind::ContentMatch;

    for (std::size_t distance = 1U; distance <= bounded_lag; ++distance) {
        out.execution_cost += 1.0F;
        const auto candidate_index = current_index - distance;
        if (window[candidate_index] != current) continue;
        bool pattern_matches = true;
        for (std::size_t index = 0; index < pattern_count; ++index) {
            const auto lag = static_cast<std::size_t>(program.lags[index]);
            if (lag > current_index || lag > candidate_index ||
                window[current_index - lag] != window[candidate_index - lag]) {
                pattern_matches = false;
                break;
            }
        }
        if (!pattern_matches) continue;
        out.matched = true;
        out.source_index = static_cast<std::uint32_t>(candidate_index);
        out.matched_index = static_cast<std::uint32_t>(candidate_index);
        out.successor = window[std::min(candidate_index + 1U, current_index)];
        out.dependency = static_cast<std::uint32_t>(distance);
        return true;
    }
    return false;
}

} // namespace sbm
```

- [x] **Step 5: Wire the target**

Add `src/modules/address_interpreter.cpp` to `CMakeLists.txt` wherever other
`src/modules/*.cpp` files are compiled into the SBM library.

Include the header from `src/modules/machine_topology.cpp`.

- [x] **Step 6: Run interpreter tests**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_tests
build-fast\sbm_tests.exe --gtest_filter=*AddressInterpreter*
```

Expected: pass.

- [x] **Step 7: Commit**

```powershell
git add CMakeLists.txt include/sbm/detail/address_interpreter.hpp src/modules/address_interpreter.cpp src/modules/machine_topology.cpp tests/test_sbm.cpp
git commit -m "feat: interpret address programs as frames"
```

### Task 4: Route From Execution Frames

**Files:**
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_topology.cpp`
- Modify: `src/modules/machine_routing.cpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/modules/machine_learning.cpp`
- Test: `tests/test_scaling.cpp`

- [x] **Step 1: Add failing bounded-work test**

In `tests/test_scaling.cpp`, add a mode `address_frames` that:

```cpp
Config config;
config.objective = ObjectiveKind::TokenCrossEntropy;
config.token_alphabet = 4096;
config.vector_dim = 4096;
config.context_width = 16;
config.address_execution_mode = AddressExecutionMode::InterpretedFrames;
config.adaptive_topology = true;
config.max_address_channels = 6;
config.beam_width = 6;
SparseBranchMachine model(config);
for (std::uint32_t i = 0; i < 20000; ++i) {
    model.step_token(i % config.token_alphabet, (i + 1U) % config.token_alphabet, true);
}
const auto diag = model.diagnostics();
EXPECT_GT(diag.address_execution_frames, 0U);
EXPECT_LE(diag.max_bucket_candidates_inspected,
          config.max_address_channels * config.bucket_scan_limit);
EXPECT_LE(diag.avg_active, static_cast<double>(config.beam_width));
```

- [x] **Step 2: Run the failing scaling mode**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_scaling_tests
build-fast\sbm_scaling_tests.exe address_frames
```

Expected: fail because frame counters are not updated.

- [x] **Step 3: Add interpreter buffers**

In `include/sbm/machine.hpp`, add:

```cpp
std::vector<AddressExecutionFrame> execution_frames_;
std::uint64_t address_execution_frames_{};
std::uint64_t address_binding_hits_{};
std::uint64_t address_binding_misses_{};
double structural_description_cost_{};
double structural_execution_cost_{};
```

Change `make_signatures` into a compatibility wrapper that fills frames first:

```cpp
[[nodiscard]] std::span<const AddressExecutionFrame> execute_address_programs(
    std::span<const std::uint32_t> window);
```

- [x] **Step 4: Implement frame production**

In `src/modules/machine_topology.cpp`, implement `execute_address_programs`:

```cpp
std::span<const AddressExecutionFrame> SparseBranchMachine::execute_address_programs(
    std::span<const std::uint32_t> window) {
    execution_frames_.clear();
    std::fill(signature_buffer_.begin(), signature_buffer_.end(), 0U);
    for (std::size_t channel = 0; channel < topology_.size(); ++channel) {
        if (!channel_enabled(channel)) continue;
        AddressExecutionFrame frame{};
        const auto& program = topology_[channel].program;
        const auto seed = config_.seed ^ mix64(address_program_key(program) +
                                              0x9E3779B97F4A7C15ULL);
        const bool matched = execute_address_program(
            window, config_.token_alphabet, program, seed, frame);
        signature_buffer_[channel] = frame.signature;
        execution_frames_.push_back(frame);
        ++address_execution_frames_;
        matched ? ++address_binding_hits_ : ++address_binding_misses_;
        structural_description_cost_ += frame.description_cost;
        structural_execution_cost_ += frame.execution_cost;
    }
    return {execution_frames_.data(), execution_frames_.size()};
}
```

Keep `make_signatures` as:

```cpp
std::span<const std::uint64_t> SparseBranchMachine::make_signatures(
    std::span<const std::uint32_t> window) {
    (void)execute_address_programs(window);
    return std::span<const std::uint64_t>(signature_buffer_.data(), topology_.size());
}
```

- [x] **Step 5: Preserve routing behavior**

Do not change scoring or bucket lookup yet. `src/modules/machine_routing.cpp`
continues to use `signature_buffer_`. This task verifies that the interpreter
is behavior-preserving before semantic routing changes.

- [x] **Step 6: Run full tests**

Run:

```powershell
cmake --build build-fast --config Release
build-fast\sbm_tests.exe
build-fast\sbm_scaling_tests.exe address_frames
build-fast\sbm_c_api_tests.exe
python tests\test_python_api.py --library E:\SPM\build-fast\libsbm_api.dll
```

Expected: all pass.

- [x] **Step 7: Commit**

```powershell
git add include/sbm/machine.hpp src/modules/machine_topology.cpp src/modules/machine_routing.cpp src/modules/token_sparse_output.cpp src/modules/machine_learning.cpp tests/test_scaling.cpp
git commit -m "feat: route through address execution frames"
```

### Task 5: Replace Physical Accepted-Channel Prune With Quarantine

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/machine_topology.cpp`
- Modify: `src/modules/machine_maintenance.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Test: `tests/test_sbm.cpp`

- [x] **Step 1: Add failing persistence test**

In `tests/test_sbm.cpp`, add:

```cpp
TEST(TopologyLifecycle, AcceptedChannelsAreNotPhysicallyErasedByDefault) {
    Config config;
    config.objective = ObjectiveKind::TokenCrossEntropy;
    config.token_alphabet = 64;
    config.vector_dim = 64;
    config.adaptive_topology = true;
    config.max_address_channels = 4;
    config.topology_prune_patience = 16;
    config.topology_prune_credit = 1.0F;
    config.accepted_channel_retirement = AcceptedChannelRetirement::Preserve;
    SparseBranchMachine model(config);
    for (std::uint32_t i = 0; i < 4096; ++i) {
        model.step_token(i % 64U, (i + 1U) % 64U, true);
    }
    const auto diag = model.diagnostics();
    EXPECT_EQ(diag.topology_pruned, 0U);
    EXPECT_EQ(diag.recoverable_retired_channels, 0U);
}
```

- [x] **Step 2: Run failing lifecycle test**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_tests
build-fast\sbm_tests.exe --gtest_filter=*TopologyLifecycle*
```

Expected: fail until lifecycle policy is implemented.

- [x] **Step 3: Extend channel phases**

In `include/sbm/types.hpp`, extend `ChannelPhase` with:

```cpp
Quarantined = 4U,
RecoverableRetired = 5U
```

Keep `Retired` as the physical deletion state for probe rejection and explicit
legacy erase policy.

- [x] **Step 4: Add lifecycle helpers**

In `include/sbm/machine.hpp`, declare:

```cpp
void quarantine_channel(std::size_t channel);
void recoverably_retire_channel(std::size_t channel);
void physically_erase_channel(std::size_t channel);
```

Rename the old `retire_channel` implementation in `src/modules/machine_topology.cpp`
to `physically_erase_channel`.

- [x] **Step 5: Implement policy**

In `maybe_finalize_topology_probe`:

- rejected probe uses `physically_erase_channel(channel)`;
- active-channel harmful result uses:

```cpp
if (config_.accepted_channel_retirement == AcceptedChannelRetirement::Preserve) {
    continue;
}
if (config_.accepted_channel_retirement == AcceptedChannelRetirement::Quarantine) {
    quarantine_channel(channel);
    ++topology_pruned_;
    return;
}
physically_erase_channel(channel);
++topology_pruned_;
return;
```

`quarantine_channel` sets phase to `Quarantined` and leaves nodes, outputs and
indexes intact. `recoverably_retire_channel` sets phase to `RecoverableRetired`
and disables routing while retaining state for audit.

**2026-06-28 corrective completion:** `RecoverableRetire` is now an exposed
accepted-channel retirement policy rather than dead helper code. The prune
path can select Preserve, Quarantine, RecoverableRetire or PhysicalErase.
Regression coverage verifies that recoverable retirement increments
`recoverable_retired_channels`, avoids physical retirement and leaves live
nodes intact.

- [x] **Step 6: Update diagnostics**

In `diagnostics()`, count `Quarantined` and `RecoverableRetired` channels into
the new fields.

- [x] **Step 7: Run lifecycle and scaling tests**

Run:

```powershell
cmake --build build-fast --config Release
build-fast\sbm_tests.exe --gtest_filter=*TopologyLifecycle*
build-fast\sbm_scaling_tests.exe
```

Expected: pass.

- [x] **Step 8: Commit**

```powershell
git add include/sbm/types.hpp include/sbm/machine.hpp src/modules/machine_topology.cpp src/modules/machine_maintenance.cpp src/modules/token_experiment.cpp tests/test_sbm.cpp
git commit -m "fix: quarantine accepted structures instead of deleting them"
```

### Task 6: Add Dependency-Aware Attribution

**Files:**
- Modify: `include/sbm/types.hpp`
- Modify: `include/sbm/machine.hpp`
- Modify: `src/modules/token_sparse_output.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `scripts/run_corpus_training.py`
- Test: `tests/test_scaling.cpp`

- [x] **Step 1: Add failing attribution test**

In `tests/test_scaling.cpp`, add a mode `dependency_attribution` that trains a
small content-follow case, freezes evaluation and asserts:

```cpp
EXPECT_GT(result.diagnostics.structural_value_nats, -1000000.0);
EXPECT_GE(result.diagnostics.structural_description_cost, 0.0);
EXPECT_GE(result.diagnostics.structural_execution_cost, 0.0);
```

Also assert the token experiment JSON contains:

```json
"eval_program_attribution": [
  {"channel": 1, "dependency": 0, "mean_credit": 0.0}
]
```

- [x] **Step 2: Run failing attribution test**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_scaling_tests
build-fast\sbm_scaling_tests.exe dependency_attribution
```

Expected: fail because program attribution JSON is missing.

- [x] **Step 3: Add attribution records**

In `include/sbm/types.hpp`, add:

```cpp
struct ProgramAttribution {
    std::uint8_t channel{};
    AddressProgram program{};
    std::uint32_t dependency{};
    double credit_sum{};
    double description_cost{};
    double execution_cost{};
    std::uint64_t observations{};
    std::uint64_t positive{};
};
```

Add `std::vector<ProgramAttribution>` to `TokenExperimentResult`.

- [x] **Step 4: Compute separate ablations**

In `src/modules/token_sparse_output.cpp`, during frozen evaluation with
`record_channel_attribution=true`, compute:

- full loss;
- loss without each channel's local logits;
- loss with caller channel removed but prerequisite channels retained;
- loss with prerequisite dependency removed when `AddressExecutionFrame::dependency`
  identifies a bound follow dependency.

Record credit as `without_loss - full_loss`. Accumulate description and
execution cost from frames.

- [x] **Step 5: Emit JSON**

In `src/modules/token_experiment.cpp`, emit:

```json
"eval_program_attribution": [
  {
    "channel": 1,
    "op": 3,
    "lags": [2, 3],
    "dependency": 3,
    "mean_credit": 0.0,
    "positive_fraction": 0.0,
    "description_cost": 0.0,
    "execution_cost": 0.0,
    "structural_value": 0.0
  }
]
```

`structural_value` is:

```text
credit_sum
- structural_description_cost_weight * description_cost
- structural_execution_cost_weight * execution_cost
```

- [x] **Step 6: Run attribution tests**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_scaling_tests sbm_api
build-fast\sbm_scaling_tests.exe dependency_attribution
python tests\test_python_api.py --library E:\SPM\build-fast\libsbm_api.dll
```

Expected: pass.

- [x] **Step 7: Commit**

```powershell
git add include/sbm/types.hpp include/sbm/machine.hpp src/modules/token_sparse_output.cpp src/modules/token_experiment.cpp scripts/run_corpus_training.py tests/test_scaling.cpp
git commit -m "feat: attribute address programs with dependencies"
```

**2026-06-28 corrective completion:** A follow-up audit found that the original
Task 6 implementation still treated `AddressExecutionFrame::dependency` as a
binding-distance value rather than a channel-level caller/prerequisite graph.
The implementation has now been extended so channel state, topology events,
execution frames, C/Python step stats, experiment JSON and C API summary JSON
carry `parent_channel` and `dependency_channel`. Frozen attribution now emits
`caller_removed_credit`, `dependency_retained_credit` and
`dependency_removed_credit`, with regression coverage proving that
`ContentFollow([1,2])` depends on the accepted `ContentMatch([2])` channel when
that prerequisite exists. This is an ABI-visible change and increments the C
API to v5.

**2026-06-28 typed-binding completion:** `AddressExecutionFrame` now carries an
`AddressBindingState` with current token, matched token, matched successor,
matched distance, pattern span, pattern-term count and match flag. The legacy
`source_index`, `matched_index`, `successor` and `dependency` fields remain as
compatibility aliases, but interpreter tests now verify the typed state for
positional, content-follow and no-match content-match cases. Frozen program
attribution reports binding match count, match fraction, mean binding distance
and mean pattern span. This extends the previous ABI-visible change and
increments the C API to v6.

**2026-06-28 binding-diagnostics completion:** Runtime diagnostics now include
`address_binding_by_kind`, a compact per-binding-kind summary of frames, hits,
hit rate, mean matched distance and mean pattern span. The counters are
first checkpointed in `SBMCKPT3` so exact resume preserves the diagnostic
stream.

**2026-06-28 shared-binding completion:** Content address signatures and
interpreted execution frames now share `resolve_address_binding_state()`.
`address_program_signature()` no longer carries a parallel ContentMatch /
ContentFollow search implementation, reducing semantic drift risk between
routing signatures and auditable execution frames.

**2026-06-28 dependency-graph completion:** Token experiment results and C API
machine summaries now emit an `address_dependency_graph`. The graph summarizes
each learned channel's program, phase, parent, dependency and direct caller
count. When frozen program attribution is enabled, token experiment JSON also
aggregates each channel's own credit, binding matches and downstream caller /
dependency-removal credit. This makes shared prerequisites and multi-caller
reuse auditable without changing the current operator set.

**2026-06-28 binding-key reuse completion:** `AddressBindingState` now carries a
stable typed `binding_key` for matched bindings. The key is derived from the
operation, program shape, current token, matched token, matched successor and
pattern shape, but not from absolute index or matched distance. This lets the
same content binding be recognized across variable distances while preserving
the existing routing signature for locality. C/Python step stats expose
`channel_binding_key`, and frozen program attribution reports
`unique_binding_keys` plus `binding_key_reuse_events`. This increments the C API
to v7.

**2026-06-29 persistent binding-reuse completion:** Binding keys now feed a
bounded per-channel training registry rather than remaining only step-local
diagnostics. The registry records observations, unique keys and repeated reuse
events during learning only; frozen evaluation stays read-only. The registry is
bounded by `max_binding_reuse_records_per_channel`, is cleared when a channel is
physically erased, and is serialized in model checkpoints. This bumps the model
checkpoint magic to `SBMCKPT4`.

**2026-06-29 reuse-aware structural value completion:** Topology events now
separate `structural_value_without_reuse` from an optional
`binding_reuse_bonus`. The bonus is derived from the bounded per-channel
binding-reuse registry and is controlled by `binding_reuse_value_weight`, whose
default is zero. Existing acceptance decisions therefore remain compatible
unless a validation run explicitly enables reuse-aware structural admission.
Because topology events are raw-serialized, the model checkpoint magic is now
`SBMCKPT5`.

**2026-06-29 small reuse-aware gate:** A 300k/100k FineWeb-Edu smoke comparison
ran raw-credit admission, cost-only structural admission and reuse-aware
structural admission with `binding_reuse_value_weight=0.05`. The gate validated
stable execution and nonzero reuse bonuses in topology events and program
attribution, but it did not change the accepted topology sequence or validation
NLL in this small window. The result is recorded in
`research_results/reuse_aware_structural_small_seed7.md`.

### Task 7: Introduce Structural Value Gates

**Files:**
- Modify: `src/modules/machine_topology.cpp`
- Modify: `src/modules/token_experiment.cpp`
- Modify: `src/api/c_api.cpp`
- Modify: `scripts/run_corpus_training.py`
- Test: `tests/test_sbm.cpp`
- Test: `tests/test_python_api.py`

- [x] **Step 1: Add failing cost-gate test**

In `tests/test_sbm.cpp`, add:

```cpp
TEST(TopologyLifecycle, AcceptanceReportsStructuralValue) {
    Config config;
    config.objective = ObjectiveKind::TokenCrossEntropy;
    config.token_alphabet = 64;
    config.vector_dim = 64;
    config.adaptive_topology = true;
    config.structural_description_cost_weight = 1.0F;
    config.structural_execution_cost_weight = 0.1F;
    SparseBranchMachine model(config);
    for (std::uint32_t i = 0; i < 8192; ++i) {
        model.step_token(i % 64U, (i + 1U) % 64U, true);
    }
    const auto events = model.topology_events();
    for (const auto& event : events) {
        if (event.decision == TopologyDecision::Accepted) {
            EXPECT_TRUE(std::isfinite(event.credit));
        }
    }
}
```

- [x] **Step 2: Run failing cost-gate test**

Run:

```powershell
cmake --build build-fast --config Release --target sbm_tests
build-fast\sbm_tests.exe --gtest_filter=*AcceptanceReportsStructuralValue*
```

Expected: fail if accepted events do not carry structural-value-compatible
credit after cost penalties.

- [x] **Step 3: Apply value formula to reporting first**

In `maybe_finalize_topology_probe`, keep the existing acceptance threshold for
the first implementation but report:

```cpp
const double structural_value =
    mean_credit
    - static_cast<double>(config_.structural_description_cost_weight) *
      channel_description_cost(channel)
    - static_cast<double>(config_.structural_execution_cost_weight) *
      channel_execution_cost(channel);
```

Store this value in the event credit field for accepted/rejected event JSON.
Do not change acceptance decisions in this task.

- [x] **Step 4: Add an explicit gate flag**

In `Config`, add:

```cpp
bool topology_accept_uses_structural_value{false};
```

Register it in the C API schema. When true, compare `structural_value` to
`topology_accept_credit`; when false, compare old `mean_credit`.

- [x] **Step 5: Verify old compatibility mode**

Run:

```powershell
cmake --build build-fast --config Release
build-fast\sbm_tests.exe
build-fast\sbm_c_api_tests.exe
python tests\test_python_api.py --library E:\SPM\build-fast\libsbm_api.dll
```

Expected: pass with default compatibility decisions unchanged except reported
diagnostic fields.

- [x] **Step 6: Commit**

```powershell
git add include/sbm/types.hpp src/modules/machine_topology.cpp src/modules/token_experiment.cpp src/api/c_api.cpp scripts/run_corpus_training.py tests/test_sbm.cpp tests/test_python_api.py
git commit -m "feat: report structural value for topology decisions"
```

### Task 8: Compatibility And Medium Validation Gate

**Files:**
- Modify: `scripts/run_corpus_training.py`
- Modify: `research_results/`
- Modify: `RESEARCH_LOG.md`
- Test: build and corpus runs

- [x] **Step 1: Run unit and API verification**

Run:

```powershell
cmake --build build-fast --config Release
build-fast\sbm_tests.exe
build-fast\sbm_scaling_tests.exe
build-fast\sbm_c_api_tests.exe
python tests\test_python_api.py --library E:\SPM\build-fast\libsbm_api.dll
git diff --check
```

Expected: all pass.

- [x] **Step 2: Run a short semantic smoke**

Run:

```powershell
python scripts\run_corpus_training.py --library E:\SPM\build-fast\libsbm_api.dll --manifest E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train10m_eval1m\manifest.json --seed 7 --output E:\SPM_EXPERIMENTS\runs\address_semantics_smoke_seed7.json --bucket-bits 14 --adaptive-topology --set address_lags=1 --set topology_enable_delta=false --set max_sparse_decisions_per_node=512 --set classification_learning_rate=0.8 --set classification_mature_learning_rate=0.2 --set record_channel_attribution=true --max-train-examples 1000000 --max-eval-examples 100000
```

Expected:

- JSON contains `address_execution_frames > 0`;
- JSON contains nonzero binding hit/miss counts;
- JSON contains lifecycle counts;
- run completes without noisy multiline progress output.

- [x] **Step 3: Run the 10M/1M gate only after the smoke passes**

Run:

```powershell
python scripts\run_corpus_training.py --library E:\SPM\build-fast\libsbm_api.dll --manifest E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train10m_eval1m\manifest.json --seed 7 --output E:\SPM_EXPERIMENTS\runs\address_semantics_10m_seed7.json --bucket-bits 14 --adaptive-topology --set address_lags=1 --set topology_enable_delta=false --set max_sparse_decisions_per_node=512 --set classification_learning_rate=0.8 --set classification_mature_learning_rate=0.2 --set record_channel_attribution=true
```

Expected:

- do not require immediate NLL improvement as the only success criterion;
- require explicit execution, persistence and attribution diagnostics;
- compare NLL against the current-token and repaired fixed `[1,2,4]` controls;
- record if quality regresses and identify which structural gate still passed.

- [x] **Step 4: Record results**

Create `research_results/address_semantics_10m_seed7.md` with:

```markdown
# Address Semantics 10M Seed 7

## Configuration

- corpus: FineWeb-Edu 10M train / 1M validation manifest
- seed: 7
- address execution: interpreted frames
- accepted-channel retirement: preserve
- attribution: dependency-aware frozen channel/program attribution

## Gates

| gate | pass condition | evidence field |
|---|---|---|
| interpreter frames emitted | `address_execution_frames > 0` | `diagnostics.address_execution_frames` |
| content bindings audited | hit and miss counters are present | `diagnostics.address_binding_hits`, `diagnostics.address_binding_misses` |
| accepted structures persistent | accepted-channel physical prune is absent under preserve policy | `diagnostics.topology_pruned`, `diagnostics.recoverable_retired_channels` |
| dependency attribution emitted | program attribution list is present when attribution is enabled | `eval_program_attribution` |
| bounded active work | active nodes and candidate checks remain within configured bounds | `diagnostics.avg_active`, `diagnostics.max_bucket_candidates_inspected` |
| predictive control | eval NLL is compared against current-token and fixed `[1,2,4]` controls | `eval_nll`, control NLL fields |

## Interpretation

The run validates or rejects the completed address-semantics framework. It is
not a hyperparameter sweep and does not by itself justify R3 scale-up unless
the structural gates and predictive control both pass.
```

- [x] **Step 5: Update research log**

Append a concise entry to `RESEARCH_LOG.md`:

```markdown
## 2026-06-27 — neuronal address semantics framework

The immediate plan was redirected away from fusion/rate sweeps and toward a
complete address-operation framework. Address programs now execute as auditable
frames, accepted structures are preserved by default, and frozen evaluation can
report dependency-aware structural value. The first validation run is recorded
in `research_results/address_semantics_10m_seed7.md`.
```

- [x] **Step 6: Commit**

```powershell
git add scripts/run_corpus_training.py research_results/address_semantics_10m_seed7.md RESEARCH_LOG.md
git commit -m "research: validate address semantics framework"
```

### Task 9: R3 Admission Gates On The Completed Framework

**Files:**
- Modify: `ROADMAP_REAL_DATA.md`
- Modify: `RESEARCH_LOG.md`
- Modify: `ISSUES.md` only if a confirmed blocker appears
- Modify: checkpoint/runtime files only when needed by the gate

The 10M/1M seed-7 gate makes the completed framework admissible for serious
validation, but it does not prove the research claim. The next work must test
stability and reproducibility of the mature address-semantics framework, not
continue local operator or rate tuning.

- [x] **Step 1: Define exact checkpoint/resume contract**

Add a checkpoint contract that serializes enough model state to reproduce
continued training after interruption, including accepted programs, channel
phase, address buckets, sparse output state, learning counters, RNG seeds and
data cursor.

Contract boundary:

- model checkpoints are exact-resume artifacts for the same repository format,
  ABI and platform family; they are not declared as long-term cross-version
  archives;
- the binary file magic is the checkpoint format version and must be bumped
  whenever a raw-serialized state struct changes layout;
- `SBMCKPT5` is the current lineage-aware model checkpoint format. It includes
  channel parent/dependency lineage in topology state and topology events,
  binding-kind execution counters, bounded binding-reuse registries,
  reuse-aware topology event value fields, plus all state required to continue
  sparse-output learning exactly;
- runner checkpoints own the data cursor, manifest identity, phase, accumulated
  metrics and path to the model checkpoint. Exact long-run resume requires both
  runner and model checkpoint files;
- verification requires an uninterrupted model and a save/load/resume model to
  match step-by-step on continuation, including predictions, live-node count,
  learned programs, channel phases and channel lineage.

- [x] **Step 1a: Implement exact model-state checkpoint**

`save_checkpoint` and `load_checkpoint` now serialize the full
`SparseBranchMachine` training state: config, topology lifecycle, proposal
cursor, topology events, node arrays, sparse output tables, sparse admission
tables, global output priors, edges, bucket directory, route trace, history and
diagnostic counters. The regression test saves after 512 token steps, reloads
and verifies that 256 more training steps match an uninterrupted model
step-by-step.

- [x] **Step 1b: Implement corpus training-run checkpoint**

The model checkpoint is not sufficient for unattended long runs. The first
runner-level implementation is now `scripts/run_resumable_corpus_training.py`.
It records manifest identity, train shard paths, current shard, cursor position,
examples consumed, accumulated train metrics, config overrides, output path and
the associated model checkpoint. The regression test interrupts at 24 examples,
resumes to 48 examples and checks that the final train metrics match an
uninterrupted run on the same mapped shard.

- [x] **Step 1c: Integrate resumable evaluation and final report assembly**

The current resumable runner solves the training replay problem but does not
yet replace `run_corpus_training.py` for every historical field, but it now
resumes both train and eval phases. It records eval shard identity, eval cursor,
strict-freeze transition state, baseline/control metrics, topology summary and
program-attribution result assembly. The output keeps top-level resumable-run
metadata and embeds a canonical-style `result` object for experiment consumers.

Verification target:

```powershell
build-fast\sbm_tests.exe --gtest_filter=*Checkpoint*
python tests\test_python_api.py --library E:\SPM\build-fast\libsbm_api.dll
python tests\test_resumable_runner.py --library E:\SPM\build-fast\libsbm_api.dll
```

Expected: uninterrupted training and checkpoint/resume training produce matching
or explicitly bounded metrics on a deterministic corpus slice.

- [x] **Step 2: Run multi-seed 10M/1M validation**

Run the completed address-semantics configuration on at least three seeds over
the same 10M/1M manifest. Compare against current-token, interpolated
multiscale and repaired fixed `[1,2,4]` controls.

Completed for seeds 7, 11 and 19. All three seeds beat the current-token
control, accepted five topology programs, physically pruned zero accepted
structures and emitted 10/10 positive-mean program-attribution entries. The
result is recorded in `research_results/address_semantics_10m_multiseed.md`.

Acceptance boundary:

- at least two of three seeds beat the current-token control;
- no seed loses without a clear structural diagnostic explanation;
- attribution remains positive for accepted content programs;
- accepted structures are not physically pruned under preserve policy.

- [x] **Step 3: Run shard-transfer validation**

Repeat the gate on a different FineWeb-Edu shard sample. This tests whether the
accepted address semantics survive corpus sampling rather than fitting one
manifest.

The execution harness now supports this as a parallel batch rather than a
serial seed loop: `scripts/run_corpus_batch.py` accepts eval split selection,
train/eval limits, runtime parameter overrides, skip-existing resume behavior
and worker memory estimates. Throughput checks must use
`scripts/run_throughput_probe.py`, which runs short C++ limited windows and
stops when the recent window speeds stabilize, instead of forcing a fixed 1M
probe before reporting.

Completed on the held-out test split for seeds 7, 11 and 19. All three seeds
beat the current-token and interpolated controls, accepted five programs,
physically pruned zero accepted structures and emitted 10/10 positive-mean
program-attribution entries. The result is recorded in
`research_results/address_semantics_10m_test_multiseed.md`.

The first local attempt with three parallel workers was stopped because it was
memory-bound rather than CPU-bound. Worker resident memory left only about 1 GB
available and CPU utilization stayed below 30%. The completed gate used two
parallel workers, which is the current local saturation point for this workload.

Acceptance boundary:

- quality remains competitive with current-token control;
- learned programs need not be identical, but program attribution must identify
useful content-conditioned structures;
- bounded-work diagnostics remain within configured limits.

- [x] **Step 4: Calibrate structural-value admission**

Only after Steps 1-3, evaluate whether `topology_accept_uses_structural_value`
can become an admissible gate. This is not a free hyperparameter sweep: compare
raw-credit acceptance and structural-value acceptance under matched seeds and
record when cost penalties reject programs with positive predictive value.

Completed for the default description-only gate. Matched 10M/1M validation
runs with `topology_accept_uses_structural_value=true` learned the same
operation/program sequence as the raw-credit gate for seeds 7, 11 and 19:
`[0, 0, 0, 2, 3, 0]` over `[[1], [2], [1, 2], [2], [1, 2], [3]]`. Each run
accepted five programs, pruned zero accepted structures and retained 10/10
positive-mean program-attribution entries. Mean NLL changed by +0.000148
nats/token versus the raw gate. The result is recorded in
`research_results/structural_value_admission_10m_validation.md`.

Nonzero `structural_execution_cost_weight` remains uncalibrated and must not be
enabled by default from this gate.

- [x] **Step 5: Commit each completed gate**

Each gate must end with a research result file, a concise `RESEARCH_LOG.md`
entry and either a commit or a confirmed blocker in `ISSUES.md`.

Completed. Task 9 gates were committed as:

- `b85722a research: validate address semantics multiseed`
- `476445c research: validate address semantics transfer`
- `60b4456 research: calibrate structural value admission`

## Execution Policy

Implement tasks in order. Each task must end with either a commit or an explicit
blocker recorded in `ISSUES.md`. Do not start a medium or large corpus run until
Tasks 1 through 7 are committed and all listed tests pass.

Do not add new operators beyond Tuple, DeltaMod, ContentMatch and ContentFollow
inside this plan. The objective is to mature execution semantics, lifecycle and
attribution for the existing promising primitives.

Do not tune `classification_learning_rate`, `residual_channel_gain`,
`topology_accept_credit`, sparse-output capacity, bucket limits or beam width as
a substitute for completing the framework. Those parameters may be changed only
inside a named validation gate after the semantic contract is implemented.

## Self-Review

- Spec coverage: the plan covers explicit address execution, persistent
  accepted structures, dependency-aware attribution, structural value accounting,
  documentation reframing and medium validation.
- Placeholder scan: the plan contains no open-ended implementation placeholder;
  every task has concrete files, commands and expected outcomes.
- Type consistency: new public names are introduced before use:
  `AddressExecutionFrame`, `AddressExecutionMode`,
  `AcceptedChannelRetirement`, `AddressBindingKind` and
  `ProgramAttribution`.
