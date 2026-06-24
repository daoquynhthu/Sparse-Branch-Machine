# Current implementation issues

> **Document role:** This is the authoritative queue of confirmed defects and
> experiment blockers in the current implementation. It does not duplicate
> staged roadmap work or unresolved theory. An item belongs here only when the
> current code or current experimental interpretation is already affected.

## Priority semantics

- **P0:** blocks interpretation of the affected experiment class or the current
  predictive-learning gate.
- **P1:** blocks diagnosis, recovery correctness or a justified parameter
  decision.
- **P2:** real defect with limited current-platform impact.

## P0: predictive-learning blockers

### P0-1: No shared global output prior

**Confirmed implementation fact.** Sparse token prediction is only the weighted
sum of decision logits stored by active address nodes. A missing local decision
contributes zero logit, and there is no global decision table shared across
addresses (`SparseBranchMachine::aggregate_sparse_logit` in
`src/modules/token_sparse_output.cpp`). New address nodes therefore restart from
an approximately uniform hierarchical distribution rather than a learned corpus
base distribution.

This is the leading falsifiable explanation for the medium-corpus result
(`eval NLL 10.6696` versus unigram `7.6826`), not a proven sole cause.

**Required gate:** add a bounded global hierarchical base distribution and
represent local node output as a residual. With fixed topology and control-edge
effects disabled, the model must at least match the online unigram control. If
it does not, reject this causal explanation and continue diagnosis.

### P0-2: Sparse decisions inherit node-wide learning-rate decay

**Confirmed learning-semantics defect.** `SparseOutputEntry` stores only
`decision` and `logit` (`include/sbm/machine.hpp`). During sparse token updates,
every path decision uses a schedule derived from the node-wide
`address_visits_[slot]` (`SparseBranchMachine::step_token_sparse` in
`src/modules/token_sparse_output.cpp`). A decision first observed late in a
mature node therefore receives the same heavily decayed rate as frequently
trained decisions.

**Required gate:** maintain bounded per-decision update evidence and show that a
new decision in a mature node learns at the declared fresh-decision rate without
changing established decisions or violating the per-node capacity bound.

### P0-3: Sparse-decision eviction uses parameter magnitude as importance

**Confirmed capacity-management defect.** At capacity,
`mutable_sparse_logit` removes the entry with minimum `abs(logit)`. A frequent,
well-calibrated binary decision can legitimately have a logit near zero, so
parameter magnitude is not evidence of coding value or dispensability.

**Required gate:** replacement must use measured evidence such as cumulative
coding benefit, usage and reconstruction cost. Compare against the current
magnitude policy at identical capacity and active work; rollback if held-out NLL
or churn worsens.

### P0-4: Output-tree randomization is coupled to the model seed

**Confirmed reproducibility defect.** `SparseBranchMachine` constructs
`ImplicitOutputTree(config_.vector_dim, config_.seed)` in
`src/modules/machine_storage.cpp`. Multi-seed runs therefore change both model
randomness and the hierarchical class decomposition, so their variance does not
isolate training stochasticity.

**Required gate:** introduce a separately reported `output_tree_seed`, fixed by
the tokenizer or experiment manifest, while retaining `seed` for model
stochasticity. A test must prove identical token paths across model seeds.

### P0-5: Cross-channel output mass is not conserved

**Confirmed for multi-channel execution.** Responsibility normalization is
performed independently per channel. Channel 0 receives mass `1`, while every
additional enabled channel receives `residual_channel_gain`
(`assign_responsibilities` in `src/modules/machine_routing.cpp`). Total output
scale consequently changes with channel count, mixing information gain with
confidence rescaling and contaminating topology credit.

This issue blocks adaptive/multi-channel topology interpretation. It does not
explain the fixed single-channel medium-corpus failure.

**Required gate:** separate channel information contribution from aggregate
logit scale and verify calibration at fixed information while varying enabled
channel count.

## P1: diagnostic and recovery blockers

### P1-1: Sparse-output saturation and churn are under-instrumented

Current diagnostics report total entries and maximum entries per node, so the
system is not completely blind. They do not report eviction count, rejected or
recreated decisions, the fraction of nodes at capacity, or retention by
frequency. The medium run averaged roughly 59 entries per node against a hard
limit of 64, making those omissions material.

**Required gate:** add zero-cost-disabled or bounded counters before changing
the default capacity. Diagnostics must distinguish stable near-capacity storage
from repeated eviction/reconstruction churn.

### P1-2: Address-bucket capacity pressure is not observable

Node creation is suppressed when `exact_count` reaches
`max_specializations_per_bucket` in all objective paths, but diagnostics expose
only global creation totals and maximum lookup candidates. They do not report
full-bucket fraction, capacity-blocked specialization attempts or per-bucket
context diversity.

**Required gate:** instrument capacity pressure before increasing the bucket
limit or interpreting the final node count as natural convergence.

### P1-3: Invalid token-shard cursors are silently advanced

`MappedTokenShard::next` rejects a mismatched shard ID, but an out-of-range
`token_offset` is treated like sequence exhaustion and silently advances to the
next sequence (`src/modules/token_shard.cpp`). This can hide corrupted resume
state in the public cursor API.

**Required gate:** reject impossible `sequence_index`/`token_offset`
combinations. End-of-shard must remain representable explicitly, and malformed
cursor tests must fail loudly rather than skip examples.

## P2: portability and documentation defects

### P2-1: C++ shard writer assumes a little-endian host

The mapped reader explicitly rejects non-little-endian hosts, but
`write_token_shard` writes `sequence_offsets` and `tokens` directly from host
memory while declaring the format little-endian. On a big-endian host it would
emit an invalid file with a valid little-endian marker.

**Required gate:** either reject non-little-endian hosts before writing or encode
payload values explicitly. Reader and writer platform policy must match.

### P2-2: Canonical status documents contradict the current repository state

`Agent.md` and `THEORY_ALIGNMENT.md` still name absence of a cloud real-language
corpus as the current blocker. `README.md` first states that versioned mapped
shards exist, then later calls versioned shards and streaming next-phase
deliverables. `ROADMAP_REAL_DATA.md` has a current status update but retains
older ingestion text that describes the implemented mapped corpus layer as the
next milestone.

**Required gate:** reconcile current status without weakening the distinction
between the `compatibility_only` artifact and an admissible R0 corpus. The active
blocker is predictive learning; strict corpus provenance remains a prerequisite
for positive claims.

## Queue discipline

This file intentionally excludes capabilities that are merely scheduled in
`ROADMAP_REAL_DATA.md` and open research questions owned by
`THEORY_ALIGNMENT.md`. Those items enter this queue only after an implemented
contract exists and the current code violates it, or when they become necessary
to interpret an already-running experiment.
