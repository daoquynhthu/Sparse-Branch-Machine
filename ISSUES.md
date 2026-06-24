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

No active P0 issue remains after the 2026-06-24 output-learning repairs.

## P1: diagnostic and recovery blockers

No active P1 issue remains after the 2026-06-24 capacity, diagnostics and
cursor repairs.

## P2: portability and documentation defects

No active P2 issue remains after the 2026-06-24 shard portability and canonical
status repairs.

## Queue discipline

This file intentionally excludes capabilities that are merely scheduled in
`ROADMAP_REAL_DATA.md` and open research questions owned by
`THEORY_ALIGNMENT.md`. Those items enter this queue only after an implemented
contract exists and the current code violates it, or when they become necessary
to interpret an already-running experiment.

## Resolved

### P2-2: Canonical current-state documentation

Resolved by reconciling the canonical documents with the implemented mapped
shards, manifest-owned multi-shard execution, strict data cursors and P0/P1
learning repairs. The compatibility corpus remains explicitly non-admissible
for R0 because document, source, tokenizer and deduplication provenance are
missing. Model checkpoint serialization remains unimplemented.

### P2-1: Portable little-endian shard writer

Resolved with buffered explicit little-endian encoding for sequence offsets
and token IDs. A byte-level regression covers 32-bit and 64-bit values, while
existing deterministic-file and mapped-read tests cover the complete shard.
The mmap reader continues to reject big-endian hosts explicitly.

### P1-3: Strict token-shard cursor validation

Resolved in the mapped shard reader and covered through C++, C and Python.
Sequence indices beyond the shard, offsets beyond the final transition and
nonzero offsets at end-of-shard now fail before mutating the cursor. The
explicit `{shard_index, sequence_count, 0}` end cursor remains valid.

### P1-2: Address-bucket capacity pressure diagnostics

Resolved by occupied/full/max-resident and capacity-blocked split diagnostics
in all objective paths. A cap 4/8/16 smoke found that larger caps reduced but
did not remove hotspot blocking, worsened NLL from 8.1694 to 8.1744/8.1789 and
reduced throughput from 16.96k to 10.15k/7.29k steps/s. The default remains 4;
hotspot conflict policy is a research question, not a hidden capacity limit.

### P1-1: Evidence-gated sparse-output admission

Resolved by the bounded two-hit admission table. A 64/128/256 capacity curve
showed that cap growth alone retained 14.4-18.5 evictions per token while
roughly doubling state and reducing throughput. At cap 64, admission reduced
evictions from 23.26 to 1.09 per token and probable reconstructions from 22.73
to 0.864 on the 998,400/99,328 seed-7 gate. Validation NLL improved from
7.54625 to 7.51149. Rejections and promotions are explicit diagnostics.

### P0-1: Shared global output prior

Resolved by `f9a2ab1` and optimized by `83fd3d3`. A Jeffreys-smoothed global
hierarchical count prior now supplies the base logit; address-local sparse
outputs are residuals. Counts update only after prediction and remain frozen in
evaluation. On the 1M/0.1M compatibility run, mean validation NLL changed from
10.6696 to 7.6069 versus unigram 7.6826 across seeds 7, 11 and 19.

### P0-4: Output-tree seed coupling

Resolved by `6e56b67`. `output_tree_seed` is independent of model `seed`, is
reported through the runtime/API result, and has cross-model-seed regression
coverage.

### P0-2: Per-decision learning statistics

Resolved by `1d5bfd7`. Each sparse decision now owns visits, coding-gain EMA and
last-update step. Fresh decisions in mature nodes use the fresh-decision rate;
the mature schedule depends on decision visits rather than node visits.

### P0-3: Evidence-based sparse-decision eviction

Resolved by `1d5bfd7`. Eviction now uses coding gain, visit evidence and bounded
probation with deterministic ties, not logit magnitude. Mean 1M/0.1M NLL
improved from 7.60693 to 7.54645. The later P1-1 admission repair reduced the
remaining reconstruction churn without changing this eviction policy.

### P0-5: Conserved cross-channel output mass

Resolved by `ed89a72`. Represented channel weights are normalized globally
before within-channel routing. Regression and multi-channel calibration keep
total responsibility error below `1e-6`, while single-channel behavior remains
unchanged.
