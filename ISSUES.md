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

No active P0 issue remains after the 2026-06-27 channel-local token residual
repair and the repaired 10M/1M R1 run.

## P1: diagnostic and recovery blockers

### P1-4: R2 accepted-structure attribution gap

The repaired R1 run passes the short-context predictive gate, but the current
evidence does not satisfy the R2 structural hard gate. The successful medium
run used fixed programs `[1]`, `[2]` and `[4]`; it did not contain an accepted
adaptive structure whose held-out codelength contribution can be traced across
multiple documents.

The offline R2 structural-control report strengthens the blocker. On the
10M/1M FineWeb-Edu split, a frozen current-token conditional table reached NLL
6.4048, while fixed lag1/lag2/lag4 controls reached 7.1816, 7.3531 and 7.4723.
The lag1/2/4 interpolation reached 6.5653, losing 0.1604 nats/token against
current-token conditioning, and only 143 of 877 validation documents had
positive current-minus-interpolated gain.

Failure mode: if this remains unresolved, the R1 model improvement can be
reported as predictive progress but cannot be interpreted as evidence for the
intended address-structure mechanism. Advancing directly to larger R3 runs
would make attribution worse, not better.

Fix plan:

- add frozen-evaluation attribution for accepted programs/channels, reporting
  per-structure codelength deltas against parent/current residual paths;
- run an adaptive R2 validation block with content-addressing proposals enabled
  and a separately sampled validation block;
- require at least one accepted structure to show positive held-out codelength
  contribution across multiple documents before moving this issue to resolved.

Status: frozen-evaluation per-channel attribution is implemented behind the
explicit `record_channel_attribution` switch, including token-level credit and
per-document positive contribution counts. The issue remains active until an
adaptive R2 run produces accepted-structure attribution on held-out validation
and an independent validation block confirms the result.

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

### P0-7: Token multi-channel residual dilution

Resolved by training dense and sparse token residual logits in channel-local
coordinates. Inference still aggregates local logits with global node
responsibility, but updates now use responsibility normalized by the active
mass assigned to the represented channel.

The fix repaired the minimal 1M FineWeb-Edu reproductions. Fixed
`address_lags=1,2,4` at cap64 improved from NLL 7.3768 to 7.2115, beating the
lag-1 cap64 control at 7.2709. With cap512 and high token learning rates,
fixed `1,2,4` improved from 7.1667 to 6.9122, beating the lag-1 high-rate
control at 6.9911. The content-focused adaptive smoke improved from 7.4110 to
7.1621, while accepting the same `ContentMatch([2])` and `ContentFollow([1,2])`
channels with larger positive credit.

### P0-6: Real-data R1 short-context control failure

Resolved by the channel-local token residual repair plus the repaired
fixed-multichannel run. On the admissible FineWeb-Edu 10M train / 1M validation
corpus, seed 7 with fixed `address_lags=1,2,4`, cap512 and high token learning
rates reached validation NLL 6.4760. This beats unigram 7.5085, interpolated
multiscale 6.7688 and the previous strongest current-token control 6.4998.
The run processed 10,988,400 examples at 4.35k examples/s with 1.20 GB
estimated state, 112,500 live nodes and strict frozen evaluation.

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
