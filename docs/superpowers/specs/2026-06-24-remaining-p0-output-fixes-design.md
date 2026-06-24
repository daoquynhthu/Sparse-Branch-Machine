# Remaining P0 Output Fixes Design

## Scope and hypothesis

This change resolves active issues P0-2, P0-3 and P0-5. It does not change the
global prior, corpus, output tree, address language or topology proposal policy.

The hypothesis is that local residual decisions currently underlearn when first
encountered in mature nodes, then churn under a magnitude-only capacity policy.
For adaptive topology, independently normalized channel masses additionally
confound information value with output scale.

## Per-decision state and learning

Each bounded local sparse decision stores:

```cpp
struct SparseOutputEntry {
    std::uint32_t decision;
    float logit;
    std::uint32_t visits;
    float gain_ema;
    std::uint64_t last_update_step;
};
```

The learning-rate schedule uses `entry.visits`, never node-wide
`address_visits`. Decisions with fewer than `mature_visits` use
`classification_learning_rate`; mature decisions use
`classification_mature_learning_rate`. The schedule remains
`1/sqrt(visits + 1)` and responsibility-scaled.

Before updating a local residual, compute its branch-level coding gain by
comparing the combined branch loss with and without that node's local residual.
Update `gain_ema` with decay `0.99`, clamp one-step gain to `[-1,1]`, increment
visits and record `total_steps`.

## Evidence-based eviction

At fixed capacity, entries updated in the last 32 model steps are in probation.
If any non-probation entry exists, only those are eviction candidates. Otherwise
all entries are candidates. Retention score is:

```text
max(gain_ema, 0) * sqrt(visits + 1) + 1e-4 * log1p(visits)
```

The first term retains demonstrated coding benefit; the second is a small
bounded reconstruction-cost proxy. Lowest score is evicted, ties choose oldest
update then largest decision ID. This is deterministic and does not use logit
magnitude.

Each node owns a 64-bit bounded bloom mask of previously evicted decision IDs.
It supports a `probable_reconstruction` diagnostic without unbounded history.
Diagnostics report insertions, evictions, probable reconstructions, saturated
nodes and maximum per-decision visits.

Merge combines matching entries by visit-weighted logit and gain, sums visits
with saturation and keeps the latest update step. Capacity pressure caused by a
merge follows the same eviction rule and remains observable.

## Conserved cross-channel mass

Only channels represented in the active route participate. Their raw weights
are `1` for the seed channel and `residual_channel_gain` for every other enabled
channel. Divide each raw weight by their sum before exact/non-exact subdivision.
Thus all node responsibilities sum to one regardless of channel count.

Diagnostics track the maximum absolute responsibility-mass error. Single-channel
execution remains exactly unchanged. Multi-channel topology credit continues to
remove one channel's local contribution while retaining global prior and other
channels.

## Acceptance gates

1. A new decision in a mature node learns at the fresh-decision rate relative
   to a prior-only control.
2. Per-entry visits, not node visits, trigger the mature rate.
3. A high-evidence near-zero-logit entry survives a magnitude-only adversarial
   capacity fixture.
4. Eviction and probable-reconstruction diagnostics become nonzero in a forced
   churn fixture; saturation count is exact.
5. Sparse entries never exceed `max_sparse_decisions_per_node`.
6. Total responsibility mass differs from one by at most `1e-6` for one through
   six represented channels.
7. Single-channel fixed-topology smoke NLL remains within `0.5%` of 7.60693 on
   the medium compatibility fixture; active work remains bounded.
8. Full C++, C and Python tests pass before each implementation commit.

## Rollback

Keep diagnostics even if the new eviction policy fails. Roll back per-decision
learning or eviction independently if three-seed held-out NLL worsens by more
than `0.5%`, sparse churn increases without quality gain, or model bytes exceed
the documented entry-size increase. P0-5 is accepted on correctness and
calibration gates, then evaluated separately with adaptive topology.
