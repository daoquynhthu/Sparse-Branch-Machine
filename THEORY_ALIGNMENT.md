# Theory alignment branch

This branch starts from commit `b47713b` and changes the address topology from a
fixed list of temporal views into a learned structural object.  It is the first
branch whose purpose is not merely better task fit or engineering hygiene, but
alignment with the original hypothesis:

```text
prediction pressure -> sparse structural proposal -> local adaptation
                    -> counterfactual validation -> accept or erase
```

## What is now learned

The model begins with one seed address channel, normally lag 1.  A minimal
meta-rule proposes one unused temporal lag at a time.  A proposal is a real
address channel with its own nodes, buckets and local logits, but its lifecycle
is explicit:

1. **Probe adaptation** — the candidate learns locally for a bounded interval.
2. **Frozen validation** — candidate parameters stop changing during the final
   part of the interval.
3. **Channel ablation** — the model computes the exact loss difference between
   the full route and the same route with the entire candidate channel removed.
4. **Selection** — only positive mean validation credit can retain the channel.
5. **Mature audit** — accepted non-seed channels continue to receive
   counterfactual credit and can later be retired if their sustained contribution
   becomes sufficiently negative.
6. **Physical reclamation** — rejection or retirement deletes all channel nodes,
   rebuilds indexes and removes stale route state while preserving other logical
   node IDs.

The fixed `address_lags` parameter still exists, but it now means **seed
structure**, not immutable final topology.  Setting `adaptive_topology=false`
recovers the previous fixed-channel behavior for controls and regression tests.

## Structural credit

For token cross-entropy, channel credit is computed by exact channel-level
ablation:

\[
C_c = L(z - z_c, y) - L(z, y),
\]

where \(z_c\) is the sum of all active logits contributed by channel \(c\).
This avoids approximating a channel by the sum of independent node ablations.

For vector regression the same idea is used with normalized MSE:

\[
C_c = \operatorname{NMSE}(\hat y-\hat y_c,y)
      -\operatorname{NMSE}(\hat y,y).
\]

A positive value means that removing the channel makes the current prediction
worse.

## Auditability

Every structural decision is recorded as a `TopologyEvent` containing:

- global training step;
- proposed lag;
- decision (`Proposed`, `Accepted`, `Rejected`, `Pruned`);
- validation or mature credit at the decision.

The event list is included in experiment JSON.  This makes topology evolution a
replayable research object rather than hidden mutable state.

## Current result

On the default mathematical token task, three seeds produce:

| topology | mean frozen NLL |
|---|---:|
| adaptive from seed `[1]` | 3.41847 |
| fixed `[1,2,4]` | 3.41132 |
| unigram baseline | about 3.4658 |
| generator oracle | about 2.9143 |

All three adaptive runs finish with `[1,2]`.  The system proposes lag 4 but
rejects it under its current local learner and validation criterion.  This is a
real remaining failure, not hidden by forcing the known generator structure into
the model.

The branch therefore demonstrates **learned topology lifecycle**, but not yet
successful recovery of all useful mathematical dependencies.

## What remains outside the theory target

- Proposal generation still enumerates bounded temporal lags; it does not yet
  synthesize arbitrary address programs.
- Nodes still store dense local logits or vectors rather than learned executable
  operators.
- Structural credit is local to one channel and does not yet compare multi-step
  topology edits.
- Accepted channels use the same residual-composition semantics; the role of a
  channel is not yet learned.
- Control edges select nodes but do not transform or bind values.

The next theoretical step should be a small address-program language whose
primitive selectors and compositions can be proposed and validated by the same
lifecycle, rather than adding more hand-written lag types.
