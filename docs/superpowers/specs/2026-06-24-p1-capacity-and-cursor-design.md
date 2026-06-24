# P1 Capacity and Cursor Repair Design

## Scope

This stage resolves the three active P1 defects without changing the model's
addressing or sparse-output semantics.

## Sparse-output capacity

The current 64-entry default is not accepted merely because it is bounded. A
capacity curve at 64, 128 and 256 entries must measure validation NLL,
throughput, estimated bytes, evictions per training token, probable
reconstructions per training token and saturated-node fraction on the same
shards and seed.

The selected default must materially reduce churn without a disproportionate
memory or throughput regression. If no fixed point does so, the stage must not
hide the result by selecting the largest cap; it must retain P1-1 and implement
a global-budgeted adaptive policy separately.

## Address-bucket capacity

Diagnostics will expose occupied buckets, full buckets, maximum residents and
capacity-blocked specialization attempts. A blocked attempt is counted only
when the conflict and persistence gates would create a specialization but the
bucket cap alone prevents creation.

The bucket default changes only if a representative run shows material blocked
specialization pressure. Otherwise the issue closes as observable and not
currently blocking, with the default unchanged.

## Token-shard cursor contract

A valid cursor is one of:

- an in-shard sequence index with `token_offset <= sequence_length - 1`;
- the explicit end cursor `{shard_index, sequence_count, 0}`.

Sequence indices beyond the shard, nonzero offsets at end-of-shard and offsets
beyond the final valid transition position are malformed and must throw. The C
API must translate the exception to its normal error contract and Python must
raise `SBMError`.

## Verification gates

- Capacity evidence is stored as a reproducible JSON artifact.
- Address counters have focused saturation tests and appear in both result
  serializers.
- Native and public API tests reject malformed cursors while accepting the
  explicit end cursor.
- The full native and Python test suite passes after each completed stage.

