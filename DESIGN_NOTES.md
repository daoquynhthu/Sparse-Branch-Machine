# Design notes and current findings

## Architectural invariant

The machine is designed around:

- a potentially large persistent address space;
- a fixed candidate budget;
- a fixed active route width;
- local state and edge updates only;
- stable logical references despite physical compaction.

The intended scaling property is that total stored capacity may grow while average active work remains bounded.

## v2 storage model

`NodeId` is a monotonic logical address. `id_to_slot` maps it to a movable physical slot. Node metadata is stored in structure-of-arrays form. Pruning uses swap-removal, updates the moved logical mapping, and rebuilds bucket indices. Edges retain logical IDs, so physical movement does not rewrite the graph.

## Hierarchical hot/cold addressing

A flat address bucket was not robust: 100,000 irrelevant prefetched nodes reduced frozen evaluation accuracy to about 78%, although throughput remained stable. v2 therefore searches:

1. learned control edges;
2. hot bucket entries that have actually participated in learning;
3. the cold global bucket.

With this separation, the same 100,000-node stress test recovered 100% evaluation accuracy in the recorded run. At one million distractors, average candidates remained 24 and average active nodes remained 4; evaluation accuracy was 96.875%.

This is evidence that bounded work is feasible, not evidence of general intelligence or superiority over dense models.

## Hard unresolved problems

- Generalization beyond finite-context recurrence.
- Counterfactual credit for nodes that should have been selected but were not.
- Formation and reuse of compositional subprograms.
- Long-horizon routing without beam explosion.
- Learned physical locality and NUMA placement.
- Fair quality-to-energy comparison against neural baselines.
