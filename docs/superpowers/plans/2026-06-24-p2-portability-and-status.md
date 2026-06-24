# P2 Portability and Status Repair Plan

1. Add a failing byte-level test for explicit little-endian scalar-array
   encoding. Implement a bounded stream encoder and route token shard offsets
   and token IDs through it. Verify deterministic shard bytes and mapped reads,
   resolve P2-1 and commit.
2. Audit `Agent.md`, `README.md`, `THEORY_ALIGNMENT.md`,
   `ROADMAP_REAL_DATA.md` and `ISSUES.md` against the implemented mapped corpus,
   P0/P1 repairs and compatibility-only data status. Remove contradictory
   current-state claims without weakening corpus provenance gates, resolve P2-2
   and commit.
3. Run all native/Python checks plus `git diff --check`, close the plan and
   commit final validation state.
