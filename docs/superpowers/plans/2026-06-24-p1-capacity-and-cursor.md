# P1 Capacity and Cursor Repair Plan

1. [completed] Run a controlled 64/128/256 sparse-output capacity curve on identical
   train/eval shards and seed. Select a default only when churn reduction is
   supported by quality, memory and throughput evidence. Record the artifact,
   update P1-1 and commit.
2. [completed] Add failing tests for address-bucket pressure counters. Instrument occupied,
   full, maximum-resident and capacity-blocked specialization metrics in all
   objective paths and serializers. Run a pressure experiment, decide whether
   the cap needs adjustment, update P1-2 and commit.
3. [completed] Replace the permissive malformed-cursor test with native and public API
   rejection tests. Enforce the cursor contract in `MappedTokenShard::next`,
   verify end-of-shard behavior, update P1-3 and commit.
4. Run the complete native and Python verification suite, reconcile `ISSUES.md`
   and the progress/research records, then commit the P1 validation evidence.
