# Address Semantics Smoke Seed 7

## Configuration

- corpus: FineWeb-Edu 10M/1M manifest, limited runner
- train examples: 1,000,000
- eval examples: 100,000
- seed: 7
- address execution: interpreted frames
- accepted-channel retirement: preserve
- attribution: frozen channel/program attribution enabled
- output: `E:\SPM_EXPERIMENTS\runs\address_semantics_smoke_seed7.json`

## Gates

| gate | status | evidence |
|---|---:|---|
| interpreter frames emitted | PASS | `address_execution_frames = 6,528,320` |
| content bindings audited | PASS | `address_binding_hits = 4,371,408`, `address_binding_misses = 2,156,912` |
| accepted structures persistent | PASS | `topology_accepted = 5`, `topology_pruned = 0`, `quarantined_channels = 0`, `recoverable_retired_channels = 0` |
| dependency attribution emitted | PASS | `eval_program_attribution` entries = 10 |
| bounded active work | PASS | `avg_active = 5.9994`, `max_bucket_candidates_inspected = 8` |
| smoke predictive control | PASS | eval NLL 6.9104 vs current-token 7.8332 and interpolated multiscale 8.1617 |

## Topology

- learned operations: `[0, 0, 0, 2, 3, 0]`
- learned programs: `[[1], [2], [1, 2], [2], [1, 2], [3]]`
- topology events: 10

## Interpretation

The completed address-semantics infrastructure is active in a real corpus run:
token steps execute address frames, bindings are audited, accepted structures
are preserved under the default policy, and frozen evaluation emits
program/dependency attribution.

This is a smoke gate, not the full 10M/1M validation gate. It validates that the
framework can run at useful throughput and produce the new diagnostics. The
full gate still has to compare against the repaired 10M fixed `[1,2,4]` control.
