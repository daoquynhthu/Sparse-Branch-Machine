# Address Semantics 10M Seed 7

## Configuration

- corpus: FineWeb-Edu 10M train / 1M validation manifest
- train examples: 9,989,918
- eval examples: 998,482
- seed: 7
- address execution: interpreted frames
- accepted-channel retirement: preserve
- attribution: dependency-aware frozen channel/program attribution
- output: `E:\SPM_EXPERIMENTS\runs\address_semantics_10m_seed7.json`

## Quality

| metric | value |
|---|---:|
| train NLL | 6.460424 |
| eval NLL | 6.366367 |
| current-token baseline NLL | 6.499756 |
| interpolated multiscale baseline NLL | 6.768842 |
| repaired fixed `[1,2,4]` R1 NLL | 6.476030 |
| seed-only eval NLL | 6.599492 |
| active-channels-only eval NLL | 6.370817 |
| content-channels-only eval NLL | 6.412807 |
| tuple-channels-only eval NLL | 6.454273 |

## Gates

| gate | status | evidence |
|---|---:|---|
| interpreter frames emitted | PASS | `address_execution_frames = 65,858,720` |
| content bindings audited | PASS | `address_binding_hits = 43,990,050`, `address_binding_misses = 21,868,670` |
| accepted structures persistent | PASS | `topology_accepted = 5`, `topology_pruned = 0`, `quarantined_channels = 0`, `recoverable_retired_channels = 0` |
| dependency attribution emitted | PASS | `eval_program_attribution` entries = 10 |
| bounded active work | PASS | `avg_active = 5.99994`, `max_bucket_candidates_inspected = 8` |
| predictive control | PASS | eval NLL 6.366367 beats current-token 6.499756 and repaired fixed `[1,2,4]` 6.476030 |

## Topology

- learned operations: `[0, 0, 0, 2, 3, 0]`
- learned programs: `[[1], [2], [1, 2], [2], [1, 2], [3]]`
- topology events: 10
- live nodes: 193,460
- estimated state: 1,921,576,328 bytes
- throughput: 9,671 examples/s

## Program Attribution

The strongest full-document program credits were:

| channel | op | lags | dependency | mean credit | positive fraction |
|---:|---:|---|---:|---:|---:|
| 4 | 3 | `[1,2]` | 0 | 0.105742 | 0.638135 |
| 3 | 2 | `[2]` | 0 | 0.107042 | 0.637733 |
| 2 | 0 | `[1,2]` | 2 | 0.087780 | 0.622560 |
| 0 | 0 | `[1]` | 1 | 0.084898 | 0.621097 |
| 1 | 0 | `[2]` | 2 | 0.079916 | 0.618261 |
| 5 | 0 | `[3]` | 3 | 0.076957 | 0.618450 |

With the current default `structural_description_cost_weight = 1.0`, reported
`structural_value` is dominated by accumulated description cost. This is useful
as an auditable cost ledger but is not yet the default acceptance rule;
`topology_accept_uses_structural_value` remains false by default.

## Interpretation

This run validates the completed address-semantics framework on the 10M/1M
gate. The model now executes address programs as auditable frames, preserves
accepted structures by default, emits dependency-aware program attribution and
beats both the current-token control and the previous repaired fixed
`[1,2,4]` R1 result.

This is the first clean evidence in this branch that the completed adaptive
content-addressing framework can outperform the strong short-context controls
at the medium scale. R3 scale-up is still gated on checkpoint/resume and
multi-seed/shard confirmation.
