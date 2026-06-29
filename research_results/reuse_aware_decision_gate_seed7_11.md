# Reuse-Aware Decision Gate Seeds 7 and 11

## Configuration

- date: 2026-06-29
- manifest: `E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_smoke_1m\manifest.json`
- corpus status: `admissible_document_level`
- train limit: 300,000 examples
- validation limit: 100,000 examples
- seeds: 7, 11
- output directory: `E:\SPM_EXPERIMENTS\runs\reuse_structural_decision_gate_20260629`
- common settings: `address_lags=1`, `topology_enable_delta=false`,
  `topology_accept_credit=0.05`, `max_sparse_decisions_per_node=512`,
  `classification_learning_rate=0.8`,
  `classification_mature_learning_rate=0.2`,
  `record_channel_attribution=true`

The acceptance threshold was intentionally set above the previously observed
base structural value of the early reusable programs. This creates a
decision-changing gate rather than another run where reuse only changes
reported margins.

## Results

| seed | run | structural gate | reuse weight | eval NLL | current-token NLL | accepted | rejected | attr rows | max attr reuse bonus | steps/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | raw credit | false | 0.00 | 7.22890170 | 8.62306488 | 5 | 28 | 8 | 0.00000000 | 9710.01 |
| 7 | cost gate | true | 0.00 | 7.22890170 | 8.62306488 | 5 | 28 | 8 | 0.00000000 | 9727.29 |
| 7 | reuse gate | true | 0.05 | 7.21415952 | 8.62306488 | 5 | 17 | 6 | 0.21266388 | 9248.28 |
| 11 | raw credit | false | 0.00 | 7.23165016 | 8.62306488 | 5 | 28 | 8 | 0.00000000 | 9735.15 |
| 11 | cost gate | true | 0.00 | 7.23165016 | 8.62306488 | 5 | 28 | 8 | 0.00000000 | 9720.69 |
| 11 | reuse gate | true | 0.05 | 7.21561020 | 8.62306488 | 5 | 17 | 6 | 0.21266388 | 9316.86 |

Mean eval NLL:

| run | mean eval NLL | delta vs raw |
|---|---:|---:|
| raw credit | 7.23027593 | 0.00000000 |
| cost gate | 7.23027593 | 0.00000000 |
| reuse gate | 7.21488486 | -0.01539107 |

Mean throughput:

| run | mean steps/s | delta vs raw |
|---|---:|---:|
| raw credit | 9722.58 | 0.0% |
| cost gate | 9723.99 | +0.0% |
| reuse gate | 9282.57 | -4.5% |

## Topology Difference

Raw and cost-only gates accepted the same high-base-value programs on both
seeds:

- `ContentMatch([2])`
- `Tuple([2,3])`
- `Tuple([2,4])`
- `Tuple([3,5])`
- `Tuple([4,6])`

The reuse-aware gate accepted earlier reusable tuple programs instead:

- `Tuple([2])`
- `Tuple([1,2])`
- `Tuple([2,3])`
- `Tuple([2,4])`
- `Tuple([3,5])`

For seed 7, the first accepted event changed from rejecting
`Tuple([2])` with `structural_value_without_reuse=0.01795413` under the raw and
cost-only gates to accepting it with `binding_reuse_bonus=0.07037891` and final
event credit `0.08833304` under the reuse-aware gate.

## Interpretation

This is the first small gate where reuse-aware structural value changed
topology admission rather than only changing diagnostics. On two matched seeds,
the reuse-aware gate improved validation NLL while preserving the same accepted
channel count and reducing rejected proposals from 28 to 17.

The evidence is still small-scale. It supports continuing the reuse-aware route
as an architectural candidate, but it does not yet justify changing defaults.
The next validation should keep this decision-changing threshold style and run a
larger matched seed set or a transfer split before using nonzero
`binding_reuse_value_weight` as a default.
