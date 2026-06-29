# Reuse-Aware Structural Value Small Gate Seed 7

## Configuration

- date: 2026-06-29
- manifest: `E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_smoke_1m\manifest.json`
- corpus status: `admissible_document_level`
- train limit: 300,000 examples
- validation limit: 100,000 examples
- seed: 7
- common settings: `address_lags=1`, `topology_enable_delta=false`,
  `max_sparse_decisions_per_node=512`, `classification_learning_rate=0.8`,
  `classification_mature_learning_rate=0.2`, `record_channel_attribution=true`
- outputs: `E:\SPM_EXPERIMENTS\runs\reuse_structural_small_20260629`

## Results

| run | structural gate | reuse weight | eval NLL | current-token NLL | interpolated NLL | accepted | rejected | pruned | attr rows | positive attr rows | max attr reuse bonus | steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw credit | false | 0.00 | 7.18416732 | 8.62306488 | 8.91180559 | 5 | 0 | 0 | 10 | 8 | 0.00000000 | 11236.09 |
| cost gate | true | 0.00 | 7.18416732 | 8.62306488 | 8.91180559 | 5 | 0 | 0 | 10 | 8 | 0.00000000 | 11271.50 |
| reuse gate | true | 0.05 | 7.18416732 | 8.62306488 | 8.91180559 | 5 | 0 | 0 | 10 | 8 | 0.21266388 | 11338.76 |

All three runs emitted the same global structural counters:

| field | value |
|---|---:|
| address execution frames | 2,328,320 |
| address binding hits | 1,566,852 |
| address binding misses | 761,468 |
| binding reuse observations | 1,165,955 |
| binding reuse unique keys | 17,488 |
| binding reuse events | 173,575 |

The first accepted event in the reuse-aware run was `Tuple([2])` at step 6144.
Its `structural_value_without_reuse` was `0.01795413`, while
`binding_reuse_bonus=0.07037891` raised the reported event credit to
`0.08833304`. The same event in the raw and cost-only runs had zero reuse bonus.

## Interpretation

This is a small validation and comparison run, not a large training result. It
shows that the current framework runs stably after the `SBMCKPT5` topology event
change, that reuse-aware fields are present in both topology events and program
attribution, and that repeated typed bindings produce a nonzero structural
bonus when enabled.

The run does not show a quality advantage for reuse-aware admission. In this
window all three gates accepted the same number of programs and produced
identical validation NLL. The immediate conclusion is therefore narrower:
reuse-aware structural value is operational and auditable, but its admission
effect needs a matched gate where the bonus can change at least one topology
decision before any research claim is made.
