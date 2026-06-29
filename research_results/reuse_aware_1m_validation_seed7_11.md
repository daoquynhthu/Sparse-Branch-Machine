# Reuse-Aware Structural Admission 1M Validation Seeds 7 and 11

## Configuration

- date: 2026-06-29
- manifest: `E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_smoke_1m\manifest.json`
- eval split: `validation`
- corpus status: `admissible_document_level`
- train limit: `1,000,000` requested; actual train split has 997,873 examples
- validation limit: 200,000 examples
- seeds: 7, 11
- output directory: `E:\SPM_EXPERIMENTS\runs\reuse_structural_1m_validation_20260629`
- common settings: `address_lags=1`, `topology_enable_delta=false`,
  `topology_accept_credit=0.05`, `max_sparse_decisions_per_node=512`,
  `classification_learning_rate=0.8`,
  `classification_mature_learning_rate=0.2`,
  `record_channel_attribution=true`

This repeats the decision-changing small gate at the full 1M smoke training
scale. It is deliberately not a larger corpus experiment.

## Results

| seed | run | structural gate | reuse weight | eval NLL | current-token NLL | accepted | rejected | live nodes | state MB | max attr reuse bonus | steps/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | raw credit | false | 0.00 | 6.85797398 | 7.78004005 | 5 | 28 | 117,372 | 601.38 | 0.00000000 | 7800.58 |
| 7 | cost gate | true | 0.00 | 6.85797398 | 7.78004005 | 5 | 28 | 117,372 | 601.38 | 0.00000000 | 7821.55 |
| 7 | reuse gate | true | 0.05 | 6.86195013 | 7.78004005 | 5 | 17 | 121,488 | 631.16 | 0.27600429 | 7420.93 |
| 11 | raw credit | false | 0.00 | 6.85884666 | 7.78004005 | 5 | 28 | 117,327 | 600.90 | 0.00000000 | 7883.94 |
| 11 | cost gate | true | 0.00 | 6.85884666 | 7.78004005 | 5 | 28 | 117,327 | 600.90 | 0.00000000 | 7915.82 |
| 11 | reuse gate | true | 0.05 | 6.86214851 | 7.78004005 | 5 | 17 | 121,495 | 630.96 | 0.27600429 | 7486.11 |

Mean eval NLL:

| run | mean eval NLL | delta vs raw |
|---|---:|---:|
| raw credit | 6.85841032 | 0.00000000 |
| cost gate | 6.85841032 | 0.00000000 |
| reuse gate | 6.86204932 | +0.00363900 |

Mean throughput and state:

| run | mean steps/s | delta vs raw | mean live nodes | mean state MB |
|---|---:|---:|---:|---:|
| raw credit | 7842.26 | 0.0% | 117,349.5 | 601.14 |
| cost gate | 7868.68 | +0.3% | 117,349.5 | 601.14 |
| reuse gate | 7453.52 | -5.0% | 121,491.5 | 631.06 |

## Topology Difference

The topology split remained the same as in the 300k/100k decision-changing
gate:

- raw and cost-only gates accepted `ContentMatch([2])`, `Tuple([2,3])`,
  `Tuple([2,4])`, `Tuple([3,5])` and `Tuple([4,6])`;
- reuse-aware admission accepted `Tuple([2])`, `Tuple([1,2])`,
  `Tuple([2,3])`, `Tuple([2,4])` and `Tuple([3,5])`;
- rejected proposals again fell from 28 to 17 on both seeds.

At 300k/100k this topology change improved validation and test NLL. At the 1M
training scale it worsened validation NLL while also increasing live nodes and
state size. The same structural preference therefore does not transfer
monotonically with longer training.

## Interpretation

This result weakens the case for enabling a fixed positive
`binding_reuse_value_weight` by default. The reuse signal is real and changes
admission, but the current bonus can over-prefer early reusable tuple programs
that displace later structures with better long-run value.

The next engineering step should not be another blind weight sweep. The
admission rule needs a stronger guard, for example requiring a minimum base
structural value, normalizing reuse by validation exposure, or delaying reuse
bonus until a candidate survives enough held-out credit. Any such change should
be tested against both the 300k/100k and 1M/200k gates.
