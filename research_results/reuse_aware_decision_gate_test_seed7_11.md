# Reuse-Aware Decision Gate Test Split Seeds 7 and 11

## Configuration

- date: 2026-06-29
- manifest: `E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_smoke_1m\manifest.json`
- eval split: `test`
- corpus status: `admissible_document_level`
- train limit: 300,000 examples
- test limit: 100,000 examples
- seeds: 7, 11
- output directory: `E:\SPM_EXPERIMENTS\runs\reuse_structural_decision_gate_test_20260629`
- common settings: `address_lags=1`, `topology_enable_delta=false`,
  `topology_accept_credit=0.05`, `max_sparse_decisions_per_node=512`,
  `classification_learning_rate=0.8`,
  `classification_mature_learning_rate=0.2`,
  `record_channel_attribution=true`

This repeats the decision-changing validation gate on the held-out test split.
Only evaluation split membership changes; train limits, seed set, threshold and
reuse weight match the validation gate.

## Results

| seed | run | structural gate | reuse weight | test NLL | current-token NLL | accepted | rejected | attr rows | positive attr rows | max attr reuse bonus | steps/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | raw credit | false | 0.00 | 7.12783220 | 8.44505155 | 5 | 28 | 8 | 7 | 0.00000000 | 9523.05 |
| 7 | cost gate | true | 0.00 | 7.12783220 | 8.44505155 | 5 | 28 | 8 | 7 | 0.00000000 | 9433.13 |
| 7 | reuse gate | true | 0.05 | 7.11181310 | 8.44505155 | 5 | 17 | 6 | 6 | 0.21262909 | 9024.05 |
| 11 | raw credit | false | 0.00 | 7.12917144 | 8.44505155 | 5 | 28 | 8 | 7 | 0.00000000 | 9483.72 |
| 11 | cost gate | true | 0.00 | 7.12917144 | 8.44505155 | 5 | 28 | 8 | 7 | 0.00000000 | 9477.98 |
| 11 | reuse gate | true | 0.05 | 7.11133061 | 8.44505155 | 5 | 17 | 6 | 6 | 0.21262909 | 9034.82 |

Mean test NLL:

| run | mean test NLL | delta vs raw |
|---|---:|---:|
| raw credit | 7.12850182 | 0.00000000 |
| cost gate | 7.12850182 | 0.00000000 |
| reuse gate | 7.11157186 | -0.01692997 |

Mean throughput:

| run | mean steps/s | delta vs raw |
|---|---:|---:|
| raw credit | 9503.38 | 0.0% |
| cost gate | 9455.56 | -0.5% |
| reuse gate | 9029.43 | -5.0% |

## Topology Transfer

The same topology split observed on validation reproduced on test:

- raw and cost-only gates accepted `ContentMatch([2])`, `Tuple([2,3])`,
  `Tuple([2,4])`, `Tuple([3,5])` and `Tuple([4,6])`;
- reuse-aware admission accepted `Tuple([2])`, `Tuple([1,2])`,
  `Tuple([2,3])`, `Tuple([2,4])` and `Tuple([3,5])`;
- rejected proposals fell from 28 to 17 on both seeds.

The accepted channel count stayed fixed at five in all runs. The improvement is
therefore not coming from simply admitting more channels; it comes from changing
which channels survive under the same capacity.

## Interpretation

This is stronger than the validation-only gate because the quality delta
transfers to a held-out split with the same training window and threshold. It
supports continuing reuse-aware structural admission as a serious architectural
candidate.

The evidence is still not sufficient to change defaults. The run uses two seeds
and a 300k/100k slice. The next gate should either expand seeds on the same
decision-changing threshold or run the same comparison on a larger document
view while tracking the roughly 5% throughput cost.
