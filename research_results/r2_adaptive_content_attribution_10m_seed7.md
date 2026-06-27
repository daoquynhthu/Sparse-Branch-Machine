# R2 Adaptive Content Attribution

Configuration: FineWeb-Edu 10M train, 1M held-out split, seed 7,
adaptive topology enabled, Delta proposals disabled, content proposals enabled,
`address_lags=1`, `max_sparse_decisions_per_node=512`,
`classification_learning_rate=0.8`,
`classification_mature_learning_rate=0.2`,
`record_channel_attribution=true`.

Raw outputs:

- validation: `E:\SPM_EXPERIMENTS\runs\r2_adaptive_content_doc_attribution_10m_seed7.json`
- test: `E:\SPM_EXPERIMENTS\runs\r2_adaptive_content_doc_attribution_10m_seed7_test.json`

## Overall Quality

| split | eval NLL | current-token NLL | interpolated lag NLL | topology accepted | topology pruned |
|---|---:|---:|---:|---:|---:|
| validation | 6.587315 | 6.499756 | 6.768842 | 16 | 11 |
| test | 6.574885 | 6.487780 | 6.748025 | 16 | 11 |

The adaptive model beats the interpolated short-context control but still loses
to the current-token baseline on both held-out splits. This is a model-level
quality blocker despite positive accepted-structure attribution.

## Accepted Content Channels

| split | channel | op | lags | mean credit | positive docs | documents | positive doc fraction |
|---|---:|---:|---|---:|---:|---:|---:|
| validation | 1 | ContentMatch | [4] | 0.041535 | 875 | 877 | 0.997720 |
| validation | 2 | ContentFollow | [2,3] | 0.165205 | 877 | 877 | 1.000000 |
| validation | 4 | ContentFollow | [1,4] | 0.041184 | 873 | 877 | 0.995439 |
| validation | 5 | ContentFollow | [3,4] | 0.029869 | 865 | 877 | 0.986317 |
| test | 1 | ContentMatch | [4] | 0.043752 | 993 | 994 | 0.998994 |
| test | 2 | ContentFollow | [2,3] | 0.167020 | 994 | 994 | 1.000000 |
| test | 4 | ContentFollow | [1,4] | 0.043259 | 990 | 994 | 0.995976 |
| test | 5 | ContentFollow | [3,4] | 0.031592 | 985 | 994 | 0.990946 |

R2 accepted-structure attribution is positive and cross-document on both
validation and independently sampled test splits. The strongest structure is
`ContentFollow([2,3])`, with positive contribution in every held-out document
on both splits.

Interpretation: the implemented content-addressing primitives are not merely
training-set artifacts. However, the full adaptive model still underperforms
the current-token control, so the next blocker is channel fusion/utilization,
not existence of accepted content structures.
