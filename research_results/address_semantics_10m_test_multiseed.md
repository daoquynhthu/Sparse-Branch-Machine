# Address Semantics 10M Test-Transfer Multi-Seed Gate

## Configuration

- corpus: FineWeb-Edu 10M train / 1M test split from `fineweb_edu_v1_r1_train10m_eval1m`
- seeds: 7, 11, 19
- address execution: interpreted frames
- accepted-channel retirement: preserve
- topology: adaptive, `address_lags=1`, Delta proposals disabled
- sparse output cap: `max_sparse_decisions_per_node=512`
- token learning rates: `classification_learning_rate=0.8`, `classification_mature_learning_rate=0.2`
- attribution: frozen channel/program attribution enabled
- runner: `scripts/run_corpus_batch.py`, `max_workers=2`

## Results

| seed | eval NLL | current-token NLL | interpolated NLL | accepted | pruned | positive program attribution | speed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 6.35648468 | 6.48778031 | 6.74802513 | 5 | 0 | 10/10 | 9,316.76/s |
| 11 | 6.35602945 | 6.48778031 | 6.74802513 | 5 | 0 | 10/10 | 9,349.80/s |
| 19 | 6.35665468 | 6.48778031 | 6.74802513 | 5 | 0 | 10/10 | 12,184.27/s |

Mean eval NLL is 6.35638960 with population standard deviation 0.00026395.
Mean throughput is 10,283.61 examples/s.

All three seeds learned the same operation/program sequence:

```text
operations: [0, 0, 0, 2, 3, 0]
programs:   [[1], [2], [1, 2], [2], [1, 2], [3]]
```

Bounded-work diagnostics remained stable:

- `avg_active`: 5.99993 to 5.99994
- `max_bucket_candidates_inspected`: 8
- `estimated_bytes`: 1.92 GB per seed
- `address_execution_frames`: 65,858,402
- `address_binding_hits`: 43,989,814
- `address_binding_misses`: 21,868,588

## Interpretation

The test-transfer gate passes. The completed address-semantics framework remains
competitive on the held-out test split, beats the current-token control on all
three seeds, preserves accepted structures, and retains positive frozen program
attribution. This supports moving Task 9 from multi-seed validation to
structural-value admission calibration.

Operational note: an attempted 3-worker local run was stopped after evidence of
memory pressure. The three workers held about 5.9 GB resident memory and left
only about 1 GB available while total CPU utilization stayed below 30%. The
successful run used two workers, which is the current local saturation point for
this gate.
