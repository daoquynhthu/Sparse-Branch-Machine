# Structural-Value Admission Calibration

## Question

Task 9 Step 4 asks whether `topology_accept_uses_structural_value` can become
an admissible topology gate after the completed address-semantics framework has
passed multi-seed validation and shard transfer.

The important distinction is scale:

- topology admission compares per-observation mean credit after a one-time
  description penalty divided by probe observations;
- frozen attribution reports total codelength credit and total accumulated
  costs over evaluation examples.

The large negative `structural_value` values in attribution JSON therefore do
not imply that topology admission is over-penalized.

## Matched Full 10M Validation

Configuration matched the raw-credit 10M/1M validation gate except for:

```text
topology_accept_uses_structural_value=true
```

All other parameters were unchanged:

- adaptive topology enabled
- `address_lags=1`
- Delta proposals disabled
- `max_sparse_decisions_per_node=512`
- `classification_learning_rate=0.8`
- `classification_mature_learning_rate=0.2`
- frozen channel/program attribution enabled

## Results

| seed | raw NLL | structural-gate NLL | delta | raw accepted | structural accepted | min accepted structural event |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 6.36636745 | 6.36685137 | +0.00048392 | 5 | 5 | 0.00645429 |
| 11 | 6.36645222 | 6.36645222 | +0.00000000 | 5 | 5 | 0.00432127 |
| 19 | 6.36633118 | 6.36629170 | -0.00003948 | 5 | 5 | 0.00541015 |

Mean raw NLL was 6.36638362. Mean structural-gate NLL was 6.36653176. The mean
delta was +0.00014815 nats/token.

Both gates learned the same operation/program sequence for every seed:

```text
operations: [0, 0, 0, 2, 3, 0]
programs:   [[1], [2], [1, 2], [2], [1, 2], [3]]
```

The structural-gate runs preserved accepted structures (`topology_pruned=0`) and
kept 10/10 positive-mean frozen program-attribution entries.

## Short Sanity Gate

A 1M/100k matched calibration was also run before the full gate. Raw-credit and
structural-value admission produced identical NLLs, accepted counts and pruned
counts for seeds 7, 11 and 19.

## Decision

The default description-only structural-value admission gate is admissible for
the current framework. It does not reject any known useful program under matched
10M validation and it changes mean NLL by only +0.00015 nats/token.

This does not calibrate nonzero `structural_execution_cost_weight`. Execution
cost penalties remain a separate future experiment because they can reject
content-follow programs by construction and were not needed for the current
admission gate.
