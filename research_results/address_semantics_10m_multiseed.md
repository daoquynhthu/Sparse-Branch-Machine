# Address Semantics 10M Multi-Seed Gate

## Configuration

- corpus: FineWeb-Edu 10M train / 1M validation manifest
- seeds: 7, 11, 19
- address execution: interpreted frames
- accepted-channel retirement: preserve
- topology: adaptive content proposals, Delta proposals disabled
- attribution: dependency-aware frozen program attribution
- controls: current-token, interpolated multiscale, seed-only, active-only,
  content-only and tuple-only frozen controls

## Quality

| seed | eval NLL | current-token NLL | interpolated NLL | accepted | pruned | positive program attrs |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 6.366367 | 6.499756 | 6.768842 | 5 | 0 | 10/10 |
| 11 | 6.366452 | 6.499756 | 6.768842 | 5 | 0 | 10/10 |
| 19 | 6.366331 | 6.499756 | 6.768842 | 5 | 0 | 10/10 |

Mean eval NLL is 6.366384 with population standard deviation 0.000051.
All three seeds beat the current-token control by about 0.1334 nats/token.

## Learned Programs

All three seeds converged to the same address program set:

- operations: `[0, 0, 0, 2, 3, 0]`
- programs: `[[1], [2], [1, 2], [2], [1, 2], [3]]`

The learned set contains Tuple, ContentMatch and ContentFollow programs. The
content-conditioned channels remain useful under frozen evaluation rather than
being transient probe artifacts.

## Gate Result

Task 9 Step 2 passes:

- three of three seeds beat the current-token control;
- no seed loses or needs a structural-failure explanation;
- every emitted program-attribution entry has positive mean credit;
- accepted structures are not physically pruned under preserve policy;
- bounded-work diagnostics remain stable (`max_bucket_candidates_inspected = 8`
  for all three seeds).

This establishes that the 10M/1M seed-7 result was not a one-seed accident. It
does not close shard-transfer validation; Task 9 Step 3 remains required.
