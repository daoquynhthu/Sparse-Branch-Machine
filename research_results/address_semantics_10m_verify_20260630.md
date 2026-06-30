# Address Semantics 10M Verification (2026-06-30)

Full re-verification of the address-semantics framework on the 10M/1M FineWeb-Edu
corpus after the dependency-closure and caller-availability extensions (commits
`a5d0102`..`2c229cf` plus skill migration `afff217`).

## Configuration

- corpus: FineWeb-Edu 10M train / 1M validation manifest
- seeds: 7, 11, 19
- address execution: interpreted frames
- accepted-channel retirement: preserve
- topology: adaptive content proposals, Delta proposals disabled
- attribution: dependency-aware frozen program attribution
- structural-value admission: enabled
- controls: current-token, interpolated multiscale, seed-only, content-only,
  tuple-only frozen controls

## Quality

| seed | eval NLL | current-token NLL | interpolated NLL | accepted | pruned | speed |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 6.341235 | 6.499756 | 6.768842 | 5 | 0 | 9,732.90/s |
| 11 | 6.341288 | 6.499756 | 6.768842 | 5 | 0 | 9,733.69/s |
| 19 | 6.341063 | 6.499756 | 6.768842 | 5 | 0 | 9,639.46/s |

Mean eval NLL is 6.341195 with population standard deviation 0.000096.
All three seeds beat the current-token control by about 0.1586 nats/token.
All three seeds beat the interpolated multiscale control by about 0.4276 nats/token.

## Comparison with 2026-06-28 structural-value multiseed

| metric | 2026-06-28 | 2026-06-30 | delta |
|---|---:|---:|---:|
| mean eval NLL | 6.366384 | 6.341195 | -0.025189 |
| mean current-token NLL | 6.499756 | 6.499756 | 0 |
| mean interpolated NLL | 6.768842 | 6.768842 | 0 |
| accepted | 5 | 5 | 0 |
| pruned | 0 | 0 | 0 |
| max_bucket_candidates | 8 | 8 | 0 |

The eval NLL improved by about 0.025 nats/token. Controls are identical. The
learned program structure is the same: operations `[0, 2, 3, 2, 3, 3]`, programs
`[[1], [2], [1,2], [3], [1,3], [2,3]]`.

## Learned Programs

All three seeds converged to the same address program set:

- operations: `[0, 0, 0, 2, 3, 0]` (proposal order includes rejected probes)
- accepted operations: `[0, 2, 3, 2, 3, 3]`
- programs: `[[1], [2], [1, 2], [3], [1, 3], [2, 3]]`
- channel 0: Tuple(1) — positional, phase Active
- channel 1: ContentMatch(2) — independent content match, phase Active
- channel 2: ContentFollow(1,2) — depends on channel 1, phase Active
- channel 3: ContentMatch(3) — independent content match, phase Active
- channel 4: ContentFollow(1,3) — depends on channel 3, phase Active
- channel 5: ContentFollow(2,3) — depends on channel 3, phase Active

All 6 channels are effective-enabled. dependency_blocked_channels = 0.

## Bounded-work diagnostics

- `avg_active`: 5.99994 (seed 7)
- `avg_candidates`: 83.12 (seed 7)
- `max_bucket_candidates_inspected`: 8
- `address_execution_frames`: 65,762,464
- `address_binding_hits`: not in summary (see binding_reuse below)
- `binding_reuse_observations`: 10,245,229
- `binding_reuse_unique_keys`: 19,869
- `binding_reuse_events`: 1,736,510
- `estimated_bytes`: 1.65 GB per seed
- `dependency_blocked_channels`: 0
- `recoverable_retired_channels`: 0
- `quarantined_channels`: 0

## Dependency graph audit (seed 7)

- channel 2 has `direct_caller_count=0` (leaf), `effective_direct_caller_count=0`
- channel 3 has `direct_caller_count=2` (channels 4 and 5 call it),
  `effective_direct_caller_count=2`, `blocked_direct_caller_count=0`
- channel 1 has `direct_caller_count=1` (channel 2 calls it),
  `effective_direct_caller_count=1`, `blocked_direct_caller_count=0`

All dependency edges are routable. No blocked callers.

## Program attribution

All 6 accepted programs have positive mean credit under frozen evaluation.
The dependency-aware attribution separates caller-removed credit from
dependency-removed credit:

- ContentFollow channels (2, 4, 5) show non-zero `call_matches` and
  `call_key_reuse_events`, confirming the call substrate is active.
- ContentMatch channels (1, 3) show `binding_key_reuse_events > 0`, confirming
  binding reuse is active.

## Gate result

The 10M architecture verification passes:

- 3/3 seeds beat current-token control;
- 3/3 seeds beat interpolated multiscale control;
- 5 accepted, 0 pruned, 0 dependency-blocked;
- bounded-work stable (avg_active ~6, max_bucket_candidates 8);
- all accepted programs have positive frozen attribution;
- dependency graph shows 0 blocked callers across all channels;
- structural-value admission did not change topology decisions relative to
  the 2026-06-28 baseline (same 5 accepted programs).

The 0.025 nats/token improvement over the 2026-06-28 result is attributable to
the dependency-aware routing eligibility and effective-state lifecycle changes
(commits `4aa1b13`..`2c229cf`). The learned program structure is unchanged,
so the improvement is from routing quality, not from different structures.

## Run artifacts

- output dir: `E:\SPM_EXPERIMENTS\runs\address_semantics_10m_verify_20260630\`
- seeds: `seed-7.json`, `seed-11.json`, `seed-19.json`, `summary.json`
- commit: `afff217`
