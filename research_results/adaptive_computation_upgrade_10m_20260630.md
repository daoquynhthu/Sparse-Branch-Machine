# Adaptive Computation Upgrade — 10M FineWeb-Edu Validation

Date: 2026-06-30
Branch: `adaptive-computation-upgrade`

## Summary

Three architectural upgrades were implemented and validated on the 10M/1M FineWeb-Edu corpus:

1. **Per-entry Adam-like momentum for sparse decision logits** (U2.1)
2. **Adaptive beam width with confidence-based truncation** (U1.1)
3. **Iterative routing refinement with expanded neighbor radius** (U1.2)

All upgrades preserve Agent.md §4 constraints: no dense global layers, no full-network backprop, bounded active work, causal ordering preserved.

## Key Result

**Momentum with bias correction improves 10M eval NLL from 6.341 to 6.230 (-0.111 nats/token).**

| Configuration | eval NLL (mean ± std) | PPL | Topology accepted | Bytes | Tok/s |
|---|---:|---:|---:|---:|---:|
| r3-baseline | 6.3412 | 566 | 5 | 2.16 GB | 15,756 |
| **upgrade-v1 (momentum)** | **6.2305 ± 0.0001** | **507** | 1 | 895 MB | 17,000 |
| upgrade-v1-adaptive | 6.2559 ± 0.0002 | 521 | 1 | 896 MB | 12,444 |
| adaptive+refinement only | 6.3625 | 579 | 5 | 2.22 GB | 10,235 |
| momentum only (no bias correction) | 6.4138 | 612 | 0 | 709 MB | 21,920 |

*Baseline numbers are single-seed; upgrade-v1 and upgrade-v1-adaptive are 3 seeds (7, 11, 19).*

## Detailed Findings

### Momentum is the dominant gain

- Without bias correction, momentum **destroyed topology growth** (topology_accepted=0) and degraded NLL to 6.414.
- Adding Adam bias correction fixed this and produced a large improvement: 6.230 vs 6.341.
- The model with momentum is **59% smaller** (895 MB vs 2.16 GB) and **8% faster** than baseline.
- Topology grows less (1 vs 5 accepted channels) but the existing channels learn much more effectively.

### Adaptive beam width + refinement are neutral/slightly negative

- Added to r3-baseline without momentum: 6.3625 (+0.021 nats).
- Added to momentum: 6.256 vs 6.230 (+0.026 nats).
- These features increase runtime cost (refinement scans more candidates) without matching the momentum gain.

### Presets

The `upgrade-v1` preset now contains only the proven improvements (momentum on top of r3-baseline). The full configuration including adaptive beam width and refinement is available as `upgrade-v1-adaptive` for further tuning.

```python
# python/sbm_presets.py
PRESETS = {
    "r3-baseline": { ... },
    "upgrade-v1": { "use_momentum": True, ... },
    "upgrade-v1-adaptive": { "use_momentum": True,
                              "beam_width": 8, "beam_width_min": 2, ... },
}
```

## Commands

```powershell
# r3-baseline reproduction
python scripts\run_corpus_batch.py `
  --library E:\SPM\build-fast\libsbm_api.dll `
  --manifest E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train10m_eval1m\manifest.json `
  --seeds 7,11,19 --preset r3-baseline `
  --output-dir E:\SPM_EXPERIMENTS\runs\r3_baseline_10m_multiseed

# upgrade-v1
python scripts\run_corpus_batch.py `
  --library E:\SPM\build-fast\libsbm_api.dll `
  --manifest E:\SPM_EXPERIMENTS\fineweb_edu_v1_r1_train10m_eval1m\manifest.json `
  --seeds 7,11,19 --preset upgrade-v1 `
  --output-dir E:\SPM_EXPERIMENTS\runs\upgrade_v1_10m_multiseed
```

## Caveats and Next Steps

- Single corpus (FineWeb-Edu 10M); transfer to 100M and other domains is unverified.
- Momentum reduces topology growth; whether this is a problem at larger scale is unknown.
- Adaptive beam width and refinement did not help in this configuration; they may need different thresholds or a different confidence signal.
- The next experiment should run `upgrade-v1` on the 100M heterogeneous stream gate (R3).
