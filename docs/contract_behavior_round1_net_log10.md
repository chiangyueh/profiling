# Contract behavior round 1 (`net_log10`)

## What ran

- 22 external-v2 workloads with 32 NPU candidates each.
- 704 independently generated schedules: 703 `contract_global` and one
  `local_bank_anchor`.
- 595 candidates passed NPU preflight and timing.
- 109 BASE candidates passed host legality and callback roundtrip but failed
  NPU output coverage; all were discarded and the run continued.
- Every workload used same-run candidate, bank-control, and official
  references. The reported reference pairs were coherent.

The compact terminal output is not the measurement count. It suppresses
ordinary candidate rows; `searched=704` and `resume exact_rows=639` account for
595 successful candidates plus 44 paired references.

## Measured result

Three workloads beat both same-run references:

| workload | template | candidate | bank | official | official speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `external_v2_fp32_nt` | single-core split-K | 0.040932 ms | 0.043390 ms | 0.042300 ms | 1.03342x |
| `external_v2_k_deep` | deterministic split-K | 0.565740 ms | 0.621716 ms | 0.618782 ms | 1.09376x |
| `external_v2_template_al1` | BASE | 0.013118 ms | 0.013826 ms | 0.013862 ms | 1.05672x |

The overall result was 3 improved, 12 within noise, and 7 regressed. This is
evidence that the independent contract path can find useful structures outside
the bank geometry, but it is not evidence of broad workload generalization.
Balanced dense, BF16-aligned, and transposed dense cases remain weak.

## Runtime rejection evidence

All 109 output-coverage failures were BASE schedules, concentrated in eight
workloads. The failures are not separable by one raw field: the same base
sizes, L1/DB choices, core counts, and L2 orders also occur in successful
schedules. The installed CANN BASE kernel supports multi-base single-core
tiles, and an exact simulation of its L2 task mapping covers the complete
output grid for these records.

These records are therefore persisted as exact NPU execution failures, not
promoted to an unsupported shape-family legality rule. They influence behavior
prediction and exclusion, but they are not used as legal parents for
counterfactual mutation.

## Persisted feedback

- `config/contract_behavior_v1_round1_fingerprints.csv` contains all 704 exact
  searched fingerprints so a clean clone does not remeasure them.
- `config/contract_behavior_v1_round1_observations.csv` contains the stable
  printed measurements and all 109 runtime rejections.

The next contract round keeps 32 NPU candidates per workload. It generates a
new frontier after exact exclusion, winner/regression feedback, and uncovered
behavior expansion; it does not refill from legacy constructors.
