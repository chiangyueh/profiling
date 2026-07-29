# General Search Round 1 (net_log6)

Environment:

```text
SoC: Ascend910B3
CANN: 8.1
scope: general_search_v1
workloads: 16
searched candidates: 187
bank controls: 16
official baselines: 16
runtime rejected: 0
history reused: 0
```

All 16 workloads used same-run searched/bank/official measurement. The screen
found three dual-baseline improvements:

| workload | source | searched ms | official ms | bank ms | official/search |
|---|---:|---:|---:|---:|---:|
| general_holdout_odd | global | 0.186564 | 0.208830 | 0.209984 | 1.11935x |
| control_skinny_n | transfer | 0.068528 | 0.104790 | 0.108196 | 1.52916x |
| control_split_k | global | 0.028676 | 0.034294 | 0.035496 | 1.19591x |

`general_holdout_odd` is the first improvement on an unseen, non-family-named
workload. Its BASE schedule is `T=224x144x64`, output grid `7x17`, with
`S=224x144`, 20 AICs, and `L2=1x1(7x17)`.

The split-K control also produced a structure different from the prior
positive anchor: `T=128x128x128`, `S=128x384`, and 16 AICs. This is evidence
that global exploration can discover useful schedules outside a known local
basin.

The remaining 13 workload winners were within the measurement threshold.
Notable but unproven signals were:

```text
skinny_n_k8192  official/search=1.02786x bank/search=1.02945x
fp32_nt         official/search=1.01580x bank/search=1.03467x
fp32_nn         official/search=1.00921x bank/search=1.01039x
wide            official/search=1.00905x bank/search=1.00438x
```

Several global and diverse candidates were severely slower on balanced,
large, transpose, and BF16 workloads. Therefore equal source quotas are kept
only for the first screen. The second round:

1. loads exact profile-resume evidence;
2. calibrates model residuals independently per source and workload;
3. excludes every measured 23-field fingerprint;
4. retains one exploration point per available source;
5. fills the remaining budget by calibrated score with a per-source cap.

Host callback validation produced 145 second-round candidates over 14
workloads, with zero fingerprint overlap against the first 187 candidates.
