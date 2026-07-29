# General Search Round 2 (net_log8)

Environment:

```text
SoC: Ascend910B3
CANN: 8.1
scope: general_search_v1
workloads: 16
searched candidates: 145
campaign fingerprints excluded: 187
runtime rejected: 0
history reused: 0
```

The run proved that the versioned round-1 manifest works after a clean clone:
all 145 NPU candidates were new and all completed. Same-run baseline CV checks
also prevented the disturbed comparisons seen in `net_log7`.

Two workloads improved against both bank and official baselines:

| workload | source | searched ms | official ms | bank ms | official/search |
|---|---:|---:|---:|---:|---:|
| `general_holdout_odd` | global | 0.193720 | 0.208592 | 0.210242 | 1.07677x |
| `control_skinny_n` | transfer | 0.069772 | 0.104188 | 0.104700 | 1.49326x |

The odd workload is the reliable general result. It independently reproduced
the round-1 discovery, although its `T=176x176x64` schedule was slower than
round 1's `T=224x144x64`. The skinny-N row is a stable control result, not
evidence of broad MatMul generalization.

Five workloads regressed:

```text
balanced  0.566x
large     0.658x
trans_a   0.785x
trans_b   0.885x
bf16      0.631x
```

The source-best records show that the analytical model underestimates the
cost of some small-baseK, transposed, BF16 and large-L2 schedules. These 44
stable source-best observations are versioned in
`config/general_search_v1_round2_observations.csv`; all 145 exact fingerprints
are versioned separately so none are measured again.

The next host frontier:

- excludes 187 round-1 plus 145 round-2 fingerprints;
- applies conservative source residuals from the 44 stable observations;
- deepens local/global/diverse/transfer pools without adding shape-name rules;
- keeps NPU work bounded to 16 candidates per workload.

On the CANN 8.1 910B3 callback this produced 231 unique third-round candidates
with zero overlap against the first 332 measurements.
