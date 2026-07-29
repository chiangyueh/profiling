# General search repeated run: net_log7

`net_log7` ran version `20260730-full-general-active-round2-v1`, but it did
not execute active round 2. Its plan was identical to round 1:

- 16 workloads
- 187 searched candidates
- `resume_exact_assigned=0`
- `npu_searched_pending=187`

The cause is state transport, not candidate generation. The exact resume
ledger is under `results/`, which is intentionally ignored by Git. A clean
clone therefore had no `npu_full_general_v1_resume.csv` when search started.

## Reliable evidence

The odd-shape result reproduced:

| workload | net_log6 | net_log7 | official speedup |
| --- | ---: | ---: | ---: |
| `general_holdout_odd` | 0.186564 ms | 0.185744 ms | 1.119x / 1.130x |

This is the strongest general-workload result because candidate, bank and
official measurements remained stable in both runs.

## Unreliable apparent wins

Several baselines were heavily disturbed while candidate latency remained
near the prior run:

| workload | net_log6 official | net_log7 official | net_log7 official CV |
| --- | ---: | ---: | ---: |
| `general_holdout_trans_ab` | 0.040316 ms | 0.084704 ms | 6.1% |
| `control_skinny_n` | 0.104790 ms | 0.262832 ms | 6.6% |
| `control_split_k` | 0.034294 ms | 0.089994 ms | over 100% |
| `general_holdout_bf16` | 0.202082 ms | 0.574274 ms | 61.5% |

Consequently, the reported 2.119x `trans_ab` and 3.787x skinny-N speedups are
not valid estimates from this run. The previously stable skinny-N result is
still valid evidence, but `net_log7` must not be used to inflate it.

## Corrective action

- Version the 187 exact round-1 fingerprints outside `results/`, so a clean
  clone excludes them without needing the timing ledger.
- Continue using a local resume ledger for source-specific calibration when
  available.
- Keep unstable exact rows excluded from future NPU work, but do not use them
  for model calibration or transfer starts.
- Treat a measurement with standard deviation above 5% of its median as
  inconclusive. If either paired baseline is unstable, defer that workload's
  candidate batch instead of spending the NPU budget under a bad baseline.

The host-side CANN 8.1 callback validation after this change generated 145
new candidates across 14 workloads with zero overlap against round 1.
