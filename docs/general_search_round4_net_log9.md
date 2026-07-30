# General search round 4 (`net_log9`)

## What actually ran

- 16 development workloads.
- 197 searched candidates prepared and measured.
- 0 history rows reused.
- 0 measurements deferred.
- 0 callback/runtime rejects.
- 197 unique exact 23-field fingerprints.

The terminal had only 222 lines because normal mode prints source leaders,
meaningful candidate wins, workload verdicts, and errors. It did not print every
completed candidate. The candidate CSV remained the complete record.

## Result

Only `general_holdout_odd` beat both same-run references:

```text
shape       1537x2305x4099 fp16 NN
candidate   0.195046 ms
bank        0.208818 ms
official    0.207586 ms
speedup     1.07061x vs bank, 1.06429x vs official
```

The overall result was 1 improved, 8 within noise, 6 regressed, and 1 workload
with no remaining candidate. In particular, the large, BF16, transposed, and
deep diverse probes showed that the analytical model still substantially
underprices some schedules.

This is a valid active-search round, but not unseen-workload evidence: all 16
workloads had already influenced prior candidate selection. They are therefore
treated as development data from this point onward.

## Persisted evidence

- `config/general_search_v1_round4_fingerprints.csv`: all 197 completed exact
  fingerprints, so a clean clone will not remeasure them.
- `config/general_search_v1_round4_observations.csv`: 30 stable source-best
  measurements used for conservative source calibration and transfer.

The next default full run uses the preregistered external-v2 workload set
instead of deepening these 16 development shapes again.
