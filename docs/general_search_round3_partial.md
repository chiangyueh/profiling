# General Search Round 3 Partial Result

Environment:

```text
SoC: Ascend910B3
CANN: 8.1
version: 20260730-general-feedback-frontier-v3
workloads: 16
searched candidates planned: 231
searched candidates measured: 48
measurement deferred: 183
campaign fingerprints excluded before the run: 332
```

Only `general_holdout_large`, `general_holdout_skinny_m`, and
`control_skinny_n` reached candidate measurement. The other 13 workloads
were deferred because a same-run reference exceeded the 5% CV reliability
limit. This is a partial campaign, not a broad-search conclusion.

`control_skinny_n` reproduced a known transfer result:

```text
candidate: 0.077278 ms
bank:      0.104510 ms
official:  0.103170 ms
speedup:   1.35239x vs bank, 1.33505x vs official
```

`general_holdout_skinny_m` remained within noise. The apparent
`general_holdout_large` 1.48514x official speedup is invalid: official was
2.08892 ms while the same-run bank control was 0.754802 ms and the previous
coherent official result was about 0.756 ms. The candidate was actually
1.863x slower than bank. No general-workload improvement was established.

The 48 completed tiling fingerprints are versioned in
`config/general_search_v1_round3_partial_fingerprints.csv`. Five stable
source-best measurements are retained in
`config/general_search_v1_round3_partial_observations.csv`; model calibration
uses their bank-relative ratio and the raw pre-calibration model ratio.
The incoherent large official measurement is not treated as positive
evidence.

The profiler now:

- retries an official or bank reference up to two times when CV exceeds 5%;
- defers candidates if stable official and bank references differ by over 15%;
- reports the pair gap as `incoherent_baselines` instead of improved/regressed;
- preserves deferred schedules and excludes all 48 completed fingerprints.

Host callback validation for the next unchanged `--mode full` command yields
197 unique, previously unmeasured candidates across 15 workloads. It has zero
overlap with the 380 versioned campaign fingerprints.
