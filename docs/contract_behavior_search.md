# Contract Behavior Search

`contract_behavior_v1` is the candidate-generation path used by
`run_npu.sh --mode full`. It does not call the legacy workload-family
constructors.

## Execution Boundary

The search path is:

1. `tiling_search.contracts`: common hardware correctness and
   template-specific kernel contracts.
2. `tiling_search.solvers`: independent lazy constructors for BASE,
   single-core split-K, deterministic split-K, AL1 full-load, and BL1
   full-load variants.
3. `tiling_search.behavior`: hardware-behavior features, feedback prediction,
   analytical prior, and max-min behavior coverage.
4. `tiling_search.feedback`: exact measured-fingerprint exclusion, winner
   mutation, regression counterfactual, and uncovered-behavior targets.
5. `tiling_search.orchestrator`: bounded interleaving of the independent
   solvers and construction of the callback frontier.
6. Existing CANN callback exact roundtrip, 23-field encoding, NPU preflight,
   paired candidate/bank/official measurement, resume, and reporting.

The RuntimeKb record is passed only to `local_anchor_mutations()`. Independent
solver constructors receive only workload dimensions/layout/dtype, hardware
capacities, and behavior targets. If the independent layer produces no
candidate, the run fails with per-template contract counts; it never calls a
legacy constructor.

## Contracts And Priors

`common_hardware_contract()` checks field domains, core limits, alignment,
L0/L1 capacity, buffering factors, and mode compatibility.

`template_kernel_contract()` checks the installed CANN 8.1 MatMulV3
relationships for the selected kernel family. Full-load residency and static
deterministic/fixpipe layouts are kernel contracts, not shape-profitability
gates.

`profitability_prior()` is used only in acquisition scoring. It cannot remove
a schedule from the legal pool.

## Bounded Search

The full default considers at most 12,000 lazy raw attempts and 5,000 unique
contract-legal schedules per workload. It retains 192 schedules by behavior
coverage, then retains 96 by a second behavior-coverage pass for exact callback
roundtrip. If a callback rejects a preferred schedule, the remaining
contract-generated behavior pool refills the callback budget; there is no
legacy refill. The paired NPU stage receives 16 schedules per workload by
default.

Behavior features include kernel family and suffix, active cores and rounds,
L0/L1 occupancy, K passes, padding efficiency, MTE-to-Cube work ratio, L2
working set/traversal, split-K reduction traffic, and full-load resident ratio.
They are not Euclidean distance over the raw 23 fields.

## Feedback

Resume measurements enter the feedback model only when the candidate and
stable official/bank references share the same run identity. Curated campaign
observation manifests must provide both ratios and non-unstable statuses and
are treated as trusted paired summaries. Previously measured exact
fingerprints are excluded.
Stable winners and regressions change the next generated pool through behavior
targets; same-workload records also create contract-checked semantic mutations.
Winner mutations and regression counterfactuals receive a bounded intervention
bonus so they can reach the callback frontier. Unseen workloads use
workload-distance plus behavior-distance prediction, not workload names.

The old `general_search_v1`, `bottleneck_guided_v1`, and manual family
constructors remain available only as regression baselines for non-full scopes.
