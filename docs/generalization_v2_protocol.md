# External generalization v2 protocol

This campaign answers a different question from the four general-search
development rounds: can the frozen candidate generator improve workloads whose
shape was not used to choose, reject, or deepen a candidate?

## Frozen inputs

- Search implementation: `general_search_v1`.
- NPU budget: at most 16 candidates per workload.
- Candidate sources: local, global, diverse, and transfer.
- Template eligibility: BASE plus hardware-underfilled deterministic split-K;
  no measured M/N/K interval is consulted by the general split-K gate.
- Training evidence: 577 completed exact fingerprints and 79 stable source-best
  observations from rounds 1 through 4.
- References: searched candidate, RuntimeKb bank control, and official operator
  are measured in the same run. Existing CV and baseline-coherence guards stay
  enabled.

The 22 workloads in `config/workloads_generalization_v2.csv` were committed
before their NPU results existed. None of their `(M,N,K)` triples occurs in any
other repository workload CSV. Candidate generation does not inspect workload
IDs, evidence labels, or the strata below.

## Strata

The workload IDs encode only evaluation strata:

- `dense`: aligned, odd-tail, tall, wide, and larger dense matrices.
- `k`: shallow K, deep K, aligned low-output/deep-K, and odd-K control.
- `skinny`: independent skinny-M and skinny-N shapes.
- `layout`: TN, NT, and TT contracts.
- `bf16`: aligned and odd-tail BF16.
- `fp32`: NN, NT, TN, and TT FP32.
- `template`: unseen AL1-full-load and BL1-full-load shapes.

These strata test mechanisms that change the legal template set, Cube
alignment, active-core count, memory traffic, or transpose callback contract.
They are evaluation coverage, not shape-family branches in the search code.

## Preregistered verdicts

`EXTERNAL_GENERALIZATION_RESULT=supported` requires:

1. at least 20 of 22 workloads have a measured searched candidate;
2. at least 6 of 22 beat both official and bank references beyond noise;
3. improvements span at least three of the seven strata.

This is evidence that the automatic method transfers beyond its development
set, not a claim that every workload is optimized. The stricter
`UNIVERSAL_UNSEEN_OPTIMIZATION_RESULT=supported` requires all 22 workloads to
improve. Both verdicts are printed so an official fallback cannot be counted as
an optimization success.
