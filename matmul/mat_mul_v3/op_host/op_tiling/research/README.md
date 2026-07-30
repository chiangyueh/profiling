# MatMulV3 Tiling Research

The search path is:

```text
workload
-> independent template contract solvers
-> common hardware and template legality
-> behavior-space coverage and measured-feedback ranking
-> official callback exact roundtrip
-> RuntimeKb lookup and isolated NPU preflight
-> paired official/control/candidate measurement
```

The default full run measures up to 40 exact callback-accepted schedules per
workload. Candidate geometry is generated from Cube alignment, L0/L1/L2
capacity, core partitioning, and one of five explicit kernel contracts:

- BASE
- single-core split-K
- deterministic split-K
- AL1 full-load
- BL1 full-load

Workload names and manually assigned shape families are not candidate gates.
The official callback schedule is measured as a control and is not copied into
the independent global solver.

The bundled `measured_fingerprints.csv` and `measured_observations.csv` contain
only the most recent independent-contract campaign. They prevent exact
remeasurement and provide paired NPU feedback. Results from a new run are read
from `results/npu_full_resume.csv`; winners, regressions, and rejected
fingerprints alter the next generated frontier.

Run from the repository root:

```bash
chmod +x run_npu.sh
./run_npu.sh --mode full
```
