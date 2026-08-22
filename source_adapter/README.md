# Source-preserving multi-tiling adapter

This directory is a guardrail for the next measurement path. It does not contain
a cost model, prior measurements, RuntimeKb replay, callback timing, or CCE
data.

`source_lock.json` and its MatMulV3 scripts are legacy provenance tools. They
are **not used** by the non-MatMul collection described below.

## Non-MatMul source-strategy collection

`non_matmul_source_lock.json` defines the current collection contract:

- MatMul is excluded.
- A candidate is emitted only by an operator's original source strategy on its
  unchanged `TilingContext`.
- No candidate field is edited and no tile-field Cartesian product is made.
- Each strategy overlay retains one original registration and disables only the
  other registrations. Strategy code, predicates, and kernels are hash-locked.
  The common entry gets one audit-only return-status passthrough that records
  the already-generated tiling key, blockDim, raw byte count, and raw digest to
  a temporary file; it never writes a tiling field or changes the dispatcher
  result.
- A candidate must later execute and pass output comparison before latency is
  admitted to the training set.

The public 8.1.RC1 advanced-operator tag has eight registered
`FlashAttentionScoreGrad` strategies. This is the actual registry count in
that pinned source—not the larger count in a newer extracted tree. The
collector creates an isolated overlay for all eight so that this inventory is
auditable. The 20,000-record multi-tiling lane builds only the three original
strategies whose own `IsCapable` predicates overlap on its reviewed fp16/BNSD
geometry lattice. It does not call source strategies known to be ineligible
for that lane.

Run this read-only audit first:

```bash
python3 source_adapter/audit_non_matmul_sources.py \
  --cann-ops-adv-root /path/outside/profiling/cann-ops-adv-8.1rc1 \
  --cann-ops-root /path/outside/profiling/cann-ops-8.1rc1 \
  --extracted-root /home/CCE_EXTRACT/ops_cce
```

If the public advanced-operator source is absent, fetch it explicitly once. It
is never downloaded by a campaign command:

```bash
python3 source_adapter/fetch_official_cann_ops_adv.py \
  --destination /path/outside/profiling/cann-ops-adv-8.1rc1
```

Create the finite original FASG strategy overlays outside this repository:

```bash
python3 source_adapter/prepare_fasg_strategy_overlays.py \
  --source-root /path/outside/profiling/cann-ops-adv-8.1rc1 \
  --output-parent /path/outside/profiling/fasg-overlays
```

Then build one overlay at a time. The source tag's `version.info` contains the
pre-release marker `7.7.T8.0`, while the installed 8.1.RC1 package reports
`7.7.0.1.<build>`. The build helper changes only this *overlay metadata* to
`7.7.0.1.0`, allowing the source project's own compatibility check to run; it
does not disable the check and does not alter tiling/kernel source. The emitted
build manifest records both metadata hashes.

```bash
python3 source_adapter/build_fasg_strategy_overlay.py \
  --overlay /path/outside/profiling/fasg-overlays/fasg_flashattentionscoregradtilingdeterministic \
  --build-dir /path/outside/profiling/fasg-build-deterministic \
  --cann-root /usr/local/Ascend/ascend-toolkit/latest \
  --target optiling --jobs 1
```

`--target package` is explicit and produces a source package for later
`ASCEND_CUSTOM_OPP_PATH` execution. It is not automatically installed into the
toolkit and it is not silently used by the old direct-ACLNN campaign. Each
strategy must be installed into a different, initially empty custom-OPP root:

```bash
python3 source_adapter/install_fasg_strategy_package.py \
  --build-manifest /path/outside/profiling/fasg-build-deterministic/source_candidate_build.json \
  --destination /path/outside/profiling/custom-opp-deterministic
```

The installer refuses a non-empty destination and records the package SHA-256,
strategy class, and `ASCEND_CUSTOM_OPP_PATH` root in a small manifest. It never
writes under the toolkit's OPP directory.

`run_non_matmul_candidate_campaign.py` is the only measurement controller for
these overlays. It requires exactly the three audited, overlapping original
strategy packages; their class names are checked, so another strategy cannot
silently replace one. It writes compact JSONL workload groups and creates each
FASG full-output reference under a temporary directory for the duration of one
workload; no trace or output tensor is retained after the comparison.

The separate semantic workload catalog is likewise source-aware and contains
no MatMul records:

```bash
python3 source_adapter/non_matmul_candidate_catalog.py --audit
```

It contains 11,207 deterministic legal geometries: the reviewed cross-op
coverage families plus an 11,000-shape fp16/BNSD FASG geometry lattice. This is
a fixed shape lattice, **not** a tiling-field product or random sampler. The
source registry remains eight strategies, while the three overlapping original
strategies are independently invoked for each multi-tiling shape. Before an
FASG kernel is timed, each source path is invoked once in host-side
tiling-only mode. A group is admitted only when at least two distinct raw
tilings recur in timed execution and both full outputs match the installed
reference. At normal completion, the compact groups contain exactly 20,000
formal device-event latency records. Rejected groups carry no latency datum.

After the one-time official source preparation, the normal NPU entry point is
still one full command (selecting a physical device maps it to worker device
zero):

```bash
CANN_OPS_ADV_SOURCE=/path/outside/profiling/cann-ops-adv-8.1rc1 \
  ./run_npu.sh --mode full -d 1
```

It creates eight isolated strategy overlays but builds and installs only the
three overlapping original strategy packages, one at a time, under the ignored
`.benchmark_state/` directory; it never modifies the installed toolkit. Its
only durable measurement output is
`results/non_matmul_source_candidate_v2/<contract>/progress.jsonl`. Each JSONL
row is a compact semantic workload group; its `valid_latency` array contains
only output-validated source-generated candidates. The campaign stops exactly
at 20,000 such entries, with no tile enumeration.

The pinned public source is `ascend/cann-ops` commit `c214b710edbe24017dc7dc92170a50bd8ff38171`, selected because it predates the installed CANN 8.1.RC1 build.  The source tree itself and every build artifact stay outside this repository.

If this official sparse checkout is not already present, fetch it explicitly once (it is never fetched by `run_npu.sh`):

```bash
python3 source_adapter/fetch_official_cann_ops.py \
  --destination /path/outside/profiling/cann-ops-8.1rc1
```

For MatMulV3, the official `ALL` path is **not** a candidate list: it stops at its first successful heuristic.  A correct comparison invokes each of the source-defined routes (`BASE`, `SINGLE_CORE_SPLIT_K`, `DETERMINISTIC_SPLIT_K`) in separate original tiling contexts, deduplicates only exact raw tilings, and replays each unchanged tiling with its matching original kernel.  A candidate is kept only after output comparison against the original operator succeeds.

No code may change `DoSelectTiling`, alter raw tiling fields, invent a tile, or use callback/RuntimeKb/CCE data to select a candidate.  Operators lacking a native 910B source route are blocked rather than relabelled as supported.

Run the read-only prerequisite check before creating any build overlay:

```bash
python3 source_adapter/audit_sources.py \
  --official-root /path/to/cann-ops-8.1rc1 \
  --extracted-root /home/CCE_EXTRACT/ops_cce
```

For each original MatMulV3 route, create a separate, disposable worktree outside the repository.  This does not compile or use an NPU; it preserves the original `mat_mul_v3_base_tiling.cpp` hash and replaces only the registration wrapper.

```bash
python3 source_adapter/prepare_matmul_route_overlay.py \
  --official-root /path/to/cann-ops-8.1rc1 \
  --output /path/to/route-base \
  --route BASE
```
