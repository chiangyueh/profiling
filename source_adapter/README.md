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
  other registrations. Strategy code, predicates, dispatchers, and kernels are
  hash-locked.
- A candidate must later execute and pass output comparison before latency is
  admitted to the training set.

The public 8.1.RC1 advanced-operator tag has eight registered
`FlashAttentionScoreGrad` strategies. This is the actual count in that pinned
source—not the larger count in a newer extracted tree. The collector creates an
isolated overlay for each. Whether one succeeds for a specific workload is
decided by its unmodified original legality checks.

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
toolkit and it is not silently used by the old direct-ACLNN campaign.

The separate semantic workload catalog is likewise source-aware and contains
no MatMul records:

```bash
python3 source_adapter/non_matmul_candidate_catalog.py --audit
```

It contains 207 explicit, non-random workload geometries. The maximum is 648
original-source strategy attempts (63 FASG geometries × 8 registered original
strategies, plus source-native single-path workloads); the actual retained
count is lower because unsuccessful original strategies and failed output
comparisons are recorded as rejections, not converted into synthetic tilings.
The campaign-wide hard ceiling remains 20,000 records.

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
