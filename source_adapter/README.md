# Source-preserving multi-tiling adapter

This directory is a guardrail for the next measurement path.  It does not contain a cost model, prior measurements, RuntimeKb replay, callback timing, or CCE data.

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
