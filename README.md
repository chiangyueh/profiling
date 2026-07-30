# CANN 8.5.0 MatMulV3 Tiling Search

This repository is a focused copy of the official CANN `ops-nn` 8.5.0
MatMulV3 operator. The upstream operator is preserved under:

```text
matmul/mat_mul_v3/
```

The independent tiling-search extension is isolated under:

```text
matmul/mat_mul_v3/op_host/op_tiling/research/
```

No operator kernel, API, numerical implementation, or official tiling source
is modified. The repository also contains the upstream `matmul/common`
dependency used by MatMulV3; unrelated operators and the old profiling search
implementation are intentionally excluded.

## Full NPU Run

From the directory containing the clone:

```bash
chmod +x profiling/run_npu.sh
profiling/run_npu.sh --mode full
```

The full mode:

1. detects the installed CANN runtime and Ascend SoC;
2. generates independent hardware-contract tilings;
3. requires exact official callback and RuntimeKb roundtrips;
4. measures official, bank control, and candidate schedules in paired runs;
5. persists exact measurements so completed schedules are not repeated.

By default, up to 40 callback-accepted candidates are sent to the NPU per
workload. Results are written under `results/`.

See [UPSTREAM_8.5.0.md](UPSTREAM_8.5.0.md) for provenance and runtime
compatibility, and
[research/README.md](matmul/mat_mul_v3/op_host/op_tiling/research/README.md)
for the search architecture.
