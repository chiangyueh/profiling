# Upstream Baseline

This repository contains the official CANN `ops-nn` MatMulV3 source from:

- Repository: `https://gitcode.com/cann/ops-nn.git`
- Tag/branch: `8.5.0`
- Commit: `9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef`
- Operator path: `matmul/mat_mul_v3`

The official operator source, kernel source, tests, examples, configuration,
and RuntimeKb files are copied without modification. The research extension is
isolated under:

`matmul/mat_mul_v3/op_host/op_tiling/research/`

The only additional root entry point is `run_npu.sh`.

The research path generates the official 23-field `MatMulV3TunnerTiling`
record. It does not change `op_kernel`, operator API behavior, or numerical
semantics.

## Runtime Compatibility

Building and installing the complete upstream 8.5.0 operator package requires
a CANN 8.5 toolkit as documented by the upstream `ops-nn` repository. This
focused repository intentionally omits unrelated operators and the top-level
packaging framework.

`run_npu.sh` executes the installed `aclnnMatmul` backend. On a machine that
still has CANN 8.1, it therefore measures the installed 8.1 kernel, while using
the 8.5.0 source as the research baseline. This compatibility mode is allowed
only because every candidate must pass:

1. the official 8.5 source-schema check;
2. installed callback exact roundtrip;
3. installed RuntimeKb lookup;
4. isolated NPU output-coverage preflight.

The script prints both the source baseline and installed runtime version. It
does not report an 8.1 execution as an 8.5 kernel measurement.
