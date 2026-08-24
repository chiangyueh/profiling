# GatherElements native dynamic-source collector

The active NPU entry point collects **GatherElements only** on Ascend910B3:

```bash
profiling/run_npu.sh --mode full -d 2
```

It does not build, load, or otherwise use a CANN 8.3 `GatherElementsV2`
custom operator. That compatibility route was removed because a CANN 8.3
host-tiler ABI cannot safely run inside the installed CANN 8.1 runtime.

## Execution route

The collector uses the CANN 8.1 `GatherElements` dynamic Python source from
the same installation that provides `aclnnGather`. Before execution it makes
a private vendor overlay under `.benchmark_state/`:

- only a patched copy of `<vendor>_impl/dynamic/gather_elements.py` and its
  one config record live in the private vendor directory;
- that source is placed in CANN's required `<vendor>_impl` directory, exactly
  as the installed CANN 8.1 custom-package template does;
- each source worker keeps `ASCEND_OPP_PATH` pointed at the installed CANN
  OPP tree and sets `ASCEND_CUSTOM_OPP_PATH` to that one private vendor
  directory, which is CANN 8.1's documented custom-operator loader layout;
- the installed CANN tree, global environment, device state, and processes
  are never modified.

The source copy emits an audit only after its original `BuildCCE` returns. A
candidate therefore counts only when the source was selected, the normal
`aclnnGather` call launched on the NPU, and the output exactly matched the
same call under the unmodified installed OPP path.

## Candidate and data contract

Each semantic shape starts with the finite native-source core budgets 1..20.
If fewer than twenty source contexts complete, selected successful contexts
are additionally evaluated with native-source visible-UB divisors 2, 4, and
8. These are bounded inputs to the original source's branch selection and
compile-info publication; no generated flow-table field, tiling key, block
dimension, workspace, or output is edited or replayed.

One admitted shape contributes exactly 20 output-validated device-event
latency records. The 5,000-record target therefore needs 250 admitted
shapes. The catalog contains 404 source-supported deterministic legal
GatherElements shapes. Logs are append-only JSONL below
`results/gather_elements_native_dynamic_v3/<contract>/logs/`; each numbered
data log is capped at 50 MiB. The collector reads no CCE data, historic
latency/tiling records, RuntimeKb, callbacks, or cost model.

The automatic preflight is one real normal-core source launch and stream
synchronization. It is not a host timeout and it does not kill or reset an
NPU.
