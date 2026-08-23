# Non-MatMul source-tiling collector

This is a data-collection path for ranking legal tilings of four non-MatMul
operators on Ascend910B3:

- FlashAttentionScoreGrad — all eight original CANN 8.1 registered strategies.
- FusedInferAttentionScore — its original decode/prefill dispatcher.
- GatherElementsV2 — extracted CANN 8.3 source in a clearly labelled CANN 8.1
  build compatibility layer.
- ScatterElementsV2 — its original CANN 8.1 last-axis tiler.

MatMul is deliberately excluded. Transpose and GatherV2 remain excluded because
there is no matching 910B source route; they are not relabelled as supported.

## Collection contract

For every semantic shape, the controller first invokes the complete finite set
of original source contexts: every retained source strategy/dispatcher and AIV
core budget 1..20. A candidate is the raw tiling emitted by that source, not a
manually constructed set of tile fields.

If and only if the complete original set yields fewer than 20 distinct raw
identities, the controller revisits selected original contexts using only the
operator-declared source-visible capacity envelope:

- FASG: L2 scheduling capacity;
- FIAS, GatherElements, ScatterElements: UB capacity.

The only heuristic values are divisors 2, 4, and 8. They lower the resource
visible to the original tiler before it calculates its own fields, so a plan
cannot demand more capacity than hardware provides. No output tiling field,
tiling key, block dimension, or workspace is edited after source generation.

A shape is admitted only when at least 20 distinct candidates execute, exactly
match an installed-operator output reference, and complete device-event
measurement. A failed candidate is recorded as rejected and does not count;
it does not erase the other legal candidates. The global formal-record ceiling
is 20,000, partitioned 6,000 / 6,000 / 4,000 / 4,000 across FASG, FIAS,
GatherElements and ScatterElements.

The collector reads neither CCE data, historic latency/tiling records,
RuntimeKb, callbacks, nor a cost model. Full reference tensors live only in a
temporary directory; the durable output is compact JSONL.

## Sources and compatibility limits

The source pins are in `non_matmul_source_lock.json`.

- FASG/FIAS use public `cann-ops-adv` 8.1 RC1 source and the pinned public
  `cann-ops` 8.1 build harness.
- ScatterElements uses pinned public `cann-ops` 8.1 source.
- Public 8.1 source does not provide the needed GatherElements route. Its
  extracted source reports CANN 8.3 RC2. The compatibility preparer copies it
  into a pinned 8.1 build parent and changes only the Ascend910B registration
  scope, CMake target wiring, two missing 8.1-compatible logging/arithmetic
  headers, and observational audit/resource inputs. It is not claimed to be
  native 8.1; package build and exact real-NPU output equality are both hard
  gates.

The advanced source tree has original op-host sources but no top-level CMake
project. Its detached overlay uses the pinned public 8.1 build harness,
selects only the requested op-host directories, and copies one unchanged,
hash-attested public packaging helper (`gen_ops_filter.sh`). This is build
plumbing only; it does not alter a tiler or a kernel.

No build or package is installed into the toolkit. Each source overlay builds
only its host tiler into an isolated custom OPP root under ignored state. The
root copies the exact installed dynamic device source/configuration for that
operator, so CANN compiles only the actually launched tiling key; it does not
eagerly precompile the release matrix of keys.

For FASG, the eight isolated overlays have different host tiling registrations.
Each root carries its own source-built host tiler and the same exact installed
dynamic device source. It neither shares a tiler nor modifies device code.

## Run

The normal entry point is one command. It uses physical device 1 by default
and maps it to worker logical device 0.

```bash
./run_npu.sh --mode full -d 1
```

Set `CANN_OPS_ADV_SOURCE`, `CANN_OPS_SOURCE`, or
`CANN_GATHER_ELEMENTS_EXTRACT_SOURCE` only when their defaults are unavailable.
The command builds serially and does not impose a host-side timeout or kill a
worker. Results are written below
`results/non_matmul_source_candidate_v5/<contract>/progress.jsonl`; generated
sources/builds remain below ignored `.benchmark_state/`.
