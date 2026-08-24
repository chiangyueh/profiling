#!/usr/bin/env bash
# One real GatherElements custom-OPP dispatch trace. This is deliberately
# separate from run_npu.sh: it performs exactly one generic compiler launch.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHYSICAL_DEVICE=""

usage() {
    cat <<'USAGE'
Usage: profiling/scripts/trace_gather_source_dispatch.sh -d PHYSICAL_NPU_ID

Runs exactly one real-NPU GatherElements call through the registered type
``GatherElements`` selected by a private OPP vendor-priority overlay. The
private source records the actual Python module path when CANN imports it. It does not run the
5,000-record campaign, use timeout, kill a process, reset an NPU, or modify
installed CANN files.

It creates one private probe directory under profiling/.benchmark_state. The
terminal shows only the decisive status, NPU result, source audit, and verdict.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--device) PHYSICAL_DEVICE="${2:?missing physical NPU ID}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "fatal: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${PHYSICAL_DEVICE}" ]] || { echo "fatal: pass the physical NPU explicitly, for example: -d 2" >&2; exit 2; }
[[ "${PHYSICAL_DEVICE}" =~ ^[0-9]+$ ]] || { echo "fatal: device must be a non-negative integer" >&2; exit 2; }
[[ -e "/dev/davinci${PHYSICAL_DEVICE}" ]] || { echo "fatal: physical NPU device node is absent: /dev/davinci${PHYSICAL_DEVICE}" >&2; exit 1; }

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
[[ -d "${CANN_ROOT}" && -f "${CANN_ROOT}/opp/version.info" ]] || {
    echo "fatal: CANN root or OPP package is missing: ${CANN_ROOT}" >&2
    exit 1
}
ENV_FILE=""
for candidate in "${CANN_ROOT}/set_env.sh" "$(dirname "${CANN_ROOT}")/set_env.sh"; do
    [[ -f "${candidate}" ]] && { ENV_FILE="${candidate}"; break; }
done
[[ -n "${ENV_FILE}" ]] || { echo "fatal: CANN environment script is missing under ${CANN_ROOT}" >&2; exit 1; }

set +u
source "${ENV_FILE}"
set -u
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
# The runner uses logical device zero after the physical card is masked.
export ASCEND_RT_VISIBLE_DEVICES="${PHYSICAL_DEVICE}"
export TILINGKEY_PAR_COMPILE=1
export OMP_NUM_THREADS=1
unset ASCEND_CUSTOM_OPP_PATH ASCENDC_CPU_DEBUG

NATIVE_SOURCE="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/gather_elements.py"
[[ -f "${NATIVE_SOURCE}" ]] || { echo "fatal: installed GatherElements dynamic source is absent: ${NATIVE_SOURCE}" >&2; exit 1; }

PROBE_FINGERPRINT="$(
    sha256sum "${ROOT}/multi_op_bench/runner.cpp" "${ROOT}/multi_op_bench/CMakeLists.txt" \
        "${ROOT}/source_adapter/prepare_gather_elements_native_dynamic.py" \
        "${ROOT}/source_adapter/check_gather_dispatch_contract.py" "${NATIVE_SOURCE}" \
        "${CANN_ROOT}/opp/version.info" | sha256sum | cut -c1-16
)"
PROBE_DIR="${ROOT}/.benchmark_state/gather_elements_runtime_dispatch_probe/${PROBE_FINGERPRINT}_$(date -u +%Y%m%dT%H%M%SZ)"
OVERLAY_PARENT="${PROBE_DIR}/overlays"
RUNNER_BUILD="${PROBE_DIR}/runner_build"
AUDIT_PATH="${PROBE_DIR}/source_audit.jsonl"
RUNNER_LOG="${PROBE_DIR}/runner.log"
mkdir -p "${OVERLAY_PARENT}" "${RUNNER_BUILD}"

show_failure() {
    local stage="$1"
    local log="$2"
    echo "GATHER_PROBE_FAILURE stage=${stage} log=${log}" >&2
    tail -n 30 "${log}" >&2 || true
    exit 1
}

echo "GATHER_PROBE_BEGIN physical_device=${PHYSICAL_DEVICE} logical_device=0 op=GatherElements shape=64 index_shape=17 axis=0 dtype=fp16"

STATIC_PRE_LOG="${PROBE_DIR}/static_pre.log"
if ! python3 "${ROOT}/source_adapter/check_gather_dispatch_contract.py" \
    --cann-root "${CANN_ROOT}" --runner-source "${ROOT}/multi_op_bench/runner.cpp" \
    --runner-cmake "${ROOT}/multi_op_bench/CMakeLists.txt" \
    --campaign-source "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
    --launch-script "${ROOT}/scripts/trace_gather_source_dispatch.sh" >"${STATIC_PRE_LOG}" 2>&1; then
    show_failure "static_contract_pre" "${STATIC_PRE_LOG}"
fi
echo "GATHER_PROBE_STATIC status=passed"

OVERLAY_PREPARE_LOG="${PROBE_DIR}/overlay_prepare.log"
if ! python3 "${ROOT}/source_adapter/prepare_gather_elements_native_dynamic.py" \
    --cann-root "${CANN_ROOT}" --output-parent "${OVERLAY_PARENT}" >"${OVERLAY_PREPARE_LOG}" 2>&1; then
    show_failure "private_overlay_prepare" "${OVERLAY_PREPARE_LOG}"
fi

OVERLAY="${OVERLAY_PARENT}/gather_elements_native_dynamic"
PACKAGE_MANIFEST="${OVERLAY}/native_dynamic_overlay.json"
[[ -f "${PACKAGE_MANIFEST}" ]] || { echo "fatal: private overlay manifest was not created" >&2; exit 1; }
STATIC_POST_LOG="${PROBE_DIR}/static_post.log"
if ! python3 "${ROOT}/source_adapter/check_gather_dispatch_contract.py" \
    --cann-root "${CANN_ROOT}" --runner-source "${ROOT}/multi_op_bench/runner.cpp" \
    --runner-cmake "${ROOT}/multi_op_bench/CMakeLists.txt" \
    --campaign-source "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
    --launch-script "${ROOT}/scripts/trace_gather_source_dispatch.sh" \
    --overlay-manifest "${PACKAGE_MANIFEST}" >"${STATIC_POST_LOG}" 2>&1; then
    show_failure "static_contract_post" "${STATIC_POST_LOG}"
fi

readarray -t OVERLAY_VALUES < <(python3 - "${PACKAGE_MANIFEST}" <<'PY'
import json
import sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("vendor_root", "source_file", "runtime_opp_root", "source_operator_type"):
    print(item[key])
PY
)
VENDOR_ROOT="${OVERLAY_VALUES[0]}"
PRIVATE_SOURCE="${OVERLAY_VALUES[1]}"
RUNTIME_OPP_ROOT="${OVERLAY_VALUES[2]}"
SOURCE_OPERATOR_TYPE="${OVERLAY_VALUES[3]}"
[[ -f "${PRIVATE_SOURCE}" ]] || { echo "fatal: private GatherElements source is absent: ${PRIVATE_SOURCE}" >&2; exit 1; }

echo "GATHER_PROBE_OVERLAY status=passed source_operator_type=${SOURCE_OPERATOR_TYPE}"

CMAKE_CONFIGURE_LOG="${PROBE_DIR}/cmake_configure.log"
if ! cmake -S "${ROOT}/multi_op_bench" -B "${RUNNER_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" >"${CMAKE_CONFIGURE_LOG}" 2>&1; then
    show_failure "runner_configure" "${CMAKE_CONFIGURE_LOG}"
fi
CMAKE_BUILD_LOG="${PROBE_DIR}/cmake_build.log"
if ! cmake --build "${RUNNER_BUILD}" --target multi_op_npu_runner --parallel 1 >"${CMAKE_BUILD_LOG}" 2>&1; then
    show_failure "runner_build" "${CMAKE_BUILD_LOG}"
fi
RUNNER="${RUNNER_BUILD}/multi_op_npu_runner"
[[ -x "${RUNNER}" ]] || { echo "fatal: runner was not built: ${RUNNER}" >&2; exit 1; }
echo "GATHER_PROBE_BUILD status=passed"

unset ASCEND_CUSTOM_OPP_PATH
export ASCEND_OPP_PATH="${RUNTIME_OPP_ROOT}"
export GATHER_ELEMENTS_TILING_AUDIT_PATH="${AUDIT_PATH}"
export GATHER_ELEMENTS_SOURCE_DISPATCH="aclop_compile_and_execute"
export GATHER_ELEMENTS_SOURCE_AIV_CAP=20
export GATHER_ELEMENTS_SOURCE_UB_DIVISOR=1

COMMAND=("${RUNNER}"
    --workload-id gather_elements_000 --op gather_elements --device 0
    --warmup 0 --samples 0 --expected-soc Ascend910B3
    --shape 64 --index-shape 17 --axis 0 --dtype fp16 --index-dtype int32
    --source-tiling-only 1 --normal-cleanup 1)

echo "GATHER_PROBE_NPU_BEGIN dispatch=aclopCompileAndExecute source_aiv_cap=20 ub_cap_divisor=1"

# There is deliberately no timeout wrapper or forced kill. The private source
# writes a module_imported record with its actual __file__ before BuildCCE.
# The runner return code is retained for diagnosis after a failed call.
set +e
"${COMMAND[@]}" >"${RUNNER_LOG}" 2>&1
RUNNER_RC=$?
set -e
echo "GATHER_PROBE_NPU_END rc=${RUNNER_RC}"

if [[ -f "${AUDIT_PATH}" ]]; then
    while IFS= read -r audit_row; do
        echo "GATHER_PROBE_AUDIT ${audit_row}"
    done < "${AUDIT_PATH}"
else
    echo "GATHER_PROBE_AUDIT missing"
fi

python3 - "${RUNNER_LOG}" "${AUDIT_PATH}" "${PRIVATE_SOURCE}" "${NATIVE_SOURCE}" "${SOURCE_OPERATOR_TYPE}" "${RUNNER_RC}" <<'PY'
import json
import sys
from pathlib import Path

runner_log, audit_path, private_source, installed_source, source_operator_type, runner_rc = sys.argv[1:]
result = None
for line in Path(runner_log).read_text(encoding="utf-8", errors="replace").splitlines():
    if line.startswith("MULTIOP_NPU_RESULT "):
        try:
            result = json.loads(line.split(" ", 1)[1])
        except json.JSONDecodeError:
            pass
audit_rows = []
path = Path(audit_path)
if path.is_file():
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("schema") == "gather_elements_native_dynamic_source_observation_v1":
            audit_rows.append(row)
import_rows = [row for row in audit_rows if row.get("event") == "module_imported"]
source_import_expected_path = any(
    Path(str(row.get("source_file", ""))).resolve() == Path(private_source).resolve()
    for row in import_rows
)
audit_matches = any(
    row.get("event") == "tiling_generated" and row.get("operator_type") == source_operator_type and
    row.get("aiv_core_cap") == 20 and row.get("ub_cap_divisor") == 1 and
    row.get("shape") == [64] and row.get("index_shape") == [17] and
    row.get("axis") == 0 and row.get("dtype") == "float16" and
    row.get("index_dtype") == "int32" and row.get("status") == 0
    for row in audit_rows
)
runner_success = isinstance(result, dict) and result.get("status") == "success"
generic_route = isinstance(result, dict) and result.get("backend") == "acl_op_compiler_private_opp_source_real_npu"
if int(runner_rc) == 0 and runner_success and generic_route and audit_matches:
    verdict = "source_selected_and_audited"
elif source_import_expected_path:
    verdict = "private_source_imported_but_dispatch_did_not_complete"
elif int(runner_rc) == 0 and runner_success and generic_route:
    verdict = "generic_dispatch_completed_but_private_source_not_proven"
else:
    verdict = "generic_dispatch_failed_before_source_selection_was_proven"
error_text = None if not isinstance(result, dict) else result.get("error")
if isinstance(error_text, str):
    error_text = " ".join(error_text.split())[:500]
runner_summary = None if not isinstance(result, dict) else {
    "status": result.get("status"), "backend": result.get("backend"), "error": error_text,
}
print("GATHER_ELEMENTS_RUNTIME_DISPATCH_PROBE " + json.dumps({
    "verdict": verdict,
    "runner_return_code": int(runner_rc),
    "runner": runner_summary,
    "audit_row_count": len(audit_rows),
    "source_import_row_count": len(import_rows),
    "source_import_expected_private_path": source_import_expected_path,
    "audit_expected_context_present": audit_matches,
    "source_operator_type": source_operator_type,
    "evidence_rule": "module_imported records the actual Python source path; only a matching tiling_generated row proves a completed source build",
}, sort_keys=True))
PY
