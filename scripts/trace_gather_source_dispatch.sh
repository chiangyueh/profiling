#!/usr/bin/env bash
# One real GatherElements custom-OPP dispatch trace. This is deliberately
# separate from run_npu.sh: it performs exactly one generic compiler launch.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHYSICAL_DEVICE=""

usage() {
    cat <<'USAGE'
Usage: profiling/scripts/trace_gather_source_dispatch.sh -d PHYSICAL_NPU_ID

Runs exactly one real-NPU GatherElements call through
aclopCompileAndExecute("GatherElements") and traces the files that process
opens. It does not run the 5,000-record campaign, use timeout, kill a
process, reset an NPU, or modify installed CANN files.

It creates one private probe directory under profiling/.benchmark_state and
prints the decisive paths and source-audit result to the terminal. Paste the
complete terminal output back for diagnosis.
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
command -v strace >/dev/null 2>&1 || {
    echo "fatal: strace is required for this path-evidence probe and is not installed; no NPU call was made." >&2
    exit 1
}

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
TRACE_PATH="${PROBE_DIR}/open_exec.trace"
RUNNER_LOG="${PROBE_DIR}/runner.log"
mkdir -p "${OVERLAY_PARENT}" "${RUNNER_BUILD}"

echo "===== GatherElements single-dispatch runtime probe ====="
echo "physical_device=${PHYSICAL_DEVICE}"
echo "worker_logical_device=0"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "CANN_ROOT=${CANN_ROOT}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH}"
echo "installed_source=${NATIVE_SOURCE}"
echo "probe_directory=${PROBE_DIR}"
echo "operation=GatherElements shape=[64] index_shape=[17] axis=0 dtype=fp16 index_dtype=int32"
echo "dispatch_api=aclopCompileAndExecute"
echo "This makes one NPU launch. It does not use timeout, kill, reset, or a full campaign."

python3 "${ROOT}/source_adapter/check_gather_dispatch_contract.py" \
    --cann-root "${CANN_ROOT}" --runner-source "${ROOT}/multi_op_bench/runner.cpp" \
    --runner-cmake "${ROOT}/multi_op_bench/CMakeLists.txt"

echo "PRIVATE_OVERLAY_PREPARE_BEGIN"
python3 "${ROOT}/source_adapter/prepare_gather_elements_native_dynamic.py" \
    --cann-root "${CANN_ROOT}" --output-parent "${OVERLAY_PARENT}" | tee "${PROBE_DIR}/overlay_manifest.json"
echo "PRIVATE_OVERLAY_PREPARE_END"

OVERLAY="${OVERLAY_PARENT}/gather_elements_native_dynamic"
PACKAGE_MANIFEST="${OVERLAY}/native_dynamic_overlay.json"
[[ -f "${PACKAGE_MANIFEST}" ]] || { echo "fatal: private overlay manifest was not created" >&2; exit 1; }
python3 "${ROOT}/source_adapter/check_gather_dispatch_contract.py" \
    --cann-root "${CANN_ROOT}" --runner-source "${ROOT}/multi_op_bench/runner.cpp" \
    --runner-cmake "${ROOT}/multi_op_bench/CMakeLists.txt" --overlay-manifest "${PACKAGE_MANIFEST}"

readarray -t OVERLAY_VALUES < <(python3 - "${PACKAGE_MANIFEST}" <<'PY'
import json
import sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("vendor_root", "source_file", "custom_opp_root"):
    print(item[key])
PY
)
VENDOR_ROOT="${OVERLAY_VALUES[0]}"
PRIVATE_SOURCE="${OVERLAY_VALUES[1]}"
CUSTOM_OPP_ROOT="${OVERLAY_VALUES[2]}"
[[ -f "${PRIVATE_SOURCE}" ]] || { echo "fatal: private GatherElements source is absent: ${PRIVATE_SOURCE}" >&2; exit 1; }

echo "private_custom_opp_root=${CUSTOM_OPP_ROOT}"
echo "ASCEND_CUSTOM_OPP_PATH=${VENDOR_ROOT}"
echo "private_source=${PRIVATE_SOURCE}"
echo "private_config=${VENDOR_ROOT}/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json"

cmake -S "${ROOT}/multi_op_bench" -B "${RUNNER_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}"
cmake --build "${RUNNER_BUILD}" --target multi_op_npu_runner --parallel 1
RUNNER="${RUNNER_BUILD}/multi_op_npu_runner"
[[ -x "${RUNNER}" ]] || { echo "fatal: runner was not built: ${RUNNER}" >&2; exit 1; }

echo "===== linked runtime libraries ====="
ldd "${RUNNER}" | rg 'libacl_op_compiler|libascendcl|libopapi|libnnopbase' || true

export ASCEND_CUSTOM_OPP_PATH="${VENDOR_ROOT}"
export GATHER_ELEMENTS_TILING_AUDIT_PATH="${AUDIT_PATH}"
export GATHER_ELEMENTS_SOURCE_DISPATCH="aclop_compile_and_execute"
export GATHER_ELEMENTS_SOURCE_AIV_CAP=20
export GATHER_ELEMENTS_SOURCE_UB_DIVISOR=1

COMMAND=("${RUNNER}"
    --workload-id gather_elements_000 --op gather_elements --device 0
    --warmup 0 --samples 0 --expected-soc Ascend910B3
    --shape 64 --index-shape 17 --axis 0 --dtype fp16 --index-dtype int32
    --source-tiling-only 1)

echo "===== exact environment supplied to the one source dispatch ====="
printf 'ASCEND_OPP_PATH=%s\nASCEND_CUSTOM_OPP_PATH=%s\nGATHER_ELEMENTS_TILING_AUDIT_PATH=%s\nGATHER_ELEMENTS_SOURCE_DISPATCH=%s\nGATHER_ELEMENTS_SOURCE_AIV_CAP=%s\nGATHER_ELEMENTS_SOURCE_UB_DIVISOR=%s\n' \
    "${ASCEND_OPP_PATH}" "${ASCEND_CUSTOM_OPP_PATH}" "${GATHER_ELEMENTS_TILING_AUDIT_PATH}" \
    "${GATHER_ELEMENTS_SOURCE_DISPATCH}" "${GATHER_ELEMENTS_SOURCE_AIV_CAP}" "${GATHER_ELEMENTS_SOURCE_UB_DIVISOR}"
echo "===== one generic-dispatch execution begins ====="
printf 'command='
printf ' %q' "${COMMAND[@]}"
printf '\n'

# This is tracing only. There is deliberately no timeout wrapper or forced
# kill. The runner return code is retained for diagnosis after a failed call.
set +e
strace -f -qq -s 512 -yy -e trace=open,openat,execve -o "${TRACE_PATH}" "${COMMAND[@]}" 2>&1 | tee "${RUNNER_LOG}"
RUNNER_RC=${PIPESTATUS[0]}
set -e
echo "===== one generic-dispatch execution ended rc=${RUNNER_RC} ====="

echo "===== source audit file ====="
if [[ -f "${AUDIT_PATH}" ]]; then
    cat "${AUDIT_PATH}"
else
    echo "MISSING: ${AUDIT_PATH}"
fi

echo "===== traced paths relevant to generic OPP dispatch (up to 160 lines) ====="
rg -n -F -e "${PRIVATE_SOURCE}" -e "${NATIVE_SOURCE}" -e "${VENDOR_ROOT}" \
    -e "aic-ascend910b-ops-info.json" -e "libacl_op_compiler.so" "${TRACE_PATH}" | tail -160 || true

python3 - "${RUNNER_LOG}" "${AUDIT_PATH}" "${TRACE_PATH}" "${PRIVATE_SOURCE}" "${NATIVE_SOURCE}" "${RUNNER_RC}" <<'PY'
import json
import sys
from pathlib import Path

runner_log, audit_path, trace_path, private_source, installed_source, runner_rc = sys.argv[1:]
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
trace = Path(trace_path).read_text(encoding="utf-8", errors="replace")
private_open_count = trace.count(private_source)
installed_open_count = trace.count(installed_source)
audit_matches = any(
    row.get("aiv_core_cap") == 20 and row.get("ub_cap_divisor") == 1 and
    row.get("shape") == [64] and row.get("index_shape") == [17] and
    row.get("axis") == 0 and row.get("dtype") == "float16" and
    row.get("index_dtype") == "int32" and row.get("status") == 0
    for row in audit_rows
)
runner_success = isinstance(result, dict) and result.get("status") == "success"
generic_route = isinstance(result, dict) and result.get("backend") == "acl_op_compiler_custom_opp_real_npu"
if int(runner_rc) == 0 and runner_success and generic_route and audit_matches:
    verdict = "source_selected_and_audited"
elif int(runner_rc) == 0 and runner_success and generic_route:
    verdict = "generic_dispatch_completed_but_private_source_not_proven"
else:
    verdict = "generic_dispatch_failed_before_source_selection_was_proven"
print("GATHER_ELEMENTS_RUNTIME_DISPATCH_PROBE " + json.dumps({
    "verdict": verdict,
    "runner_return_code": int(runner_rc),
    "runner_result": result,
    "audit_row_count": len(audit_rows),
    "audit_expected_context_present": audit_matches,
    "trace_private_source_path_occurrences": private_open_count,
    "trace_installed_source_path_occurrences": installed_open_count,
    "evidence_rule": "only an expected private-source audit row proves selection; the generic backend alone proves only the requested API route",
}, sort_keys=True))
PY

echo "probe_artifacts_retained=${PROBE_DIR}"
echo "Paste the complete terminal output above; do not start the full campaign from this probe."
