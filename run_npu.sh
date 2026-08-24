#!/usr/bin/env bash
set -Eeuo pipefail

# One operator, one execution route: native CANN-8.1 GatherElements dynamic
# source. The source overlay is private to this checkout; installed CANN is
# only read through immutable links.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-1}"
WARMUP="${OP_NPU_WARMUP:-2}"
SAMPLES="${OP_NPU_SAMPLES:-5}"
RECORD_TARGET=5000

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs GatherElements only. The installed reference uses ``aclnnGather``; each
source candidate uses CANN's generic ``aclopCompileAndExecute`` OPP dispatch,
which is the API that can select the checkout-local dynamic source overlay.

- 5,000 real-NPU device-event records (250 complete 20-candidate groups).
- Each admitted candidate launches and output-matches an installed reference.
- Candidate axes only lower the native source's published AIV/UB budgets.
- Results are rotating JSONL files under profiling/results, each <= 50 MiB.
- No installed CANN file, global OPP path, device state, process, or reset is
  modified by this script.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
        -d|--device) PHYSICAL_DEVICE="${2:?missing physical NPU ID}"; shift 2 ;;
        --warmup) WARMUP="${2:?missing warmup count}"; shift 2 ;;
        --samples) SAMPLES="${2:?missing sample count}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "fatal: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${MODE}" != "full" ]]; then
    echo "fatal: only --mode full is supported" >&2
    exit 2
fi
for value_name in PHYSICAL_DEVICE WARMUP SAMPLES; do
    value="${!value_name}"
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "fatal: ${value_name} must be a non-negative integer" >&2; exit 2; }
done
(( SAMPLES >= 1 )) || { echo "fatal: samples must be at least 1" >&2; exit 2; }
[[ -e "/dev/davinci${PHYSICAL_DEVICE}" ]] || { echo "fatal: physical NPU device node is absent: /dev/davinci${PHYSICAL_DEVICE}" >&2; exit 1; }

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
[[ -d "${CANN_ROOT}" && -f "${CANN_ROOT}/opp/version.info" ]] || { echo "fatal: CANN root or OPP package is missing: ${CANN_ROOT}" >&2; exit 1; }
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
export ASCEND_RT_VISIBLE_DEVICES="${PHYSICAL_DEVICE}"
export TILINGKEY_PAR_COMPILE=1
export OMP_NUM_THREADS=1
unset ASCEND_CUSTOM_OPP_PATH ASCENDC_CPU_DEBUG

NATIVE_SOURCE="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/gather_elements.py"
[[ -f "${NATIVE_SOURCE}" ]] || { echo "fatal: installed native CANN GatherElements source is absent" >&2; exit 1; }

SOURCE_ID="$({
    sha256sum "${ROOT}/run_npu.sh" "${ROOT}/multi_op_bench/CMakeLists.txt" "${ROOT}/multi_op_bench/runner.cpp" \
        "${ROOT}/source_adapter/non_matmul_candidate_catalog.py" \
        "${ROOT}/source_adapter/check_gather_dispatch_contract.py" \
        "${ROOT}/source_adapter/prepare_gather_elements_native_dynamic.py" \
        "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py"
    sha256sum "${NATIVE_SOURCE}" "${CANN_ROOT}/opp/version.info"
    readlink -f "${CANN_ROOT}"
} | sha256sum | cut -c1-20)"

STATE="${ROOT}/.benchmark_state/gather_elements_native_dynamic_v5/${SOURCE_ID}"
OVERLAY_PARENT="${STATE}/overlays"
RUNNER_BUILD="${STATE}/runner_build"
RESULTS="${ROOT}/results/gather_elements_native_dynamic_v5/${SOURCE_ID}"
LOGS="${RESULTS}/logs"
mkdir -p "${OVERLAY_PARENT}" "${RUNNER_BUILD}" "${LOGS}"

echo "GatherElements native CANN dynamic-source campaign"
echo "  target:        physical NPU ${PHYSICAL_DEVICE} -> worker logical NPU 0"
echo "  formal target: ${RECORD_TARGET} real-NPU latency records"
echo "  source:        ${NATIVE_SOURCE}"
echo "  source API:    aclopCompileAndExecute(GatherElementsSourceCandidate), a non-colliding private CANN custom-OPP type"
echo "  reference API: installed aclnnGather under the unmodified installed OPP path"
echo "  output:        ${LOGS}/1.log, 2.log, ... (JSONL, each <= 50 MiB)"

# A wrong API path was the cause of the prior no-audit failures.  Check every
# local prerequisite before creating a build directory or touching the NPU.
python3 "${ROOT}/source_adapter/check_gather_dispatch_contract.py" \
    --cann-root "${CANN_ROOT}" --runner-source "${ROOT}/multi_op_bench/runner.cpp" \
    --runner-cmake "${ROOT}/multi_op_bench/CMakeLists.txt"

OVERLAY="${OVERLAY_PARENT}/gather_elements_native_dynamic"
PACKAGE_MANIFEST="${OVERLAY}/native_dynamic_overlay.json"
if [[ ! -f "${PACKAGE_MANIFEST}" ]]; then
    [[ ! -e "${OVERLAY}" ]] || { echo "fatal: incomplete private native GatherElements overlay exists: ${OVERLAY}" >&2; exit 2; }
    echo "NATIVE_DYNAMIC_OVERLAY_PREPARE_BEGIN operator=gather_elements"
    python3 "${ROOT}/source_adapter/prepare_gather_elements_native_dynamic.py" \
        --cann-root "${CANN_ROOT}" --output-parent "${OVERLAY_PARENT}" >"${STATE}/overlay_prepare.log" 2>&1
    echo "NATIVE_DYNAMIC_OVERLAY_PREPARE_END operator=gather_elements"
fi
python3 "${ROOT}/source_adapter/check_gather_dispatch_contract.py" \
    --cann-root "${CANN_ROOT}" --runner-source "${ROOT}/multi_op_bench/runner.cpp" \
    --runner-cmake "${ROOT}/multi_op_bench/CMakeLists.txt" \
    --overlay-manifest "${PACKAGE_MANIFEST}"

cmake -S "${ROOT}/multi_op_bench" -B "${RUNNER_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}"
cmake --build "${RUNNER_BUILD}" --target multi_op_npu_runner --parallel 1

# The controller's sole preflight performs one installed-reference launch and
# one generic custom-source launch/audit, then stops immediately on failure.
# Keeping that gate in one place avoids duplicate NPU smoke executions.
PYTHONPATH="${ROOT}/multi_op_bench" python3 "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
    --runner "${RUNNER_BUILD}/multi_op_npu_runner" --log-dir "${LOGS}" --device 0 \
    --warmup "${WARMUP}" --samples "${SAMPLES}" --operator gather_elements \
    --record-target "${RECORD_TARGET}" --source-package-manifest "${PACKAGE_MANIFEST}"
