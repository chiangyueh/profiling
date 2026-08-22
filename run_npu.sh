#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${ROOT}/multi_op_bench"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-1}"
WARMUP="${OP_NPU_WARMUP:-2}"
SAMPLES="${OP_NPU_SAMPLES:-7}"

usage() {
    cat <<'USAGE'
Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs 272 fixed, manually reviewed real-ACLNN workloads on one selected NPU:
MatMul, Transpose, GatherV2, GatherElements, ScatterElements,
FlashAttentionScoreGrad, and FusedInferAttentionScore.

The selected physical NPU is exposed as logical device 0 inside the worker.
No CPU/simulator path, random shapes, cost-model fitting, subprocess timeout,
or forced process kill exists in this runner.
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
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "fatal: ${value_name} must be a non-negative integer" >&2
        exit 2
    fi
done
if (( SAMPLES < 1 )); then
    echo "fatal: samples must be at least 1" >&2
    exit 2
fi
if [[ ! -e "/dev/davinci${PHYSICAL_DEVICE}" ]]; then
    echo "fatal: physical NPU device node is absent: /dev/davinci${PHYSICAL_DEVICE}" >&2
    exit 1
fi

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ ! -d "${CANN_ROOT}" ]]; then
    echo "fatal: CANN root does not exist: ${CANN_ROOT}" >&2
    exit 1
fi
ENV_FILE=""
for candidate in "${CANN_ROOT}/set_env.sh" "$(dirname "${CANN_ROOT}")/set_env.sh"; do
    if [[ -f "${candidate}" ]]; then
        ENV_FILE="${candidate}"
        break
    fi
done
if [[ -z "${ENV_FILE}" ]]; then
    echo "fatal: CANN environment script is missing under ${CANN_ROOT}" >&2
    exit 1
fi

set +u
source "${ENV_FILE}"
set -u
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export ASCEND_RT_VISIBLE_DEVICES="${PHYSICAL_DEVICE}"
unset ASCENDC_CPU_DEBUG

if [[ ! -f "${ASCEND_OPP_PATH}/version.info" ]]; then
    echo "fatal: OPP package under selected CANN root is missing: ${ASCEND_OPP_PATH}" >&2
    exit 1
fi

# ASCEND_RT_VISIBLE_DEVICES maps the requested physical NPU to worker device 0.
LOGICAL_DEVICE=0
SOURCE_ID="$({
    sha256sum "${BENCH}/workloads.py" "${BENCH}/run_campaign.py" \
        "${BENCH}/runner.cpp" "${BENCH}/CMakeLists.txt"
    readlink -f "${CANN_ROOT}"
} | sha256sum | cut -c1-20)"
STATE="${ROOT}/.benchmark_state/multi_op_real_npu_v2/${SOURCE_ID}"
BUILD="${STATE}/build"
RESULTS="${ROOT}/results/multi_op_real_npu_v2"
PROGRESS="${RESULTS}/progress.jsonl"
mkdir -p "${BUILD}" "${RESULTS}"

echo "Multi-op real-NPU viability and latency campaign"
echo "  target:      physical NPU ${PHYSICAL_DEVICE} -> worker logical NPU ${LOGICAL_DEVICE}"
echo "  operators:   MatMul, Transpose, GatherV2, GatherElements, ScatterElements, FlashAttentionScoreGrad, FusedInferAttentionScore"
echo "  workloads:   272 manually reviewed deterministic shapes; no random generation"
echo "  schedule:    per-op preflight; one failed op does not block the other operators"
echo "  measurement: ${WARMUP} immediate same-workload warmups + ${SAMPLES} device-event samples"
echo "  failures:    recorded per workload; no host watchdog, forced kill, CPU, or simulator fallback"
echo "  resume:      only successful exact workload/spec records in ${PROGRESS} are reused"
echo "  CANN root:   ${CANN_ROOT}"

python3 "${BENCH}/workloads.py" --audit | sed 's/^/MULTIOP_CATALOG_AUDIT /'
cmake -S "${BENCH}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}"
cmake --build "${BUILD}" --target multi_op_npu_runner --parallel "${BUILD_JOBS:-1}"

PYTHONPATH="${BENCH}" python3 "${BENCH}/run_campaign.py" \
    --runner "${BUILD}/multi_op_npu_runner" \
    --progress "${PROGRESS}" \
    --device "${LOGICAL_DEVICE}" \
    --warmup "${WARMUP}" \
    --samples "${SAMPLES}"
