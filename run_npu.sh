#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH="${ROOT}/matmul/mat_mul_v3/op_host/op_tiling/research"
MODE="full"
WORKLOADS="${RESEARCH}/config/workloads.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?missing value for --mode}"
            shift 2
            ;;
        --workloads)
            WORKLOADS="$(realpath "${2:?missing workload CSV}")"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_npu.sh --mode full [--workloads FILE]"
            exit 0
            ;;
        *)
            echo "fatal: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
    echo "fatal: mode must be full or smoke" >&2
    exit 2
fi

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ ! -d "${CANN_ROOT}" ]]; then
    echo "fatal: CANN root does not exist: ${CANN_ROOT}" >&2
    exit 1
fi
if [[ -f "${CANN_ROOT}/set_env.sh" ]]; then
    set +u
    source "${CANN_ROOT}/set_env.sh"
    set -u
fi
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export PYTHONUNBUFFERED=1

ARCH="$(uname -m)"
PLATFORM_ROOT="${CANN_ROOT}/${ARCH}-linux"
OP_TILING_LIB="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}"
export PYTHONPATH="${RESEARCH}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PLATFORM_ROOT}/lib64:${PLATFORM_ROOT}/devlib:${CANN_ROOT}/lib64:${OP_TILING_LIB}:${LD_LIBRARY_PATH:-}"

mkdir -p "${ROOT}/results/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${ROOT}/results/logs/run_npu_${RUN_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

BUILD_DIR="${ROOT}/.build/matmul_v3_tiling_research"
CANDIDATES="${ROOT}/results/npu_full_search_candidates.csv"
ALL_CANDIDATES="${ROOT}/results/npu_full_search_all.csv"
SUMMARY="${ROOT}/results/npu_full_summary.csv"
CANDIDATE_RESULTS="${ROOT}/results/npu_full_candidates.csv"
RESUME="${ROOT}/results/npu_full_resume.csv"

echo
echo "NPU run"
echo "  upstream:   CANN ops-nn 8.5.0 matmul/mat_mul_v3"
echo "  scope:      independent_contract_behavior_search"
echo "  mode:       ${MODE}"
echo "  workloads:  ${WORKLOADS}"
echo "  summary:    ${SUMMARY}"
echo "  candidates: ${CANDIDATE_RESULTS}"
echo "  resume:     ${RESUME}"
echo "  log:        ${RUN_LOG}"
echo

echo "[1/4] Build callback/bank/NPU tools ..."
cmake -S "${RESEARCH}" -B "${BUILD_DIR}" \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${BUILD_DIR}" --parallel >/dev/null
RUNNER="${BUILD_DIR}/official_matmul_runner"
PROBE="${BUILD_DIR}/tiling_bank_probe"
echo "  ok"

echo "[2/4] Detect NPU and platform ..."
ACL_OUTPUT="$("${RUNNER}" --acl-only --device "${DEVICE_ID:-0}")"
SOC="$(printf '%s\n' "${ACL_OUTPUT}" | sed -n 's/^aclrtGetSocName=//p' | tail -1)"
if [[ -z "${SOC}" || "${SOC}" == "<null>" ]]; then
    echo "${ACL_OUTPUT}"
    echo "fatal: aclrtGetSocName did not return an NPU SoC" >&2
    exit 1
fi
export ASCENDC_SOC_VERSION="${SOC}"
export SOC_VERSION="${SOC}"
PLATFORM="$("${PROBE}" --platform "${SOC}")"
echo "  ${PLATFORM}"
platform_field() {
    printf '%s\n' "${PLATFORM}" |
        sed -n "s/.*[[:space:]]$1=\\([0-9][0-9.]*\\).*/\\1/p"
}
AIC="$(platform_field aic)"
L0A="$(platform_field L0A)"
L0B="$(platform_field L0B)"
L0C="$(platform_field L0C)"
L1="$(platform_field L1)"
L2="$(platform_field L2)"
L2_BPC="$(platform_field L2_Bpc_per_core)"
HBM_BPC="$(platform_field HBM_Bpc_per_core)"
for value in "${AIC}" "${L0A}" "${L0B}" "${L0C}" "${L1}" "${L2}"; do
    if [[ -z "${value}" ]]; then
        echo "fatal: incomplete platform capability record: ${PLATFORM}" >&2
        exit 1
    fi
done
TOOLKIT_VERSION="$(basename "$(readlink -f "${CANN_ROOT}")")"
echo "  detected_soc=${SOC} aic=${AIC} runtime_toolkit=${TOOLKIT_VERSION}"
if [[ "${TOOLKIT_VERSION}" != 8.5* ]]; then
    echo "  compatibility: source baseline is 8.5.0; execution uses installed ${TOOLKIT_VERSION}"
    echo "  compatibility: exact callback roundtrip and RuntimeKb preflight remain mandatory"
fi

NPU_CANDIDATES="${NPU_CANDIDATES:-40}"
if [[ "${MODE}" == "smoke" ]]; then
    NPU_CANDIDATES=4
fi

echo "[3/4] Generate independent hardware-contract candidates ..."
python3 "${RESEARCH}/generate.py" \
    --workloads "${WORKLOADS}" \
    --output "${CANDIDATES}" \
    --all-output "${ALL_CANDIDATES}" \
    --source-root "${ROOT}/matmul/mat_mul_v3" \
    --soc "${SOC}" \
    --aic-cores "${AIC}" \
    --l0a-bytes "${L0A}" \
    --l0b-bytes "${L0B}" \
    --l0c-bytes "${L0C}" \
    --l1-bytes "${L1}" \
    --l2-bytes "${L2}" \
    --l2-bpc "${L2_BPC:-1}" \
    --hbm-bpc "${HBM_BPC:-1}" \
    --observations "${RESEARCH}/config/measured_observations.csv" \
    --exclusions "${RESEARCH}/config/measured_fingerprints.csv" \
    --resume-feedback "${RESUME}" \
    --npu-candidates "${NPU_CANDIDATES}" \
    --callback-candidates 48 \
    --behavior-candidates 320
echo "  ok"

echo "[4/4] Run paired official/bank/candidate NPU profiling ..."
python3 "${RESEARCH}/profile.py" \
    --candidates "${CANDIDATES}" \
    --summary "${SUMMARY}" \
    --candidate-results "${CANDIDATE_RESULTS}" \
    --resume "${RESUME}" \
    --runner "${RUNNER}" \
    --probe "${PROBE}" \
    --cann-root "${CANN_ROOT}" \
    --soc "${SOC}" \
    --aic "${AIC}" \
    --toolkit "${TOOLKIT_VERSION}" \
    --timeout "${PROFILE_TIMEOUT_SEC:-120}" \
    --warmup "${WARMUP:-10}" \
    --repeat "${REPEAT:-50}" \
    --samples "${SAMPLES:-15}"
echo "  ok"

echo
echo "NPU run completed"
echo "  Summary:    ${SUMMARY}"
echo "  Candidates: ${CANDIDATE_RESULTS}"
echo "  Resume:     ${RESUME}"
echo "  log:        ${RUN_LOG}"
