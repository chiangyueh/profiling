#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH="${ROOT}/matmul/mat_mul_v3/op_host/op_tiling/research"
MODE="full"
WORKLOADS="${RESEARCH}/config/workloads.csv"
CALIBRATION_WORKLOADS="${RESEARCH}/config/workloads_calibration.csv"

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

filter_ascend_path() {
    python3 - "$1" <<'PY'
import sys

blocked = (
    "/usr/local/Ascend/ascend-toolkit",
    "/usr/local/Ascend/driver",
    "/usr/local/Ascend/nnrt",
)
items = [item for item in sys.argv[1].split(":") if item]
print(":".join(item for item in items if not any(token in item for token in blocked)))
PY
}

join_existing_dirs() {
    local result=""
    local path
    for path in "$@"; do
        [[ -d "${path}" ]] || continue
        if [[ -n "${result}" ]]; then
            result="${result}:${path}"
        else
            result="${path}"
        fi
    done
    printf '%s' "${result}"
}

find_set_env() {
    local root="$1"
    local candidate
    for candidate in \
        "${root}/set_env.sh" \
        "$(dirname "${root}")/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s' "${candidate}"
            return 0
        fi
    done
    return 1
}

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ ! -d "${CANN_ROOT}" ]]; then
    echo "fatal: CANN root does not exist: ${CANN_ROOT}" >&2
    exit 1
fi

# A shell reused across CANN releases can retain toolkit or driver paths from a
# previous install. Start from non-Ascend paths, then source this install.
export LD_LIBRARY_PATH="$(filter_ascend_path "${LD_LIBRARY_PATH:-}")"
export PYTHONPATH="$(filter_ascend_path "${PYTHONPATH:-}")"
export PATH="$(filter_ascend_path "${PATH:-}")"
unset ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_AICPU_PATH ASCEND_OPP_PATH
unset ASCEND_LATEST_INSTALL_PATH TOOLCHAIN_HOME

SET_ENV_SH="$(find_set_env "${CANN_ROOT}" || true)"
if [[ -n "${SET_ENV_SH}" ]]; then
    set +u
    source "${SET_ENV_SH}"
    set -u
fi

export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export ASCEND_AICPU_PATH="${CANN_ROOT}"
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export ASCEND_LATEST_INSTALL_PATH="$(dirname "$(dirname "${CANN_ROOT}")")"
export PYTHONUNBUFFERED=1

ARCH="$(uname -m)"
PLATFORM_ROOT="${CANN_ROOT}/${ARCH}-linux"
OP_TILING_LIB="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}"

# Real toolkit runtime and driver libraries must resolve before devlib. devlib
# contains compile/link stubs on CANN installations and is only a final fallback.
NON_ASCEND_LD="$(filter_ascend_path "${LD_LIBRARY_PATH:-}")"
RUNTIME_LD="$(join_existing_dirs \
    "${PLATFORM_ROOT}/lib64" \
    "${CANN_ROOT}/lib64" \
    "${CANN_ROOT}/runtime/lib64" \
    "${CANN_ROOT}/fwkacllib/lib64" \
    "${CANN_ROOT}/atc/lib64" \
    "/usr/local/Ascend/driver/lib64" \
    "/usr/local/Ascend/driver/lib64/common" \
    "/usr/local/Ascend/driver/lib64/driver" \
    "${CANN_ROOT}/tools/aml/lib64" \
    "${CANN_ROOT}/tools/aml/lib64/plugin" \
    "${OP_TILING_LIB}")"
DEVLIB_LD="$(join_existing_dirs "${PLATFORM_ROOT}/devlib")"
export LD_LIBRARY_PATH="${RUNTIME_LD}"
[[ -z "${NON_ASCEND_LD}" ]] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NON_ASCEND_LD}"
[[ -z "${DEVLIB_LD}" ]] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${DEVLIB_LD}"

NON_ASCEND_PYTHON="$(filter_ascend_path "${PYTHONPATH:-}")"
CANN_PYTHON="$(join_existing_dirs \
    "${CANN_ROOT}/python/site-packages" \
    "${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe")"
export PYTHONPATH="${RESEARCH}"
[[ -z "${CANN_PYTHON}" ]] || export PYTHONPATH="${PYTHONPATH}:${CANN_PYTHON}"
[[ -z "${NON_ASCEND_PYTHON}" ]] || export PYTHONPATH="${PYTHONPATH}:${NON_ASCEND_PYTHON}"

NON_ASCEND_PATH="$(filter_ascend_path "${PATH:-}")"
CANN_BIN="$(join_existing_dirs \
    "${CANN_ROOT}/bin" \
    "${CANN_ROOT}/compiler/ccec_compiler/bin" \
    "${CANN_ROOT}/tools/ccec_compiler/bin" \
    "${CANN_ROOT}/tools/bishengir/bin")"
export PATH="${CANN_BIN}"
[[ -z "${NON_ASCEND_PATH}" ]] || export PATH="${PATH}:${NON_ASCEND_PATH}"

mkdir -p "${ROOT}/results/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${ROOT}/results/logs/run_npu_${RUN_ID}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

BUILD_DIR="${ROOT}/.build/matmul_v3_tiling_research"
CANDIDATES="${ROOT}/results/npu_full_search_candidates.csv"
ALL_CANDIDATES="${ROOT}/results/npu_full_search_all.csv"
SUMMARY="${ROOT}/results/npu_full_summary.csv"
FAMILY_SUMMARY="${ROOT}/results/npu_full_family_summary.csv"
CANDIDATE_RESULTS="${ROOT}/results/npu_full_candidates.csv"
RESUME="${ROOT}/results/npu_full_resume.csv"
CALIBRATION_CANDIDATES="${ROOT}/results/npu_calibration_search_candidates.csv"
CALIBRATION_ALL="${ROOT}/results/npu_calibration_search_all.csv"
CALIBRATION_SUMMARY="${ROOT}/results/npu_calibration_summary.csv"
CALIBRATION_FAMILY_SUMMARY="${ROOT}/results/npu_calibration_family_summary.csv"
CALIBRATION_RESULTS="${ROOT}/results/npu_calibration_candidates.csv"
CALIBRATION_RESUME="${ROOT}/results/npu_calibration_resume.csv"

echo
echo "NPU run"
echo "  script:     run_npu.sh 20260804-paired-calibration-oneshot-v1"
echo "  upstream:   CANN ops-nn 8.5.0 matmul/mat_mul_v3"
echo "  scope:      paired_bank_relative_calibration_then_unseen_one_shot"
echo "  mode:       ${MODE}"
echo "  workloads:  ${WORKLOADS}"
echo "  summary:    ${SUMMARY}"
echo "  families:   ${FAMILY_SUMMARY}"
echo "  candidates: ${CANDIDATE_RESULTS}"
echo "  resume:     ${RESUME}"
echo "  calibration:${CALIBRATION_RESUME}"
echo "  log:        ${RUN_LOG}"
echo

echo "[1/5] Build callback/bank/NPU tools ..."
cmake -S "${RESEARCH}" -B "${BUILD_DIR}" \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${BUILD_DIR}" --parallel >/dev/null
RUNNER="${BUILD_DIR}/official_matmul_runner"
PROBE="${BUILD_DIR}/tiling_bank_probe"
echo "  ok"

print_acl_loader_diag() {
    echo "  runtime_loader:"
    ldd "${RUNNER}" 2>&1 |
        grep -E 'libascendcl|libruntime|libdrv|libplatform|libnnopbase|libopapi|not found' |
        sed 's/^/    /' || true
}

echo "[2/5] Detect NPU and platform ..."
DETECT_TIMEOUT_SEC="${DETECT_TIMEOUT_SEC:-30}"
PLATFORM_TIMEOUT_SEC="${PLATFORM_TIMEOUT_SEC:-60}"
SOC_PROBE_LOG="${ROOT}/results/logs/soc_probe_${RUN_ID}.log"
PLATFORM_PROBE_LOG="${ROOT}/results/logs/platform_probe_${RUN_ID}.log"

echo "  detect_stage: ACL SoC probe"
set +e
timeout --signal=TERM --kill-after=5 "${DETECT_TIMEOUT_SEC}" \
    "${RUNNER}" --soc-only --device "${DEVICE_ID:-0}" \
    2>&1 | tee "${SOC_PROBE_LOG}"
SOC_PROBE_RC="${PIPESTATUS[0]}"
set -e
ACL_OUTPUT="$(cat "${SOC_PROBE_LOG}")"
if [[ "${SOC_PROBE_RC}" -ne 0 ]]; then
    if [[ "${SOC_PROBE_RC}" -eq 124 ]]; then
        echo "fatal: ACL SoC probe timed out after ${DETECT_TIMEOUT_SEC}s" >&2
    else
        echo "fatal: ACL SoC probe failed rc=${SOC_PROBE_RC}" >&2
    fi
    print_acl_loader_diag >&2
    echo "  probe_log: ${SOC_PROBE_LOG}" >&2
    exit 1
fi
SOC="$(printf '%s\n' "${ACL_OUTPUT}" | sed -n 's/^aclrtGetSocName=//p' | tail -1)"
if [[ -z "${SOC}" || "${SOC}" == "<null>" ]]; then
    echo "${ACL_OUTPUT}"
    echo "fatal: aclrtGetSocName did not return an NPU SoC" >&2
    exit 1
fi
export ASCENDC_SOC_VERSION="${SOC}"
export SOC_VERSION="${SOC}"
echo "  detect_stage: platform capacities"
set +e
timeout --signal=TERM --kill-after=5 "${PLATFORM_TIMEOUT_SEC}" \
    "${PROBE}" --platform "${SOC}" \
    2>&1 | tee "${PLATFORM_PROBE_LOG}"
PLATFORM_PROBE_RC="${PIPESTATUS[0]}"
set -e
PLATFORM="$(sed -n '/^soc=/{p;q;}' "${PLATFORM_PROBE_LOG}")"
if [[ "${PLATFORM_PROBE_RC}" -ne 0 || -z "${PLATFORM}" ]]; then
    if [[ "${PLATFORM_PROBE_RC}" -eq 124 ]]; then
        echo "fatal: platform probe timed out after ${PLATFORM_TIMEOUT_SEC}s" >&2
    else
        echo "fatal: platform probe failed rc=${PLATFORM_PROBE_RC}" >&2
    fi
    echo "  probe_log: ${PLATFORM_PROBE_LOG}" >&2
    exit 1
fi
platform_field() {
    printf '%s\n' "${PLATFORM}" |
        sed -n "s/.*[[:space:]]$1=\\([0-9][0-9.]*\\).*/\\1/p"
}
toolkit_identity() {
    local version_file="${CANN_ROOT}/toolkit/version.info"
    local release=""
    local component=""
    if [[ -f "${version_file}" ]]; then
        release="$(sed -n 's/^version_dir=//p' "${version_file}" | head -1)"
        component="$(sed -n 's/^Version=//p' "${version_file}" | head -1)"
    fi
    if [[ -z "${release}" && -f "${CANN_ROOT}/pyACL/version.info" ]]; then
        release="$(sed -n 's/^Version=//p' "${CANN_ROOT}/pyACL/version.info" | head -1)"
    fi
    if [[ -z "${release}" ]]; then
        release="$(basename "$(readlink -f "${CANN_ROOT}")")"
    fi
    if [[ -n "${component}" && "${component}" != "${release}" ]]; then
        printf '%s+toolkit-%s' "${release}" "${component}"
    else
        printf '%s' "${release}"
    fi
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
TOOLKIT_VERSION="$(toolkit_identity)"
echo "  detected_soc=${SOC} aic=${AIC} runtime_toolkit=${TOOLKIT_VERSION}"
if [[ "${TOOLKIT_VERSION}" != 8.5* ]]; then
    echo "  compatibility: source baseline is 8.5.0; execution uses installed ${TOOLKIT_VERSION}"
    echo "  compatibility: exact callback roundtrip and RuntimeKb preflight remain mandatory"
fi

NPU_CANDIDATES=1
CALIBRATION_NPU_CANDIDATES=16
CALLBACK_CANDIDATES=192
CALIBRATION_CALLBACK_CANDIDATES=128
BEHAVIOR_CANDIDATES=512
if [[ "${MODE}" == "smoke" ]]; then
    CALIBRATION_NPU_CANDIDATES=6
    CALLBACK_CANDIDATES=48
    CALIBRATION_CALLBACK_CANDIDATES=48
    BEHAVIOR_CANDIDATES=128
fi

generate_candidates() {
    local workloads="$1"
    local output="$2"
    local all_output="$3"
    local selection_mode="$4"
    local npu_candidates="$5"
    local callback_candidates="$6"
    local resume_feedback="${7:-}"
    local command=(
        python3 "${RESEARCH}/generate.py"
        --workloads "${workloads}"
        --output "${output}"
        --all-output "${all_output}"
        --source-root "${ROOT}/matmul/mat_mul_v3"
        --soc "${SOC}"
        --toolkit "${TOOLKIT_VERSION}"
        --aic-cores "${AIC}"
        --l0a-bytes "${L0A}"
        --l0b-bytes "${L0B}"
        --l0c-bytes "${L0C}"
        --l1-bytes "${L1}"
        --l2-bytes "${L2}"
        --l2-bpc "${L2_BPC:-1}"
        --hbm-bpc "${HBM_BPC:-1}"
        --observations "${RESEARCH}/config/measured_observations.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log11.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log14.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log15.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log17.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log18.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log19.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log21.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log22.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log23.csv"
        --observations "${RESEARCH}/config/measured_observations_net_log24.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log14.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log15.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log17.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log18.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log19.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log21.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log22.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log23.csv"
        --exclusions "${RESEARCH}/config/measured_fingerprints_net_log24.csv"
        --npu-candidates "${npu_candidates}"
        --callback-candidates "${callback_candidates}"
        --behavior-candidates "${BEHAVIOR_CANDIDATES}"
        --selection-mode "${selection_mode}"
    )
    if [[ -n "${resume_feedback}" ]]; then
        command+=(--resume-feedback "${resume_feedback}")
    fi
    if [[ "${selection_mode}" == "calibration" ]]; then
        command+=(--skip-model-validation)
    fi
    "${command[@]}"
}

profile_stage() {
    local candidates="$1"
    local summary="$2"
    local results="$3"
    local family_summary="$4"
    local resume="$5"
    local pair_block_size="$6"
    python3 "${RESEARCH}/profile.py" \
        --candidates "${candidates}" \
        --summary "${summary}" \
        --family-summary "${family_summary}" \
        --candidate-results "${results}" \
        --resume "${resume}" \
        --runner "${RUNNER}" \
        --probe "${PROBE}" \
        --cann-root "${CANN_ROOT}" \
        --soc "${SOC}" \
        --aic "${AIC}" \
        --toolkit "${TOOLKIT_VERSION}" \
        --timeout "${PROFILE_TIMEOUT_SEC:-120}" \
        --warmup "${WARMUP:-10}" \
        --repeat "${REPEAT:-50}" \
        --samples "${SAMPLES:-15}" \
        --baseline-repeat "${BASELINE_REPEAT:-30}" \
        --baseline-samples "${BASELINE_SAMPLES:-9}" \
        --numeric-preflight-max-mib "${NUMERIC_PREFLIGHT_MAX_MIB:-512}" \
        --baseline-drift-pct "${BASELINE_DRIFT_PCT:-3}" \
        --pair-block-size "${pair_block_size}"
}

echo "[3/5] Calibrate bank-relative effects with controlled candidates ..."
generate_candidates \
    "${CALIBRATION_WORKLOADS}" \
    "${CALIBRATION_CANDIDATES}" \
    "${CALIBRATION_ALL}" \
    calibration \
    "${CALIBRATION_NPU_CANDIDATES}" \
    "${CALIBRATION_CALLBACK_CANDIDATES}"
profile_stage \
    "${CALIBRATION_CANDIDATES}" \
    "${CALIBRATION_SUMMARY}" \
    "${CALIBRATION_RESULTS}" \
    "${CALIBRATION_FAMILY_SUMMARY}" \
    "${CALIBRATION_RESUME}" \
    4
echo "  ok"

echo "[4/5] Generate one-shot decisions for unseen workloads ..."
generate_candidates \
    "${WORKLOADS}" \
    "${CANDIDATES}" \
    "${ALL_CANDIDATES}" \
    one-shot \
    "${NPU_CANDIDATES}" \
    "${CALLBACK_CANDIDATES}" \
    "${CALIBRATION_RESUME}"
echo "  ok"

echo "[5/5] Measure one-shot decision against original MatMulV3 ..."
profile_stage \
    "${CANDIDATES}" \
    "${SUMMARY}" \
    "${CANDIDATE_RESULTS}" \
    "${FAMILY_SUMMARY}" \
    "${RESUME}" \
    1
echo "  ok"

echo
echo "NPU run completed"
echo "  Summary:    ${SUMMARY}"
echo "  Families:   ${FAMILY_SUMMARY}"
echo "  Candidates: ${CANDIDATE_RESULTS}"
echo "  Resume:     ${RESUME}"
echo "  Calibration:${CALIBRATION_RESULTS}"
echo "  log:        ${RUN_LOG}"
