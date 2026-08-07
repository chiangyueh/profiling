#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH="${ROOT}/matmul/mat_mul_v3/op_host/op_tiling/research"
MODE="full"
WORKLOADS=""
WORKLOADS_SUPPLIED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?missing value for --mode}"
            shift 2
            ;;
        --workloads)
            WORKLOADS="$(realpath "${2:?missing workload CSV}")"
            WORKLOADS_SUPPLIED=1
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
STATE_ROOT="${ROOT}/.benchmark_state/hardware_coverage_v2"
PROBE_STATE="${STATE_ROOT}/probe"
mkdir -p "${PROBE_STATE}"

echo
echo "NPU run"
echo "  script:     run_npu.sh 20260807-audited-hardware-coverage-v24"
echo "  upstream:   CANN ops-nn 8.5.0 matmul/mat_mul_v3"
echo "  scope:      broad_contract_search_with_same_run_feedback"
echo "  mode:       ${MODE}"
echo "  domain:     rank2_ND_no_bias fp16,bf16,fp32 NN,NT,TN,TT"
echo "  measurement:full_numeric_preflight + paired_ACL_event_latency"
echo "  handoff:    one_log_only"
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
SOC_PROBE_LOG="${PROBE_STATE}/soc_probe.log"
PLATFORM_PROBE_LOG="${PROBE_STATE}/platform_probe.log"

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
UB="$(platform_field UB)"
L2="$(platform_field L2)"
L2_BPC="$(platform_field L2_Bpc_per_core)"
HBM_BPC="$(platform_field HBM_Bpc_per_core)"
for value in "${AIC}" "${L0A}" "${L0B}" "${L0C}" "${L1}" "${UB}" "${L2}"; do
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

CAMPAIGN_ID_SOURCE="hardware_coverage_v2|${SOC}|${AIC}|${TOOLKIT_VERSION}|${MODE}"
if [[ "${WORKLOADS_SUPPLIED}" -eq 1 ]]; then
    CAMPAIGN_ID_SOURCE="${CAMPAIGN_ID_SOURCE}|$(sha256sum "${WORKLOADS}" | cut -d' ' -f1)"
fi
CAMPAIGN_ID="$(printf '%s' "${CAMPAIGN_ID_SOURCE}" | sha256sum | cut -c1-16)"
CAMPAIGN_STATE="${STATE_ROOT}/${CAMPAIGN_ID}"
mkdir -p "${CAMPAIGN_STATE}"
MANIFEST="${CAMPAIGN_STATE}/workloads.csv"
RESUME="${CAMPAIGN_STATE}/measurements.csv"
STAGE1_CANDIDATES="${CAMPAIGN_STATE}/stage1_candidates.csv"
STAGE1_ALL="${CAMPAIGN_STATE}/stage1_all.csv"
STAGE2_CANDIDATES="${CAMPAIGN_STATE}/stage2_candidates.csv"
STAGE2_ALL="${CAMPAIGN_STATE}/stage2_all.csv"

if [[ "${WORKLOADS_SUPPLIED}" -eq 0 ]]; then
    WORKLOADS="${MANIFEST}"
    python3 "${RESEARCH}/benchmark_manifest.py" \
        --output "${MANIFEST}" \
        --aic-cores "${AIC}" \
        --mode "${MODE}"
else
    echo "BENCHMARK_MANIFEST custom_workloads=${WORKLOADS}"
fi

STAGE1_NPU=64
STAGE1_CALLBACK=192
STAGE1_BEHAVIOR=640
STAGE2_NPU=32
STAGE2_CALLBACK=160
STAGE2_BEHAVIOR=512
if [[ "${MODE}" == "smoke" ]]; then
    STAGE1_NPU=8
    STAGE1_CALLBACK=32
    STAGE1_BEHAVIOR=64
    STAGE2_NPU=4
    STAGE2_CALLBACK=24
    STAGE2_BEHAVIOR=48
fi

append_historical_feedback() {
    local -n command_ref="$1"
    local path
    for path in "${RESEARCH}"/config/measured_observations*.csv; do
        [[ -f "${path}" ]] && command_ref+=(--observations "${path}")
    done
    for path in "${RESEARCH}"/config/measured_fingerprints*.csv; do
        [[ -f "${path}" ]] && command_ref+=(--exclusions "${path}")
    done
    for path in "${RESEARCH}"/config/paired_measurements*.csv; do
        [[ -f "${path}" ]] && command_ref+=(--resume-feedback "${path}")
    done
}

generate_stage() {
    local stage="$1"
    local output="$2"
    local all_output="$3"
    local npu_candidates="$4"
    local callback_candidates="$5"
    local behavior_candidates="$6"
    local command=(
        python3 "${RESEARCH}/generate.py"
        --workloads "${WORKLOADS}"
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
        --ub-bytes "${UB}"
        --l2-bytes "${L2}"
        --l2-bpc "${L2_BPC:-1}"
        --hbm-bpc "${HBM_BPC:-1}"
        --npu-candidates "${npu_candidates}"
        --callback-candidates "${callback_candidates}"
        --behavior-candidates "${behavior_candidates}"
        --selection-mode campaign
        --search-stage "${stage}"
        --fixed-campaign-budget
        --skip-model-validation
    )
    append_historical_feedback command
    if [[ "${stage}" == "stage2" ]]; then
        command+=(--resume-feedback "${RESUME}")
    fi
    "${command[@]}"
}

profile_stage() {
    local stage="$1"
    local candidates="$2"
    local repeat="$3"
    local samples="$4"
    local warmup="$5"
    local pair_block="$6"
    python3 "${RESEARCH}/profile.py" \
        --campaign-stage "${stage}" \
        --run-id "${RUN_ID}" \
        --candidates "${candidates}" \
        --summary "${CAMPAIGN_STATE}/${stage}_summary.csv" \
        --family-summary "${CAMPAIGN_STATE}/${stage}_families.csv" \
        --candidate-results "${CAMPAIGN_STATE}/${stage}_results.csv" \
        --resume "${RESUME}" \
        --runner "${RUNNER}" \
        --probe "${PROBE}" \
        --cann-root "${CANN_ROOT}" \
        --soc "${SOC}" \
        --aic "${AIC}" \
        --toolkit "${TOOLKIT_VERSION}" \
        --timeout "${PROFILE_TIMEOUT_SEC:-120}" \
        --warmup "${warmup}" \
        --repeat "${repeat}" \
        --samples "${samples}" \
        --baseline-repeat "${repeat}" \
        --baseline-samples "${samples}" \
        --numeric-preflight-max-mib "${NUMERIC_PREFLIGHT_MAX_MIB:-512}" \
        --baseline-drift-pct "${BASELINE_DRIFT_PCT:-3}" \
        --pair-attempts "${PAIR_ATTEMPTS:-3}" \
        --pair-block-size "${pair_block}"
}

stage_status() {
    local stage="$1"
    local candidates="$2"
    local expected_per_workload="$3"
    shift 3
    python3 "${RESEARCH}/benchmark_status.py" \
        --stage "${stage}" \
        --candidates "${candidates}" \
        --workloads "${WORKLOADS}" \
        --expected-per-workload "${expected_per_workload}" \
        --resume "${RESUME}" \
        --soc "${SOC}" \
        --aic "${AIC}" \
        --toolkit "${TOOLKIT_VERSION}" \
        "$@"
}

telemetry_snapshot() {
    local point="$1"
    echo "BENCHMARK_TELEMETRY_BEGIN point=${point} scope=stage_boundary"
    if command -v npu-smi >/dev/null 2>&1; then
        timeout 15 npu-smi info || true
    else
        echo "npu_smi=unavailable"
    fi
    echo "BENCHMARK_TELEMETRY_END point=${point}"
}

echo "[3/5] Generate fixed broad coverage frontier ..."
if [[ ! -s "${STAGE1_CANDIDATES}" ]]; then
    STAGE_WALL_START="$(date +%s%N)"
    generate_stage \
        stage1 "${STAGE1_CANDIDATES}" "${STAGE1_ALL}" \
        "${STAGE1_NPU}" "${STAGE1_CALLBACK}" "${STAGE1_BEHAVIOR}"
    STAGE_WALL_END="$(date +%s%N)"
    echo "BENCHMARK_HOST_STAGE stage=stage1 wall_ms=$(( (STAGE_WALL_END - STAGE_WALL_START) / 1000000 ))"
else
    echo "BENCHMARK_FRONTIER_REUSE stage=stage1 path=hidden_state"
fi
stage_status stage1 "${STAGE1_CANDIDATES}" "${STAGE1_NPU}" --frontier-only
stage_status stage1 "${STAGE1_CANDIDATES}" \
    "${STAGE1_NPU}" --emit-records \
    --exclude-run-id "${RUN_ID}" || true

echo "[4/5] Measure broad coverage frontier on NPU ..."
telemetry_snapshot stage1_pre
STAGE_WALL_START="$(date +%s%N)"
profile_stage stage1 "${STAGE1_CANDIDATES}" \
    "${STAGE1_REPEAT:-20}" "${STAGE1_SAMPLES:-7}" \
    "${STAGE1_WARMUP:-5}" "${STAGE1_PAIR_BLOCK:-8}"
STAGE_WALL_END="$(date +%s%N)"
echo "BENCHMARK_NPU_STAGE stage=stage1 wall_ms=$(( (STAGE_WALL_END - STAGE_WALL_START) / 1000000 ))"
telemetry_snapshot stage1_post
set +e
stage_status stage1 "${STAGE1_CANDIDATES}" "${STAGE1_NPU}"
STAGE1_STATUS="$?"
set -e
if [[ "${STAGE1_STATUS}" -ne 0 ]]; then
    echo
    echo "NPU run incomplete"
    echo "  stage: stage1"
    echo "  action: rerun the same command; completed fingerprints are preserved"
    echo "  analysis_log: ${RUN_LOG}"
    exit 3
fi

echo "[5/5] Generate and measure feedback-expanded frontier ..."
if [[ ! -s "${STAGE2_CANDIDATES}" ]]; then
    STAGE_WALL_START="$(date +%s%N)"
    generate_stage \
        stage2 "${STAGE2_CANDIDATES}" "${STAGE2_ALL}" \
        "${STAGE2_NPU}" "${STAGE2_CALLBACK}" "${STAGE2_BEHAVIOR}"
    STAGE_WALL_END="$(date +%s%N)"
    echo "BENCHMARK_HOST_STAGE stage=stage2 wall_ms=$(( (STAGE_WALL_END - STAGE_WALL_START) / 1000000 ))"
else
    echo "BENCHMARK_FRONTIER_REUSE stage=stage2 path=hidden_state"
fi
stage_status stage2 "${STAGE2_CANDIDATES}" \
    "${STAGE2_NPU}" --frontier-only \
    --prior-candidates "${STAGE1_CANDIDATES}"
telemetry_snapshot stage2_pre
STAGE_WALL_START="$(date +%s%N)"
profile_stage stage2 "${STAGE2_CANDIDATES}" \
    "${STAGE2_REPEAT:-50}" "${STAGE2_SAMPLES:-15}" \
    "${STAGE2_WARMUP:-10}" "${STAGE2_PAIR_BLOCK:-4}"
STAGE_WALL_END="$(date +%s%N)"
echo "BENCHMARK_NPU_STAGE stage=stage2 wall_ms=$(( (STAGE_WALL_END - STAGE_WALL_START) / 1000000 ))"
telemetry_snapshot stage2_post
set +e
stage_status stage2 "${STAGE2_CANDIDATES}" \
    "${STAGE2_NPU}" --prior-candidates "${STAGE1_CANDIDATES}"
STAGE2_STATUS="$?"
set -e

echo
if [[ "${STAGE2_STATUS}" -eq 0 ]]; then
    echo "BENCHMARK_COMPLETION status=ACQUISITION_COMPLETE stages=2"
    echo "NPU run completed"
else
    echo "BENCHMARK_COMPLETION status=INCOMPLETE stage=stage2"
    echo "NPU run incomplete; rerun the same command"
fi
echo "  analysis_log: ${RUN_LOG}"
exit "${STAGE2_STATUS}"
