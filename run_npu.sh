#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NPU_VERSION="20260822-unseen5-hardware-cost-v1"
mkdir -p "${ROOT}/results/logs"
RUN_LOG="${ROOT}/results/logs/run_npu_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${RUN_LOG}") 2>&1
RUN_VERBOSE="${RUN_VERBOSE:-0}"
LAST_PHASE_LOG=""
ERROR_ALREADY_REPORTED=0
SEARCH_WORK_DIR=""
ERROR_PATTERN='\[ERROR\]|RegisterAscendBinary|LaunchAscendKernel ret|fatal:|terminate|throwing|Aborted|(^|[^a-z])error([^a-z]|$)|failed|failure|aclrtBinaryLoadFromFile|rc=[0-9]+|exit_code|symbol lookup error|undefined symbol|not found|missing|No successful|invalid|cannot|Traceback|Exception'

if [[ "${RUN_DEBUG:-0}" == "1" ]]; then
    export PS4='+ ${BASH_SOURCE}:${LINENO}: '
    set -x
fi

print_error_lines() {
    local log_file="$1"
    if [[ ! -s "${log_file}" ]]; then
        echo "  phase_log: ${log_file} (empty or missing)"
        return
    fi
    local matches
    matches="$(grep -Eai "${ERROR_PATTERN}" "${log_file}" | tail -40 || true)"
    if [[ -n "${matches}" ]]; then
        echo "${matches}"
    else
        echo "  no matched error line; last 40 phase-log lines:"
        tail -40 "${log_file}"
    fi
    if grep -Eaiq '(^|[^0-9])507008([^0-9]|$)' "${log_file}"; then
        echo "  hint: 507008 = ACL_ERROR_RT_SOC_VERSION (SoC version error)."
        echo "        Check CANN/runtime path and ASCENDC_SOC_VERSION for this card."
        echo "        Run ./scripts/check_server.sh to compare minimal ACL vs linked runner ACL."
    fi
    if grep -Eaiq '(^|[^0-9])507015([^0-9]|$)' "${log_file}"; then
        echo "  hint: 507015 = ACL_ERROR_RT_AICORE_EXCEPTION."
        echo "        Full profiling stops because later candidate results are not trustworthy."
    fi
    echo "  phase_log: ${log_file}"
}

print_diag_summary() {
    local diag_file="$1"
    if [[ ! -s "${diag_file}" ]]; then
        echo "  runtime_diag: ${diag_file} (empty or missing)"
        return
    fi
    echo "  runtime_diag: ${diag_file}"
    grep -Eai '^(ASCEND_MATMUL_STRICT_ENV|CANN_ROOT|CANN_PLATFORM_ROOT|ASCEND_HOME_PATH|ASCEND_TOOLKIT_HOME|ASCEND_OPP_PATH|ASCEND_LATEST_INSTALL_PATH|ASCENDC_SOC_VERSION|DETECTED_|arch=|device_id=|exists:|missing:|official_matmul_runner|tiling_bank_probe|.*libascendcl|.*libnnopbase|.*libopapi|.*liboptiling|.*not found|npu-smi_rc=|symbol lookup error|undefined symbol)' "${diag_file}" \
        | head -80 || true
}

run_server_check_summary() {
    local server_log="${ROOT}/results/logs/server_check_auto_$(date +%Y%m%d_%H%M%S).log"
    if [[ ! -x "${ROOT}/scripts/check_server.sh" ]]; then
        echo "  server_check: ${ROOT}/scripts/check_server.sh missing or not executable"
        return 0
    fi
    echo "  server_check: ${server_log}"
    "${ROOT}/scripts/check_server.sh" >"${server_log}" 2>&1 || true
    grep -Eai '^(server_check_summary|minimal_acl=|runner_acl=|classification=|next_step=|CANN_ROOT=|ASCEND_TOOLKIT_HOME=|ASCENDC_SOC_VERSION=|DETECTED_NPU_SOC=|\\+ .*acl_min|\\+ .*official_matmul_runner.*--acl-only|minimal acl check|official runner ACL check|aclInit rc=|aclrtGetSocName=|fatal:|error|failed)' "${server_log}" \
        | tail -120 || true
}

filter_profile_terminal() {
    awk '
        BEGIN { in_error = 0 }
        /^TILING_ERROR_BEGIN$/ { in_error = 1; print; next }
        /^TILING_ERROR_END$/ { print; in_error = 0; next }
        in_error { print; next }
        /^fatal:/ || /^Traceback/ || /^ValueError:/ || /^ProfileError:/ ||
        /^\[ERROR\]/ || /^candidate_abort / || /^candidate_rejected / ||
        /^OFFICIAL_BASELINE_ERROR_/ {
            print
            next
        }
        /^bank_schema:/ || /^bank_records_prepared:/ || /^profile_plan:/ ||
        /^resume_history:/ || /^workload_skip:/ ||
        /^official_tiling_profile completed/ ||
        /^profile_npu completed/ {
            print
            next
        }
        /^WORKLOAD / && ENVIRON["PROFILE_SHOW_WORKLOADS"] == "1" {
            print
            next
        }
        /^bank_control_done / && ENVIRON["PROFILE_SHOW_WORKLOADS"] == "1" {
            print
            next
        }
        /^WORKLOAD_RESULT / &&
        (ENVIRON["PROFILE_SHOW_WORKLOADS"] == "1" ||
         $0 ~ /optimization_result=improved/ ||
         $0 ~ /status_vs_official=regressed/ ||
         $0 ~ /status_vs_bank=regressed/) {
            print
            next
        }
        /^candidate_done / &&
        ($0 ~ /status_vs_official=improved/ || $0 ~ /status_vs_bank=improved/) {
            print
            next
        }
        /^candidate_history_reuse / && ENVIRON["PROFILE_SHOW_REUSE"] == "1" {
            print
            next
        }
        /^official_history_reuse / && ENVIRON["PROFILE_SHOW_REUSE"] == "1" {
            print
            next
        }
        /^bank_control_history_reuse / && ENVIRON["PROFILE_SHOW_REUSE"] == "1" {
            print
            next
        }
    '
}

run_quiet_phase() {
    local label="$1"
    local log_file="$2"
    shift 2
    LAST_PHASE_LOG="${log_file}"
    ERROR_ALREADY_REPORTED=0
    echo -n "${label} ... "
    if [[ "${RUN_VERBOSE}" == "1" ]]; then
        local rc
        if "$@" 2>&1 | tee "${log_file}"; then
            rc=0
        else
            rc=${PIPESTATUS[0]}
        fi
    else
        local rc
        if "$@" >"${log_file}" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi
    if [[ "${rc}" -eq 0 ]]; then
        echo "ok"
        return 0
    fi
    echo "failed"
    print_error_lines "${log_file}"
    if grep -Eaiq '(^|[^0-9])507008([^0-9]|$)' "${log_file}"; then
        local diag_log="${ROOT}/results/logs/runtime_diag_$(date +%Y%m%d_%H%M%S).log"
        "${ROOT}/scripts/diagnose_runtime.sh" >"${diag_log}" 2>&1 || true
        print_diag_summary "${diag_log}"
        run_server_check_summary
    fi
    ERROR_ALREADY_REPORTED=1
    return "${rc}"
}

on_error() {
    local rc=$?
    echo
    echo "NPU run failed"
    echo "  exit_code: ${rc}"
    echo "  log:       ${RUN_LOG}"
    if [[ "${ERROR_ALREADY_REPORTED}" != "1" && -n "${LAST_PHASE_LOG}" ]]; then
        print_error_lines "${LAST_PHASE_LOG}"
    fi
    exit "${rc}"
}

cleanup_search_work() {
    if [[ -n "${SEARCH_WORK_DIR}" && -d "${SEARCH_WORK_DIR}" ]]; then
        find "${SEARCH_WORK_DIR}" -mindepth 1 -delete
        rmdir "${SEARCH_WORK_DIR}" 2>/dev/null || true
    fi
}

trap on_error ERR
trap cleanup_search_work EXIT

cd "${ROOT}"

usage() {
    cat <<'USAGE'
Usage:
  ./run_npu.sh --check-server [--device PHYSICAL_NPU_ID] [--verbose]
  ./run_npu.sh --mode smoke [--device PHYSICAL_NPU_ID] [--workloads FILE] [--output-stem STEM]
  ./run_npu.sh --mode full  [--device PHYSICAL_NPU_ID]

Modes:
  smoke  Quick NPU validation: official baseline, bank control, and one
         bottleneck-transition candidate.
  full   Five fixed unseen shapes: original installed MatMulV3 versus one
         hardware-cost-solver tiling per shape. No legacy history is loaded.

Environment overrides:
  CANN_ROOT optionally selects a toolkit root; official set_env.sh is preferred
  PHYSICAL_NPU_ID defaults to 1 and selects the physical NPU; the application
  always uses logical DEVICE_ID=0 after ASCEND_RT_VISIBLE_DEVICES mapping
  SOC_VERSION
  The following search and profiling overrides apply to smoke mode only:
  BEAM_WIDTH, TABU_ITERS, LNS_ROUNDS, TOP_K, MAX_CORE_ROUNDS, MODEL_RATIO_LIMIT,
  SEARCH_SCOPE, RANK_LIMIT, WARMUP, REPEAT, SAMPLES,
  NUMERIC_PREFLIGHT_MAX_MIB, PROFILE_STALL_TIMEOUT_SEC, PROFILE_PROGRESS_EVERY
  PROFILE_SHOW_WORKLOADS=1 prints every workload result during live profiling
  PROFILE_SHOW_REUSE=1 prints reused-history lines in the compact live output
  KEEP_DETAILS=1 preserves search/profile internals in results/*_details/
  PRINT_ALL_RESULTS=1 prints every workload in the final result block
  PRINT_RESULTS=0 suppresses the compact terminal result block
  FORCE_ASCENDC_SOC_VERSION=1 permits an explicit exact-SoC override
  RUN_VERBOSE=1 or --verbose prints full build/search/profile/check output
  --check-server runs CANN/driver/minimal-ACL/official-runner diagnostics only

This script never falls back to CPU smoke.
USAGE
}

MODE="${MODE:-smoke}"
WORKLOADS_CSV=""
RESULT_STEM=""
CHECK_SERVER=0
PHYSICAL_NPU_ID="${PHYSICAL_NPU_ID:-1}"
PHASE_COUNT=4
WORKLOADS_WERE_OVERRIDDEN=0
DISABLE_LEGACY_HISTORY=0
SKIP_BANK_SEED_CONTROL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-server)
            CHECK_SERVER=1
            shift
            ;;
        --verbose)
            RUN_VERBOSE=1
            shift
            ;;
        -d|--device)
            PHYSICAL_NPU_ID="${2:?missing physical NPU ID for --device}"
            shift 2
            ;;
        --mode)
            MODE="${2:?missing value for --mode}"
            shift 2
            ;;
        --workloads)
            WORKLOADS_CSV="${2:?missing value for --workloads}"
            WORKLOADS_WERE_OVERRIDDEN=1
            shift 2
            ;;
        --output-stem)
            RESULT_STEM="${2:?missing value for --output-stem}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! "${PHYSICAL_NPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "Invalid physical NPU ID: ${PHYSICAL_NPU_ID}" >&2
    exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES="${PHYSICAL_NPU_ID}"
export DEVICE_ID=0
echo "NPU mapping: physical ${PHYSICAL_NPU_ID} -> logical DEVICE_ID=0"

if [[ "${CHECK_SERVER}" == "1" ]]; then
    mkdir -p results results/logs
    echo "Server check"
    echo "  script: run_npu.sh ${RUN_NPU_VERSION}"
    echo "  root:   ${ROOT}"
    echo
    SERVER_CHECK_VERBOSE="${RUN_VERBOSE:-0}" "${ROOT}/scripts/check_server.sh"
    exit $?
fi

case "${MODE}" in
    smoke)
        WORKLOADS_CSV="${WORKLOADS_CSV:-${WORKLOADS:-config/workloads_smoke.csv}}"
        RESULT_STEM="${RESULT_STEM:-results/npu_smoke}"
        DEFAULT_BEAM_WIDTH=16
        DEFAULT_TABU_ITERS=8
        DEFAULT_LNS_ROUNDS=2
        DEFAULT_TOP_K=2
        DEFAULT_MAX_CORE_ROUNDS=0
        DEFAULT_MODEL_RATIO_LIMIT=1.03
        DEFAULT_RANK_LIMIT=1
        DEFAULT_WARMUP=2
        DEFAULT_REPEAT=5
        DEFAULT_SAMPLES=3
        DEFAULT_PROFILE_STALL_TIMEOUT_SEC=60
        DEFAULT_NUMERIC_PREFLIGHT_MAX_MIB=4
        DEFAULT_PROFILE_PROGRESS_EVERY=1
        DEFAULT_SEARCH_SCOPE=bottleneck_guided_v1
        DEFAULT_REQUIRE_EXACT_RESUME_PREFIX=0
        ;;
    full)
        if [[ "${WORKLOADS_WERE_OVERRIDDEN}" == "1" || -n "${WORKLOADS:-}" ]]; then
            echo "--mode full has a fixed five-shape unseen contract; --workloads is not allowed" >&2
            exit 1
        fi
        WORKLOADS_CSV="config/workloads_unseen_5.csv"
        RESULT_STEM="results/unseen5_hardware_solver_v1"
        DEFAULT_BEAM_WIDTH=16
        DEFAULT_TABU_ITERS=0
        DEFAULT_LNS_ROUNDS=0
        DEFAULT_TOP_K=1
        DEFAULT_MAX_CORE_ROUNDS=0
        DEFAULT_MODEL_RATIO_LIMIT=1.03
        DEFAULT_RANK_LIMIT=1
        DEFAULT_WARMUP=2
        DEFAULT_REPEAT=20
        DEFAULT_SAMPLES=7
        DEFAULT_PROFILE_STALL_TIMEOUT_SEC=60
        DEFAULT_NUMERIC_PREFLIGHT_MAX_MIB=4
        DEFAULT_PROFILE_PROGRESS_EVERY=1
        DEFAULT_SEARCH_SCOPE=hardware_breakpoints_v1
        DEFAULT_REQUIRE_EXACT_RESUME_PREFIX=0
        DISABLE_LEGACY_HISTORY=1
        SKIP_BANK_SEED_CONTROL=1
        # The five-shape comparison is a fixed contract: one solver tiling
        # and the same measurement budget for every shape.
        BEAM_WIDTH=16
        TABU_ITERS=0
        LNS_ROUNDS=0
        TOP_K=1
        MAX_CORE_ROUNDS=0
        MODEL_RATIO_LIMIT=1.03
        RANK_LIMIT=1
        WARMUP=2
        REPEAT=20
        SAMPLES=7
        PROFILE_STALL_TIMEOUT_SEC=60
        NUMERIC_PREFLIGHT_MAX_MIB=4
        PROFILE_PROGRESS_EVERY=1
        SEARCH_SCOPE=hardware_breakpoints_v1
        ;;
    *)
        echo "Invalid --mode: ${MODE}. Expected smoke or full." >&2
        usage >&2
        exit 1
        ;;
esac

if [[ ! -f "${WORKLOADS_CSV}" ]]; then
    echo "Workload CSV not found: ${WORKLOADS_CSV}" >&2
    exit 1
fi

mkdir -p results results/logs

echo "NPU run"
echo "  script:    run_npu.sh ${RUN_NPU_VERSION}"
echo "  root:      ${ROOT}"
echo "  mode:      ${MODE}"
echo "  workloads: ${WORKLOADS_CSV}"
if [[ "${MODE}" == "full" ]]; then
    echo "  comparison: one installed MatMulV3 baseline + one rank-1 solver tiling per shape"
    echo "  solver:     hardware capacity/traffic/cycle model + official RuntimeKb callback"
    echo "  history:    legacy measurement history disabled; local exact resume only"
    echo "  sampling:   2 warmups + 7 event samples, 20 launches/sample"
else
    echo "  scope:     ${SEARCH_SCOPE:-${DEFAULT_SEARCH_SCOPE}}"
    echo "  summary:   ${RESULT_STEM}_summary.csv"
    echo "  candidates: ${RESULT_STEM}_candidates.csv"
    echo "  resume:    ${RESULT_STEM}_resume.csv"
fi
echo "  log:       ${RUN_LOG}"
echo

ENV_LOG="${ROOT}/results/logs/env_$(date +%Y%m%d_%H%M%S).log"
LAST_PHASE_LOG="${ENV_LOG}"
ERROR_ALREADY_REPORTED=0
echo -n "[0/${PHASE_COUNT}] Setup CANN environment ... "
if {
    echo "phase0=no_acl_probe"
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/env.sh"
    echo "ASCEND_MATMUL_STRICT_ENV=${ASCEND_MATMUL_STRICT_ENV:-1}"
    echo "CANN_ROOT=${CANN_ROOT}"
    echo "CANN_PLATFORM_ROOT=${CANN_PLATFORM_ROOT}"
    echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-}"
    echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-}"
    echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
    echo "ASCEND_LATEST_INSTALL_PATH=${ASCEND_LATEST_INSTALL_PATH:-}"
    echo "ASCENDC_SOC_VERSION=${ASCENDC_SOC_VERSION:-}"
} >"${ENV_LOG}" 2>&1; then
    env_rc=0
else
    env_rc=$?
fi
if [[ "${env_rc}" -ne 0 ]]; then
    echo "failed"
    print_error_lines "${ENV_LOG}"
    ERROR_ALREADY_REPORTED=1
    exit "${env_rc}"
fi
echo "ok"
echo "  cann_root:    ${CANN_ROOT}"

SOC_ENV_FILE="${ROOT}/results/logs/detected_soc_$(date +%Y%m%d_%H%M%S).env"
SOC_DETECT_LOG="${ROOT}/results/logs/detect_soc_$(date +%Y%m%d_%H%M%S).log"
run_quiet_phase "[1/${PHASE_COUNT}] Detect NPU SoC" "${SOC_DETECT_LOG}" \
    "${ROOT}/scripts/detect_soc.sh" "${SOC_ENV_FILE}"

# shellcheck disable=SC1090
source "${SOC_ENV_FILE}"
export ASCENDC_SOC_VERSION SOC_VERSION DETECTED_NPU_SOC DETECTED_NPU_SOC_SOURCE DETECTED_NPU_SOC_RAW
echo "  detected_soc: ${DETECTED_NPU_SOC} (${DETECTED_NPU_SOC_SOURCE})"
echo "  ascendc_soc:  ${ASCENDC_SOC_VERSION}"

run_quiet_phase "[2/${PHASE_COUNT}] Build tiling host/official runner" "${ROOT}/results/logs/build_$(date +%Y%m%d_%H%M%S).log" \
    "${ROOT}/scripts/build_all.sh"

SEARCH_WORK_DIR="$(mktemp -d "${ROOT}/results/.search_${MODE}.XXXXXX")"
SEARCH_CANDIDATES_CSV="${SEARCH_WORK_DIR}/candidates.csv"
SEARCH_ALL_CSV="${SEARCH_WORK_DIR}/all_evaluated.csv"
SEARCH_TILING_DIR="${SEARCH_WORK_DIR}/tilings"
SEARCH_LOG="${ROOT}/results/logs/search_$(date +%Y%m%d_%H%M%S).log"
run_quiet_phase "[3/4] Search tiling candidates" "${SEARCH_LOG}" \
    env BEAM_WIDTH="${BEAM_WIDTH:-${DEFAULT_BEAM_WIDTH}}" \
        TABU_ITERS="${TABU_ITERS:-${DEFAULT_TABU_ITERS}}" \
        LNS_ROUNDS="${LNS_ROUNDS:-${DEFAULT_LNS_ROUNDS}}" \
        TOP_K="${TOP_K:-${DEFAULT_TOP_K}}" \
        MAX_CORE_ROUNDS="${MAX_CORE_ROUNDS:-${DEFAULT_MAX_CORE_ROUNDS}}" \
        MODEL_RATIO_LIMIT="${MODEL_RATIO_LIMIT:-${DEFAULT_MODEL_RATIO_LIMIT}}" \
        SEARCH_SCOPE="${SEARCH_SCOPE:-${DEFAULT_SEARCH_SCOPE}}" \
        DISABLE_MEASUREMENT_HISTORY="${DISABLE_LEGACY_HISTORY}" \
        SEARCH_OUTPUT="${SEARCH_CANDIDATES_CSV}" \
        SEARCH_ALL_OUTPUT="${SEARCH_ALL_CSV}" \
        SEARCH_TILING_DIR="${SEARCH_TILING_DIR}" \
        "${ROOT}/scripts/run_search.sh" "${WORKLOADS_CSV}"

if [[ ! -s "${SEARCH_CANDIDATES_CSV}" ]]; then
    echo "Search did not produce its candidate input" >&2
    echo "  phase_log: ${SEARCH_LOG}"
    exit 1
fi

PLATFORM_LINE="$(sed -n '/^CANN platform=/{p;q;}' "${SEARCH_LOG}")"
platform_field() {
    printf '%s\n' "${PLATFORM_LINE}" |
        sed -n "s/.*[[:space:]]$1=\\([0-9][0-9.]*\\).*/\\1/p"
}
PLATFORM_AIC_CORES="$(platform_field cores)"
PLATFORM_L0A_BYTES="$(platform_field L0A)"
PLATFORM_L0B_BYTES="$(platform_field L0B)"
PLATFORM_L0C_BYTES="$(platform_field L0C)"
PLATFORM_L1_BYTES="$(platform_field L1)"
for value in \
    "${PLATFORM_AIC_CORES}" "${PLATFORM_L0A_BYTES}" \
    "${PLATFORM_L0B_BYTES}" "${PLATFORM_L0C_BYTES}" \
    "${PLATFORM_L1_BYTES}"; do
    if [[ -z "${value}" ]]; then
        echo "Cannot read complete platform capacities from the official tiling search" >&2
        echo "  phase_log: ${SEARCH_LOG}" >&2
        exit 1
    fi
done
if [[ "${PLATFORM_AIC_CORES}" -le 0 ]]; then
    echo "Platform AIC count is not positive" >&2
    echo "  phase_log: ${SEARCH_LOG}" >&2
    exit 1
fi
echo "  platform: aic=${PLATFORM_AIC_CORES} L0A=${PLATFORM_L0A_BYTES} L0B=${PLATFORM_L0B_BYTES} L0C=${PLATFORM_L0C_BYTES} L1=${PLATFORM_L1_BYTES}"

echo "[4/4] Run NPU ACL Event profiling ..."
PROFILE_LOG="${ROOT}/results/logs/profile_$(date +%Y%m%d_%H%M%S).log"
LAST_PHASE_LOG="${PROFILE_LOG}"
ERROR_ALREADY_REPORTED=0
profile_cmd=(
    "${ROOT}/scripts/profile_npu.sh"
    "${SEARCH_CANDIDATES_CSV}"
    "${RESULT_STEM}"
    "${WORKLOADS_CSV}"
)
profile_env=(
    "RANK_LIMIT=${RANK_LIMIT:-${DEFAULT_RANK_LIMIT}}"
    "WARMUP=${WARMUP:-${DEFAULT_WARMUP}}"
    "REPEAT=${REPEAT:-${DEFAULT_REPEAT}}"
    "SAMPLES=${SAMPLES:-${DEFAULT_SAMPLES}}"
    "NUMERIC_PREFLIGHT_MAX_MIB=${NUMERIC_PREFLIGHT_MAX_MIB:-${DEFAULT_NUMERIC_PREFLIGHT_MAX_MIB}}"
    "PROFILE_STALL_TIMEOUT_SEC=${PROFILE_STALL_TIMEOUT_SEC:-${DEFAULT_PROFILE_STALL_TIMEOUT_SEC}}"
    "PROFILE_PROGRESS_EVERY=${PROFILE_PROGRESS_EVERY:-${DEFAULT_PROFILE_PROGRESS_EVERY}}"
    "DISABLE_MEASUREMENT_HISTORY=${DISABLE_LEGACY_HISTORY}"
    "SKIP_BANK_SEED_CONTROL=${SKIP_BANK_SEED_CONTROL}"
    "REQUIRE_EXACT_RESUME_PREFIX=${REQUIRE_EXACT_RESUME_PREFIX:-$(
        if [[ "${MODE}" == "full" && "${WORKLOADS_CSV}" == "config/workloads.csv" && -s "${RESULT_STEM}_resume.csv" ]]; then
            printf '%s' "${DEFAULT_REQUIRE_EXACT_RESUME_PREFIX}"
        else
            printf '0'
        fi
    )}"
    "PLATFORM_AIC_CORES=${PLATFORM_AIC_CORES}"
    "PLATFORM_L0A_BYTES=${PLATFORM_L0A_BYTES}"
    "PLATFORM_L0B_BYTES=${PLATFORM_L0B_BYTES}"
    "PLATFORM_L0C_BYTES=${PLATFORM_L0C_BYTES}"
    "PLATFORM_L1_BYTES=${PLATFORM_L1_BYTES}"
    "DEVICE_ID=${DEVICE_ID:-0}"
)
if [[ "${RUN_VERBOSE}" == "1" ]]; then
    if env "${profile_env[@]}" PROFILE_VERBOSE=1 "${profile_cmd[@]}" 2>&1 | tee "${PROFILE_LOG}"; then
        profile_rc=0
    else
        profile_rc=${PIPESTATUS[0]}
    fi
else
    if env "${profile_env[@]}" "${profile_cmd[@]}" 2>&1 | tee "${PROFILE_LOG}" | filter_profile_terminal; then
        profile_rc=0
    else
        profile_rc=${PIPESTATUS[0]}
    fi
fi
unset profile_env
if [[ "${profile_rc}" -ne 0 ]]; then
    echo "[4/4] Run NPU ACL Event profiling ... failed"
    if [[ -s "${RESULT_STEM}_resume.csv" ]]; then
        echo "  completed exact measurements were preserved:"
        echo "  ${ROOT}/${RESULT_STEM}_resume.csv"
    fi
    if grep -q '^TILING_ERROR_BEGIN$' "${PROFILE_LOG}"; then
        echo "  classification: tiling/kernel preflight failure"
        echo "  phase_log:      ${PROFILE_LOG}"
    else
        print_error_lines "${PROFILE_LOG}"
    fi
    if grep -Eaiq \
        '507008|aclInit failed|aclrtBinaryLoadFromFile|RegisterAscendBinary|(^|[^0-9])107000([^0-9]|$)|symbol lookup error|undefined symbol' \
        "${PROFILE_LOG}"; then
        DIAG_LOG="${ROOT}/results/logs/runtime_diag_$(date +%Y%m%d_%H%M%S).log"
        "${ROOT}/scripts/diagnose_runtime.sh" >"${DIAG_LOG}" 2>&1 || true
        echo "runtime_diag: ${DIAG_LOG}" >>"${PROFILE_LOG}"
        print_diag_summary "${DIAG_LOG}"
    fi
    if grep -Eaiq '(^|[^0-9])507008([^0-9]|$)' "${PROFILE_LOG}"; then
        run_server_check_summary
    fi
    ERROR_ALREADY_REPORTED=1
    exit "${profile_rc}"
fi
echo "[4/4] Run NPU ACL Event profiling ... ok"

if [[ "${KEEP_DETAILS:-0}" == "1" ]]; then
    DETAILS_DIR="${RESULT_STEM}_details"
    mkdir -p "${DETAILS_DIR}/tilings"
    cp "${SEARCH_CANDIDATES_CSV}" "${DETAILS_DIR}/search_candidates.csv"
    cp "${SEARCH_ALL_CSV}" "${DETAILS_DIR}/all_evaluated.csv"
    cp -a "${SEARCH_TILING_DIR}/." "${DETAILS_DIR}/tilings/"
fi

cleanup_search_work
SEARCH_WORK_DIR=""

if [[ "${PRINT_RESULTS:-1}" == "1" ]]; then
    echo
    PRINT_SUMMARY_ARGS=()
    if [[ "${MODE}" == "full" ]]; then
        PRINT_SUMMARY_ARGS=(--all-workloads --direct-comparison-only)
    elif [[ "${PRINT_ALL_RESULTS:-0}" == "1" ]]; then
        PRINT_SUMMARY_ARGS=(--all-workloads)
    fi
    if ! python3 "${ROOT}/tools/print_npu_summary.py" \
        --summary "${RESULT_STEM}_summary.csv" \
        --candidates "${RESULT_STEM}_candidates.csv" \
        "${PRINT_SUMMARY_ARGS[@]}"; then
        echo "warning: compact result rendering failed; result CSV files are valid"
    fi
fi

echo
echo "NPU run completed"
echo "  Summary:     ${ROOT}/${RESULT_STEM}_summary.csv"
echo "  Candidates:  ${ROOT}/${RESULT_STEM}_candidates.csv"
echo "  Resume:      ${ROOT}/${RESULT_STEM}_resume.csv"
if [[ "${KEEP_DETAILS:-0}" == "1" ]]; then
    echo "  Details:     ${ROOT}/${RESULT_STEM}_details/"
fi
echo "  log:         ${RUN_LOG}"
