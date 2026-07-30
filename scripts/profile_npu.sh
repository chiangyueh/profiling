#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"
cd "$ROOT"

CANDIDATES="${1:-results/candidates.csv}"
OUT_STEM="${2:-results/npu}"
WORKLOADS="${3:-${CANDIDATES}}"
OUT_DIR="$(dirname "${OUT_STEM}")"
OUT_NAME="$(basename "${OUT_STEM}")"
WORK_DIR="${OUT_DIR}/.${OUT_NAME}_work_$$"
PROFILE_CSV="${WORK_DIR}/tiling_profile.csv"
SAMPLES_CSV="${WORK_DIR}/tiling_samples.csv"
OFFICIAL_PROFILE_CSV="${WORK_DIR}/official_profile.csv"
OFFICIAL_SAMPLES_CSV="${WORK_DIR}/official_samples.csv"
RANKED_CSV="${WORK_DIR}/ranked.csv"
BEST_CSV="${WORK_DIR}/best.csv"
SUMMARY_CSV="${WORK_DIR}/summary.csv"
FINAL_CANDIDATES_CSV="${OUT_STEM}_candidates.csv"
FINAL_SUMMARY_CSV="${OUT_STEM}_summary.csv"
RESUME_CSV="${OUT_STEM}_resume.csv"
RESUME_RUN_ID="$(date +%Y%m%d_%H%M%S)"
HISTORY_CSV="${MEASUREMENT_HISTORY:-results/npu_full_ocr_measurements.csv}"
HISTORY_ARGS=()
if [[ -f "${HISTORY_CSV}" ]]; then
    HISTORY_ARGS=(--history "${HISTORY_CSV}")
fi

mkdir -p "${OUT_DIR}" "${WORK_DIR}"

cleanup() {
    if [[ -s "${PROFILE_CSV}" && -n "${PLATFORM_AIC_CORES:-}" ]]; then
        merge_resume_history >/dev/null 2>&1 || true
    fi
    if [[ -d "${WORK_DIR}" ]]; then
        find "${WORK_DIR}" -mindepth 1 -delete
        rmdir "${WORK_DIR}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

merge_resume_history() {
    local sources=()
    local source
    for source in \
        "${RESUME_CSV}" "${FINAL_CANDIDATES_CSV}" \
        "${OFFICIAL_PROFILE_CSV}" "${PROFILE_CSV}"; do
        if [[ -s "${source}" ]]; then
            sources+=("${source}")
        fi
    done
    if [[ "${#sources[@]}" -eq 0 ]]; then
        return
    fi
    local count
    count="$(
        PYTHONPATH="${ROOT}/tools" python3 - \
            "${RESUME_CSV}" \
            "${ASCENDC_SOC_VERSION:-${SOC_VERSION}}" \
            "${PLATFORM_AIC_CORES}" \
            "${RESUME_RUN_ID}" \
            "${sources[@]}" <<'PY'
import sys
from pathlib import Path

from profile_official_tilings import merge_exact_profile_history

count = merge_exact_profile_history(
    Path(sys.argv[1]),
    [Path(value) for value in sys.argv[5:]],
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4],
)
print(count)
PY
    )"
    echo "resume_history: exact_rows=${count} path=${RESUME_CSV}"
}

finalize_results() {
    if [[ ! -s "${PROFILE_CSV}" || "$(wc -l <"${PROFILE_CSV}")" -le 1 ]]; then
        return 1
    fi
    if ! python3 tools/rank_npu_results.py \
        --input "${PROFILE_CSV}" \
        --official-input "${OFFICIAL_PROFILE_CSV}" \
        --output "${RANKED_CSV}" \
        --summary "${BEST_CSV}" \
        --comparison "${SUMMARY_CSV}" \
        >/dev/null; then
        return 1
    fi
    mv -f "${RANKED_CSV}" "${FINAL_CANDIDATES_CSV}"
    mv -f "${SUMMARY_CSV}" "${FINAL_SUMMARY_CSV}"
}

preserve_details() {
    if [[ "${KEEP_DETAILS:-0}" != "1" ]]; then
        return
    fi
    DETAILS_DIR="${OUT_STEM}_details"
    if [[ -d "${DETAILS_DIR}" ]]; then
        find "${DETAILS_DIR}" -mindepth 1 -delete
    fi
    mkdir -p "${DETAILS_DIR}"
    cp "${PROFILE_CSV}" "${DETAILS_DIR}/profile.csv"
    cp "${SAMPLES_CSV}" "${DETAILS_DIR}/samples.csv"
    cp "${OFFICIAL_PROFILE_CSV}" "${DETAILS_DIR}/official_profile.csv"
    cp "${OFFICIAL_SAMPLES_CSV}" "${DETAILS_DIR}/official_samples.csv"
    if [[ -s "${BEST_CSV}" ]]; then
        cp "${BEST_CSV}" "${DETAILS_DIR}/best.csv"
    fi
}

if [[ ! -x build/official_matmul_runner ]]; then
    echo "fatal: build/official_matmul_runner is missing" >&2
    exit 2
fi
if [[ ! -x build/tiling_bank_probe ]]; then
    echo "fatal: build/tiling_bank_probe is missing" >&2
    exit 2
fi
for name in \
    PLATFORM_AIC_CORES PLATFORM_L0A_BYTES PLATFORM_L0B_BYTES \
    PLATFORM_L0C_BYTES PLATFORM_L1_BYTES; do
    if [[ -z "${!name:-}" ]]; then
        echo "fatal: ${name} was not provided by the official tiling search" >&2
        exit 2
    fi
done

merge_resume_history
PROFILE_HISTORY_ARGS=()
if [[ -s "${RESUME_CSV}" ]]; then
    PROFILE_HISTORY_ARGS=(--profile-history "${RESUME_CSV}")
fi
RESUME_GUARD_ARGS=()
if [[ "${REQUIRE_EXACT_RESUME_PREFIX:-0}" -gt 0 ]]; then
    RESUME_GUARD_ARGS=(
        --require-exact-resume-prefix
        "${REQUIRE_EXACT_RESUME_PREFIX}"
    )
fi

profile_rc=0
if python3 tools/profile_official_tilings.py \
    --runner build/official_matmul_runner \
    --bank-probe build/tiling_bank_probe \
    --candidates "${CANDIDATES}" \
    --workloads "${WORKLOADS}" \
    --custom-output "${PROFILE_CSV}" \
    --custom-samples-output "${SAMPLES_CSV}" \
    --official-output "${OFFICIAL_PROFILE_CSV}" \
    --official-samples-output "${OFFICIAL_SAMPLES_CSV}" \
    "${HISTORY_ARGS[@]}" \
    "${PROFILE_HISTORY_ARGS[@]}" \
    "${RESUME_GUARD_ARGS[@]}" \
    --cann-root "${CANN_ROOT}" \
    --soc "${ASCENDC_SOC_VERSION:-${SOC_VERSION}}" \
    --aic-cores "${PLATFORM_AIC_CORES}" \
    --l0a-bytes "${PLATFORM_L0A_BYTES}" \
    --l0b-bytes "${PLATFORM_L0B_BYTES}" \
    --l0c-bytes "${PLATFORM_L0C_BYTES}" \
    --l1-bytes "${PLATFORM_L1_BYTES}" \
    --device "${DEVICE_ID:-0}" \
    --warmup "${WARMUP:-10}" \
    --repeat "${REPEAT:-50}" \
    --samples "${SAMPLES:-15}" \
    --rank-limit "${RANK_LIMIT:-20}" \
    --numeric-preflight-max-mib "${NUMERIC_PREFLIGHT_MAX_MIB:-4}" \
    --timeout-sec "${PROFILE_STALL_TIMEOUT_SEC:-60}" \
    --reference-retries "${REFERENCE_RETRIES:-2}" \
    --progress-every "${PROFILE_PROGRESS_EVERY:-10}"; then
    profile_rc=0
else
    profile_rc=$?
fi
merge_resume_history

results_ready=0
if [[ "${profile_rc}" -eq 0 ]]; then
    if finalize_results; then
        results_ready=1
    fi
fi
preserve_details

if [[ "${profile_rc}" -ne 0 ]]; then
    echo "profile_npu failed"
    if [[ "${results_ready}" == "1" ]]; then
        echo "partial_results:"
        echo "  summary:    ${FINAL_SUMMARY_CSV}"
        echo "  candidates: ${FINAL_CANDIDATES_CSV}"
    fi
    exit "${profile_rc}"
fi
if [[ "${results_ready}" != "1" ]]; then
    echo "fatal: profiling completed without rankable NPU rows" >&2
    exit 1
fi

echo "profile_npu completed"
echo "  summary:    ${FINAL_SUMMARY_CSV}"
echo "  candidates: ${FINAL_CANDIDATES_CSV}"
