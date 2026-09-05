#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

CANDIDATES="${1:-results/candidates.csv}"
OUT_STEM="${2:-results/npu}"
WORKLOADS="${3:-${CANDIDATES}}"
DETAILS_DIR="${OUT_STEM}_details"
PROFILE_CSV="${DETAILS_DIR}/profile.csv"
SAMPLES_CSV="${DETAILS_DIR}/samples.csv"
OFFICIAL_PROFILE_CSV="${DETAILS_DIR}/official_profile.csv"
OFFICIAL_SAMPLES_CSV="${DETAILS_DIR}/official_samples.csv"
MANIFEST="${DETAILS_DIR}/direct_manifest.csv"
TILING_DIRECTORY="${DETAILS_DIR}/direct_tilings"
RANKED_CSV="${OUT_STEM}_candidates.csv"
SUMMARY_CSV="${OUT_STEM}_summary.csv"
BEST_CSV="${DETAILS_DIR}/best.csv"

[[ -x build/official_matmul_runner ]] || {
    echo "fatal: build/official_matmul_runner is missing" >&2
    exit 2
}
for name in PLATFORM_AIC_CORES PLATFORM_L2_BYTES; do
    [[ -n "${!name:-}" ]] || { echo "fatal: ${name} is missing" >&2; exit 2; }
done
[[ -n "${MEASUREMENT_JSONL_LOG_DIRECTORY:-}" ]] || {
    echo "fatal: MEASUREMENT_JSONL_LOG_DIRECTORY is missing" >&2
    exit 2
}
mkdir -p "${DETAILS_DIR}" "${TILING_DIRECTORY}"

python3 tools/profile_direct_matmul.py \
    --variant-builder scripts/build_all.sh \
    --variant-runner-directory build/direct_runners \
    --official-runner build/official_matmul_runner \
    --candidates "${CANDIDATES}" \
    --workloads "${WORKLOADS}" \
    --manifest "${MANIFEST}" \
    --tiling-directory "${TILING_DIRECTORY}" \
    --profile-output "${PROFILE_CSV}" \
    --samples-output "${SAMPLES_CSV}" \
    --official-output "${OFFICIAL_PROFILE_CSV}" \
    --official-samples-output "${OFFICIAL_SAMPLES_CSV}" \
    --log-directory "${MEASUREMENT_JSONL_LOG_DIRECTORY}" \
    --log-max-bytes "${MEASUREMENT_JSONL_LOG_MAX_BYTES:-52428800}" \
    --l2-bytes "${PLATFORM_L2_BYTES}" \
    --aic-cores "${PLATFORM_AIC_CORES}" \
    --device "${DEVICE_ID:-0}" \
    --warmup "${WARMUP:-1}" \
    --repeat "${REPEAT:-1}" \
    --samples "${SAMPLES:-3}" \
    --progress-every "${PROFILE_PROGRESS_EVERY:-20}"

python3 tools/rank_npu_results.py \
    --input "${PROFILE_CSV}" \
    --official-input "${OFFICIAL_PROFILE_CSV}" \
    --output "${RANKED_CSV}" \
    --summary "${BEST_CSV}" \
    --comparison "${SUMMARY_CSV}" \
    --official-only-comparison >/dev/null

echo "NPU_RESULTS_READY candidates=${RANKED_CSV} summary=${SUMMARY_CSV}"
