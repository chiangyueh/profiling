#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Measures the MatMul hardware-cost frontier on a physical NPU:
250 deterministic shapes x 20 distinct, numerically validated tilings = 5,000 records.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
        -d|--device) PHYSICAL_DEVICE="${2:?missing physical NPU ID}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "fatal: unsupported argument: $1" >&2; exit 2 ;;
    esac
done

[[ "${MODE}" == "full" ]] || { usage >&2; exit 2; }
[[ "${PHYSICAL_DEVICE}" =~ ^[0-9]+$ ]] || {
    echo "fatal: physical NPU ID must be a non-negative integer" >&2
    exit 2
}

cd "${ROOT}"
export CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-Ascend910B3}"
export SOC_VERSION="${SOC_VERSION:-${ASCENDC_SOC_VERSION}}"
export ASCEND_RT_VISIBLE_DEVICES="${PHYSICAL_DEVICE}"
export DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset ASCEND_CUSTOM_OPP_PATH || true

# All environment changes are process-local. No installed OPP, tuning bank,
# toolkit file, driver setting, or another user's cache is modified.
source "${ROOT}/scripts/env.sh" >/dev/null

CATALOG_TMP="$(mktemp "${TMPDIR:-/tmp}/matmul-controlled-workloads.XXXXXX.csv")"
STAGED_LOGS=""
cleanup() {
    [[ -f "${CATALOG_TMP:-}" ]] && rm -f -- "${CATALOG_TMP}"
    if [[ -n "${STAGED_LOGS:-}" && -d "${STAGED_LOGS}" ]]; then
        find "${STAGED_LOGS}" -mindepth 1 -maxdepth 1 -type f -delete
        rmdir "${STAGED_LOGS}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

python3 tools/generate_matmul_controlled_workloads.py \
    --output "${CATALOG_TMP}" >/dev/null
CAMPAIGN_ID="$({
    sha256sum \
        "${CATALOG_TMP}" \
        tools/generate_matmul_controlled_workloads.py \
        tools/generate_matmul_controlled_candidates.py \
        tools/export_matmul_controlled_frontier.py \
        tools/profile_official_tilings.py \
        runner/official_matmul_runner.cpp
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_controlled_frontier_v1/${CAMPAIGN_ID}"
WORKLOADS="${CAMPAIGN_DIR}/workloads.csv"
CANDIDATES="${CAMPAIGN_DIR}/candidates.csv"
ALL_CANDIDATES="${CAMPAIGN_DIR}/all_callback_fixed.csv"
TILING_DIR="${CAMPAIGN_DIR}/tilings"
OUT_STEM="${CAMPAIGN_DIR}/measurement"
DETAILS_DIR="${OUT_STEM}_details"
LOG_DIR="${CAMPAIGN_DIR}/logs"
PROGRESS_LOG="${CAMPAIGN_DIR}/progress.log"
mkdir -p "${CAMPAIGN_DIR}" "${TILING_DIR}"
cp "${CATALOG_TMP}" "${WORKLOADS}"

if [[ -s "${LOG_DIR}/1.log" ]] && \
   grep -q '"record_type":"campaign_summary".*"status":"complete"' \
       "${LOG_DIR}"/*.log 2>/dev/null; then
    echo "MATMUL_CONTROLLED_FRONTIER_COMPLETE records=5000 shapes=250 logs=${LOG_DIR}"
    exit 0
fi

echo "CAMPAIGN_READY operator=matmul records=5000 shapes=250 tilings_per_shape=20"
echo "blocks=l2:1600,concurrency:1200,buffer:1200,splitk:1000 device=${PHYSICAL_DEVICE}"
echo "logs=${LOG_DIR}"

BUILD_INPUT_HASH="$({
    find host runner cmake_npu -type f -print0
    printf '%s\0' tools/tiling_bank_probe.cpp scripts/build_all.sh
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
BUILD_STAMP="${ROOT}/build/.matmul_controlled_build.sha256"
if [[ ! -x build/matmul_tiling_search || \
      ! -x build/official_matmul_runner || \
      ! -x build/tiling_bank_probe || \
      ! -f "${BUILD_STAMP}" || \
      "$(<"${BUILD_STAMP}")" != "${BUILD_INPUT_HASH}" ]]; then
    echo "RUNNER_BUILD begin jobs=1"
    BUILD_JOBS=1 scripts/build_all.sh >"${CAMPAIGN_DIR}/build.log" 2>&1
    printf '%s\n' "${BUILD_INPUT_HASH}" >"${BUILD_STAMP}"
    echo "RUNNER_BUILD passed"
else
    echo "RUNNER_BUILD cached"
fi

export DISABLE_MEASUREMENT_HISTORY=1
export SEARCH_SCOPE=controlled_frontier_v1
export TOP_K=28
export BEAM_WIDTH=64
export SEARCH_OUTPUT="${CANDIDATES}"
export SEARCH_ALL_OUTPUT="${ALL_CANDIDATES}"
export SEARCH_TILING_DIR="${TILING_DIR}"

if python3 - "${CANDIDATES}" "${ALL_CANDIDATES}" <<'PY' >/dev/null 2>&1
import csv
import sys
from collections import Counter
from pathlib import Path

if not all(Path(value).is_file() for value in sys.argv[1:]):
    raise SystemExit(1)
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
controls = Counter(
    row["workload_id"] for row in rows
    if row.get("candidate_role") == "bank_seed_control"
)
searched = Counter(
    row["workload_id"] for row in rows
    if row.get("candidate_role") == "searched"
)
if (
    len(controls) != 250
    or set(controls.values()) != {1}
    or len(searched) != 250
    or set(searched.values()) != {28}
):
    raise SystemExit(1)
PY
then
    export REUSE_CONTROLLED_CANDIDATES=1
fi

SEARCH_LOG="${CAMPAIGN_DIR}/candidate_generation.log"
set +e
source "${ROOT}/scripts/run_search.sh" "${WORKLOADS}" \
    > >(tee "${SEARCH_LOG}" | awk '
        /CONTROLLED_CANDIDATES/ {
            if ($2 ~ /^\[/) {
                split(substr($2,2,length($2)-2), a, "/");
                if (a[1] == 1 || a[1] == a[2] || a[1] % 25 == 0) print;
            } else print;
        }
        /MATMUL_CONTROLLED_CANDIDATES|fatal:/ {print}
    ') 2>&1
search_rc=$?
set -e
if [[ "${search_rc}" -ne 0 ]]; then
    echo "SOURCE_TILING_CANDIDATE_GENERATION_FAILED log=${SEARCH_LOG}"
    exit "${search_rc}"
fi
export PLATFORM_AIC_CORES PLATFORM_L0A_BYTES PLATFORM_L0B_BYTES
export PLATFORM_L0C_BYTES PLATFORM_L1_BYTES PLATFORM_L2_BYTES
export PLATFORM_L2_BPC PLATFORM_HBM_BPC

echo "SOURCE_TILING_MEASUREMENT_BEGIN shapes=250 target=5000"
export KEEP_DETAILS=1
export WARMUP=2
export REPEAT=20
export SAMPLES=7
export RANK_LIMIT=28
export SUCCESSFUL_TILINGS_PER_WORKLOAD=20
export STRUCTURED_FULL_PREFLIGHT=1
export NUMERIC_PREFLIGHT_MAX_MIB=256
export PROFILE_STALL_TIMEOUT_SEC=0
export PROFILE_PROGRESS_EVERY=100

PROFILE_LOG="${CAMPAIGN_DIR}/measurement_progress.log"
set +e
"${ROOT}/scripts/profile_npu.sh" \
    "${CANDIDATES}" "${OUT_STEM}" "${WORKLOADS}" \
    > >(tee "${PROFILE_LOG}" | awk '
        /profile_plan:|WORKLOAD_GROUP_RESULT|official_tiling_profile completed|fatal:|candidate_abort|candidate_rejected|profile_npu failed/ {print}
    ') 2>&1
profile_rc=$?
set -e
if [[ "${profile_rc}" -ne 0 ]]; then
    echo "SOURCE_TILING_MEASUREMENT_FAILED log=${PROFILE_LOG}"
    exit "${profile_rc}"
fi

for required in \
    "${DETAILS_DIR}/profile.csv" \
    "${DETAILS_DIR}/samples.csv" \
    "${DETAILS_DIR}/official_profile.csv" \
    "${DETAILS_DIR}/official_samples.csv"; do
    [[ -s "${required}" ]] || {
        echo "SOURCE_TILING_EXPORT_FAILED missing=${required}"
        exit 1
    }
done

STAGED_LOGS="${CAMPAIGN_DIR}/.logs_stage_$$"
mkdir "${STAGED_LOGS}"
python3 tools/export_matmul_controlled_frontier.py \
    --workloads "${WORKLOADS}" \
    --candidates "${CANDIDATES}" \
    --profile "${DETAILS_DIR}/profile.csv" \
    --samples "${DETAILS_DIR}/samples.csv" \
    --official-profile "${DETAILS_DIR}/official_profile.csv" \
    --official-samples "${DETAILS_DIR}/official_samples.csv" \
    --log-directory "${STAGED_LOGS}" \
    --soc "${ASCENDC_SOC_VERSION}" \
    --aic-cores "${PLATFORM_AIC_CORES}" >/dev/null

mkdir -p "${LOG_DIR}"
find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -type f \
    -name '[0-9]*.log' -delete
mv "${STAGED_LOGS}"/*.log "${LOG_DIR}/"
rmdir "${STAGED_LOGS}"
STAGED_LOGS=""

printf '%s\n' \
    'MATMUL_CONTROLLED_FRONTIER_COMPLETE records=5000 shapes=250' \
    'blocks=l2:1600,concurrency:1200,buffer:1200,splitk:1000' \
    "logs=${LOG_DIR}" >"${PROGRESS_LOG}"
cat "${PROGRESS_LOG}"
