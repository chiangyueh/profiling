#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Paired MatMulV3 model validation: 200 unique shapes, one final simulator
tiling selected from a hardware-derived ideal neighbourhood and one official
autotiling baseline per shape (400 latency records).
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
source "${ROOT}/scripts/env.sh" >/dev/null

catalog_started_ns="$(date +%s%N)"
CATALOG_TMP="$(mktemp "${TMPDIR:-/tmp}/matmul-model-validation.XXXXXX.csv")"
cleanup() {
    [[ -f "${CATALOG_TMP:-}" ]] && rm -f -- "${CATALOG_TMP}"
}
trap cleanup EXIT

python3 tools/generate_matmul_model_validation_workloads.py \
    --output "${CATALOG_TMP}" >/dev/null
catalog_wall_ms=$(( ($(date +%s%N) - catalog_started_ns) / 1000000 ))
CAMPAIGN_ID="$({
    sha256sum \
        "${CATALOG_TMP}" \
        tools/generate_matmul_model_validation_workloads.py \
        tools/generate_matmul_model_validation_candidates.py \
        tools/analyze_matmul_model_validation.py \
        tools/refine_matmul_v3_candidates.py \
        tools/profile_official_tilings.py \
        scripts/profile_npu.sh \
        runner/official_matmul_runner.cpp \
        npu_cost_model/*.py
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_model_validation_v3/${CAMPAIGN_ID}"
CATALOG="${CAMPAIGN_DIR}/catalog.csv"
WORKLOADS="${CAMPAIGN_DIR}/workloads.csv"
CANDIDATES="${CAMPAIGN_DIR}/candidates.csv"
ALL_CANDIDATES="${CAMPAIGN_DIR}/all_callback_fixed.csv"
TILING_DIR="${CAMPAIGN_DIR}/tilings"
OUT_STEM="${CAMPAIGN_DIR}/measurement"
DETAILS_DIR="${OUT_STEM}_details"
LOG_DIR="${CAMPAIGN_DIR}/logs"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
mkdir -p "${CAMPAIGN_DIR}" "${TILING_DIR}" "${LOG_DIR}"
cp "${CATALOG_TMP}" "${CATALOG}"

if [[ -s "${ANALYSIS}" ]] && \
   grep -q '"status": "complete"' "${ANALYSIS}"; then
    echo "MATMUL_MODEL_VALIDATION_COMPLETE shapes=200 records=400"
    echo "analysis=${ANALYSIS} logs=${LOG_DIR}"
    exit 0
fi

echo "CAMPAIGN_READY operator=matmul shapes=200 selected_tilings_per_shape=1 records=400 device=${PHYSICAL_DEVICE}"
echo "comparison=official_matmul_v3,new_hardware_simulator"
echo "search=all_declared_execution_graphs,hardware_ideal_region,adjacent_transitions"
echo "logs=${LOG_DIR}"
echo "CAMPAIGN_STAGE_TIMING stage=workload_catalog wall_ms=${catalog_wall_ms}"

BUILD_INPUT_HASH="$({
    find host runner cmake_npu -type f -print0
    printf '%s\0' tools/tiling_bank_probe.cpp scripts/build_all.sh
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
BUILD_STAMP="${ROOT}/build/.matmul_model_validation_build.sha256"
LEGACY_BUILD_STAMP="${ROOT}/build/.matmul_controlled_build.sha256"
if [[ ! -f "${BUILD_STAMP}" && -f "${LEGACY_BUILD_STAMP}" && \
      "$(cat "${LEGACY_BUILD_STAMP}" 2>/dev/null || true)" == "${BUILD_INPUT_HASH}" && \
      -x build/matmul_tiling_search && -x build/official_matmul_runner && \
      -x build/tiling_bank_probe ]]; then
    printf '%s\n' "${BUILD_INPUT_HASH}" >"${BUILD_STAMP}"
fi
build_started_ns="$(date +%s%N)"
if [[ ! -x build/matmul_tiling_search || \
      ! -x build/official_matmul_runner || \
      ! -x build/tiling_bank_probe || \
      ! -f "${BUILD_STAMP}" || \
      "$(cat "${BUILD_STAMP}" 2>/dev/null || true)" != "${BUILD_INPUT_HASH}" ]]; then
    echo "RUNNER_BUILD begin jobs=1"
    BUILD_JOBS=1 scripts/build_all.sh >"${CAMPAIGN_DIR}/build.log" 2>&1
    printf '%s\n' "${BUILD_INPUT_HASH}" >"${BUILD_STAMP}"
    echo "RUNNER_BUILD passed"
    build_cached=0
else
    echo "RUNNER_BUILD cached"
    build_cached=1
fi
build_wall_ms=$(( ($(date +%s%N) - build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=runner_build wall_ms=${build_wall_ms} cached=${build_cached}"

export DISABLE_MEASUREMENT_HISTORY=1
export SEARCH_SCOPE=matmul_model_validation_v3
export SEARCH_OUTPUT="${CANDIDATES}"
export SEARCH_ALL_OUTPUT="${ALL_CANDIDATES}"
export SEARCH_TILING_DIR="${TILING_DIR}"
export MODEL_VALIDATION_WORKLOADS_OUTPUT="${WORKLOADS}"
export MEASUREMENT_JSONL_LOG_DIRECTORY="${LOG_DIR}"
export MEASUREMENT_JSONL_LOG_MAX_BYTES=52428800

if python3 - "${WORKLOADS}" "${CANDIDATES}" "${ALL_CANDIDATES}" <<'PY' >/dev/null 2>&1
import csv
import sys
from collections import Counter
from pathlib import Path

if not all(Path(value).is_file() for value in sys.argv[1:]):
    raise SystemExit(1)
workloads = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
candidates = list(csv.DictReader(open(sys.argv[2], newline="", encoding="utf-8")))
searched = Counter(
    row["workload_id"] for row in candidates
    if row.get("candidate_role") == "searched"
)
hashes = {}
for row in candidates:
    hashes.setdefault(row["workload_id"], set()).add(
        row.get("callback_tiling_sha256", "")
    )
if (
    len(workloads) != 200
    or {row.get("search_family") for row in workloads} != {"hardware_ideal_region"}
    or len(candidates) != 200
    or len(searched) != 200 or set(searched.values()) != {1}
    or len(hashes) != 200 or {len(value) for value in hashes.values()} != {1}
):
    raise SystemExit(1)
PY
then
    export REUSE_MODEL_VALIDATION_CANDIDATES=1
fi

SEARCH_LOG="${CAMPAIGN_DIR}/candidate_generation.log"
search_started_ns="$(date +%s%N)"
set +e
source "${ROOT}/scripts/run_search.sh" "${CATALOG}" \
    > >(tee "${SEARCH_LOG}" | awk '
        /MODEL_VALIDATION_CANDIDATES \[/ {
            split(substr($2,2,length($2)-2), a, "/");
            if (a[1] == 1 || a[1] == a[2] || a[1] % 20 == 0) print;
        }
        /MATMUL_MODEL_VALIDATION_CANDIDATES|fatal:/ {print}
    ') 2>&1
search_rc=$?
set -e
search_wall_ms=$(( ($(date +%s%N) - search_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=tiling_selection wall_ms=${search_wall_ms}" | tee -a "${SEARCH_LOG}"
if [[ "${search_rc}" -ne 0 ]]; then
    echo "CANDIDATE_GENERATION_FAILED log=${SEARCH_LOG}"
    exit "${search_rc}"
fi
export PLATFORM_AIC_CORES PLATFORM_L0A_BYTES PLATFORM_L0B_BYTES
export PLATFORM_L0C_BYTES PLATFORM_L1_BYTES PLATFORM_L2_BYTES
export PLATFORM_L2_BPC PLATFORM_HBM_BPC

echo "NPU_MEASUREMENT_BEGIN shapes=200 simulator_tilings=200 official_baselines=200"
export KEEP_DETAILS=1
export WARMUP=2
export REPEAT=20
export SAMPLES=7
export RANK_LIMIT=1
export SUCCESSFUL_TILINGS_PER_WORKLOAD=1
export SKIP_BANK_SEED_CONTROL=1
export STRUCTURED_FULL_PREFLIGHT=1
export NUMERIC_PREFLIGHT_MAX_MIB=256
export PROFILE_STALL_TIMEOUT_SEC=0
export PROFILE_PROGRESS_EVERY=50

PROFILE_LOG="${CAMPAIGN_DIR}/measurement_progress.log"
profile_started_ns="$(date +%s%N)"
set +e
"${ROOT}/scripts/profile_npu.sh" \
    "${CANDIDATES}" "${OUT_STEM}" "${WORKLOADS}" \
    > >(tee "${PROFILE_LOG}" | awk '
        /profile_plan:|WORKLOAD_GROUP_RESULT|official_tiling_profile completed|fatal:|candidate_abort|profile_npu failed/ {print}
    ') 2>&1
profile_rc=$?
set -e
profile_wall_ms=$(( ($(date +%s%N) - profile_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=npu_measurement wall_ms=${profile_wall_ms}" | tee -a "${PROFILE_LOG}"
if [[ "${profile_rc}" -ne 0 ]]; then
    echo "NPU_MEASUREMENT_INCOMPLETE log=${PROFILE_LOG} records=${LOG_DIR}"
    exit "${profile_rc}"
fi

for required in \
    "${DETAILS_DIR}/profile.csv" \
    "${DETAILS_DIR}/official_profile.csv"; do
    [[ -s "${required}" ]] || {
        echo "ANALYSIS_FAILED missing=${required}"
        exit 1
    }
done

analysis_started_ns="$(date +%s%N)"
python3 tools/analyze_matmul_model_validation.py \
    --workloads "${WORKLOADS}" \
    --candidates "${CANDIDATES}" \
    --profile "${DETAILS_DIR}/profile.csv" \
    --official-profile "${DETAILS_DIR}/official_profile.csv" \
    --output "${ANALYSIS}" \
    --log-directory "${LOG_DIR}"
analysis_wall_ms=$(( ($(date +%s%N) - analysis_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
echo "analysis=${ANALYSIS}"
