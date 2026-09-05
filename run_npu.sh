#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

MatMulV3 original-source-route frontier mapping for the three preregistered
Victor shapes.  The formal output contains 2,160 validated candidate
latencies plus three separate official baselines.
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
while IFS='=' read -r name _; do
    case "${name}" in
        *RUNTIME_KB*|*TUNING_BANK*) unset "${name}" ;;
    esac
done < <(env)
source "${ROOT}/scripts/env.sh" >/dev/null

CANN_VERSION_FILE="${CANN_ROOT}/version.cfg"
MATMUL_V3_SOURCE_DIR="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3"
[[ -f "${CANN_VERSION_FILE}" ]] || {
    echo "fatal: CANN version.cfg is missing: ${CANN_VERSION_FILE}" >&2
    exit 2
}
grep -Eq '^toolkit_running_version=.*:8\.1' "${CANN_VERSION_FILE}" || {
    echo "fatal: this direct campaign requires installed CANN 8.1" >&2
    exit 2
}
[[ -f "${MATMUL_V3_SOURCE_DIR}/mat_mul_v3.cpp" ]] || {
    echo "fatal: installed CANN 8.1 MatMulV3 source is missing" >&2
    exit 2
}

catalog_started_ns="$(date +%s%N)"
CATALOG_TMP="$(mktemp "${TMPDIR:-/tmp}/matmul-victor-frontier.XXXXXX.csv")"
cleanup() {
    [[ -f "${CATALOG_TMP:-}" ]] && rm -f -- "${CATALOG_TMP}"
}
trap cleanup EXIT

python3 tools/generate_matmul_victor_frontier_workloads.py \
    --output "${CATALOG_TMP}" >/dev/null
catalog_wall_ms=$(( ($(date +%s%N) - catalog_started_ns) / 1000000 ))
CAMPAIGN_ID="$({
    sha256sum \
        "${CATALOG_TMP}" \
        tools/generate_matmul_victor_frontier_workloads.py \
        tools/generate_matmul_victor_frontier_candidates.py \
        tools/collect_matmul_source_routes.py \
        tools/analyze_matmul_hardware_calibration.py \
        tools/direct_matmul_tiling.py \
        tools/profile_direct_matmul.py \
        tools/refine_matmul_v3_candidates.py \
        run_npu.sh \
        scripts/run_search.sh \
        scripts/profile_npu.sh \
        runner/official_matmul_runner.cpp \
        direct_matmul/kernel_entry.cpp \
        direct_matmul/mat_mul_v3_tiling_data.h \
        direct_matmul/runner.cpp \
        npu_cost_model/*.py \
        "${CANN_VERSION_FILE}"
    find src/matmul/mat_mul_v3 cmake -type f -print0 | sort -z | xargs -0 sha256sum
    find "${MATMUL_V3_SOURCE_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_source_route_frontier_direct_v1/${CAMPAIGN_ID}"
CATALOG="${CAMPAIGN_DIR}/catalog.csv"
WORKLOADS="${CAMPAIGN_DIR}/workloads.csv"
CANDIDATES="${CAMPAIGN_DIR}/candidates.csv"
ALL_CANDIDATES="${CAMPAIGN_DIR}/fixed_design_scored.csv"
TILING_DIR="${CAMPAIGN_DIR}/tilings"
OUT_STEM="${CAMPAIGN_DIR}/measurement"
DETAILS_DIR="${OUT_STEM}_details"
LOG_DIR="${CAMPAIGN_DIR}/logs"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SOURCE_AUDIT="${CAMPAIGN_DIR}/source_routes.jsonl"
mkdir -p "${CAMPAIGN_DIR}" "${TILING_DIR}" "${LOG_DIR}"
cp "${CATALOG_TMP}" "${CATALOG}"

if [[ -s "${ANALYSIS}" ]] && \
   grep -q '"status": "complete"' "${ANALYSIS}"; then
    echo "MATMUL_VICTOR_FRONTIER_COMPLETE shapes=3 records=2163"
    echo "analysis=${ANALYSIS} logs=${LOG_DIR}"
    exit 0
fi

echo "CAMPAIGN_READY operator=matmul shapes=3 candidate_records=2160 official_baselines=3 records=2163 device=${PHYSICAL_DEVICE}"
echo "measurement=1_warmup+3_device_event_samples+validate_last_timed_output"
echo "design=every_applicable_original_source_route_and_core_cap_plus_hardware_local_frontier"
echo "candidate_selection=source_routes_no_runtimekb_no_latency_no_cost_score"
echo "logs=${LOG_DIR}"
echo "CAMPAIGN_STAGE_TIMING stage=workload_catalog wall_ms=${catalog_wall_ms}"

HOST_BUILD_HASH="$({
    find host compat -type f -print0
    printf '%s\0' scripts/build_all.sh "${CANN_VERSION_FILE}"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
HOST_BUILD_STAMP="${ROOT}/build/.matmul_platform_host.sha256"
host_build_started_ns="$(date +%s%N)"
if [[ ! -x build/matmul_tiling_search || ! -f "${HOST_BUILD_STAMP}" || \
      "$(cat "${HOST_BUILD_STAMP}" 2>/dev/null || true)" != "${HOST_BUILD_HASH}" ]]; then
    echo "PLATFORM_HOST_BUILD begin jobs=1"
    if ! BUILD_COMPONENTS=host BUILD_JOBS=1 scripts/build_all.sh \
        >"${CAMPAIGN_DIR}/platform_host_build.log" 2>&1; then
        echo "PLATFORM_HOST_BUILD failed log=${CAMPAIGN_DIR}/platform_host_build.log"
        tail -20 "${CAMPAIGN_DIR}/platform_host_build.log"
        exit 1
    fi
    printf '%s\n' "${HOST_BUILD_HASH}" >"${HOST_BUILD_STAMP}"
    echo "PLATFORM_HOST_BUILD passed"
    host_build_cached=0
else
    echo "PLATFORM_HOST_BUILD cached"
    host_build_cached=1
fi
host_build_wall_ms=$(( ($(date +%s%N) - host_build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=platform_host_build wall_ms=${host_build_wall_ms} cached=${host_build_cached}"

RUNNER_BUILD_HASH="$({
    printf '%s\0' runner/official_matmul_runner.cpp \
        cmake_npu/CMakeLists.txt scripts/build_all.sh "${CANN_VERSION_FILE}"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
RUNNER_BUILD_STAMP="${ROOT}/build/.matmul_official_runner.sha256"
runner_build_started_ns="$(date +%s%N)"
if [[ ! -x build/official_matmul_runner || ! -f "${RUNNER_BUILD_STAMP}" || \
      "$(cat "${RUNNER_BUILD_STAMP}" 2>/dev/null || true)" != "${RUNNER_BUILD_HASH}" ]]; then
    echo "OFFICIAL_RUNNER_BUILD begin jobs=1"
    if ! BUILD_COMPONENTS=official BUILD_JOBS=1 scripts/build_all.sh \
        >"${CAMPAIGN_DIR}/official_runner_build.log" 2>&1; then
        echo "OFFICIAL_RUNNER_BUILD failed log=${CAMPAIGN_DIR}/official_runner_build.log"
        tail -20 "${CAMPAIGN_DIR}/official_runner_build.log"
        exit 1
    fi
    printf '%s\n' "${RUNNER_BUILD_HASH}" >"${RUNNER_BUILD_STAMP}"
    echo "OFFICIAL_RUNNER_BUILD passed"
    runner_build_cached=0
else
    echo "OFFICIAL_RUNNER_BUILD cached"
    runner_build_cached=1
fi
runner_build_wall_ms=$(( ($(date +%s%N) - runner_build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=official_runner_build wall_ms=${runner_build_wall_ms} cached=${runner_build_cached}"

SOURCE_HOST_HASH="$({
    find src/matmul/mat_mul_v3 cmake -type f -print0 | sort -z | xargs -0 sha256sum
    sha256sum CMakeLists.txt "${CANN_VERSION_FILE}"
} | sha256sum | cut -c1-20)"
SOURCE_STATE="${ROOT}/.benchmark_state/matmul_source_route_host/${SOURCE_HOST_HASH}"
SOURCE_BUILD="${SOURCE_STATE}/build"
SOURCE_INSTALL="${SOURCE_STATE}/install"
SOURCE_PACKAGE_ROOT="${SOURCE_INSTALL}/packages/vendors/matmul_source_routes"
SOURCE_OPAPI="${SOURCE_PACKAGE_ROOT}/op_api/lib/libcust_opapi.so"
SOURCE_TILING_LIB_GLOB="${SOURCE_PACKAGE_ROOT}/op_impl/ai_core/tbe/op_tiling/lib/linux"/*/libcust_opmaster_rt2.0.so
source_host_started_ns="$(date +%s%N)"
if [[ ! -f "${SOURCE_OPAPI}" ]] || ! compgen -G "${SOURCE_TILING_LIB_GLOB}" >/dev/null; then
    mkdir -p "${SOURCE_BUILD}" "${SOURCE_INSTALL}"
    echo "SOURCE_HOST_PACKAGE_BUILD begin jobs=1"
    SOURCE_BUILD_LOG="${SOURCE_STATE}/build.log"
    if ! cmake -S "${ROOT}" -B "${SOURCE_BUILD}" -G "Unix Makefiles" \
        -DBUILD_OPEN_PROJECT=ON \
        -DASCEND_COMPUTE_UNIT=ascend910b \
        -DASCEND_OP_NAME=mat_mul_v3 \
        -DVENDOR_NAME=matmul_source_routes \
        -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" \
        -DASCEND_THIRD_LIB_PATH="${ROOT}/third_party" \
        -DENABLE_OPS_KERNEL=OFF \
        -DENABLE_TEST=OFF \
        -DENABLE_EXAMPLE=OFF \
        -DENABLE_CCACHE=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${SOURCE_INSTALL}" \
        >"${SOURCE_BUILD_LOG}" 2>&1 || \
       ! cmake --build "${SOURCE_BUILD}" --target package -- -j1 \
        >>"${SOURCE_BUILD_LOG}" 2>&1 || \
       ! cmake --install "${SOURCE_BUILD}" >>"${SOURCE_BUILD_LOG}" 2>&1; then
        echo "SOURCE_HOST_PACKAGE_BUILD failed log=${SOURCE_BUILD_LOG}"
        tail -20 "${SOURCE_BUILD_LOG}"
        exit 1
    fi
    echo "SOURCE_HOST_PACKAGE_BUILD passed"
    source_host_cached=0
else
    echo "SOURCE_HOST_PACKAGE_BUILD cached"
    source_host_cached=1
fi
if [[ ! -f "${SOURCE_OPAPI}" ]] || ! compgen -G "${SOURCE_TILING_LIB_GLOB}" >/dev/null; then
    echo "fatal: private MatMul source host package is incomplete" >&2
    exit 1
fi
SOURCE_TILING_LIBRARY="$(compgen -G "${SOURCE_TILING_LIB_GLOB}" | head -1)"
source_host_wall_ms=$(( ($(date +%s%N) - source_host_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=source_host_package wall_ms=${source_host_wall_ms} cached=${source_host_cached}"

source_route_started_ns="$(date +%s%N)"
python3 tools/collect_matmul_source_routes.py \
    --runner "${ROOT}/build/official_matmul_runner" \
    --workloads "${CATALOG}" \
    --audit "${SOURCE_AUDIT}" \
    --state-dir "${CAMPAIGN_DIR}/source_route_state" \
    --package-root "${SOURCE_PACKAGE_ROOT}" \
    --opapi "${SOURCE_OPAPI}" \
    --tiling-library "${SOURCE_TILING_LIBRARY}" \
    --device 0 \
    --max-cores 20
source_route_wall_ms=$(( ($(date +%s%N) - source_route_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=source_route_discovery wall_ms=${source_route_wall_ms}"

export DISABLE_MEASUREMENT_HISTORY=1
export SEARCH_SCOPE=matmul_victor_frontier_v1
export SEARCH_OUTPUT="${CANDIDATES}"
export SEARCH_ALL_OUTPUT="${ALL_CANDIDATES}"
export SEARCH_TILING_DIR="${TILING_DIR}"
export FRONTIER_WORKLOADS_OUTPUT="${WORKLOADS}"
export MEASUREMENT_JSONL_LOG_DIRECTORY="${LOG_DIR}"
export MEASUREMENT_JSONL_LOG_MAX_BYTES=52428800
export SOURCE_ROUTE_AUDIT

candidate_contract() {
python3 - "${WORKLOADS}" "${CANDIDATES}" "${ALL_CANDIDATES}" "${SOURCE_AUDIT}" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

if not all(Path(value).is_file() for value in sys.argv[1:]):
    raise SystemExit(1)
workloads = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
candidates = list(csv.DictReader(open(sys.argv[2], newline="", encoding="utf-8")))
all_candidates = list(csv.DictReader(open(sys.argv[3], newline="", encoding="utf-8")))
workload_ids = [row["workload_id"] for row in workloads]
candidate_ids = [
    row["workload_id"] for row in candidates
    if row.get("candidate_role") == "searched"
]
searched = Counter(
    row["workload_id"] for row in candidates
    if row.get("candidate_role") == "searched"
)
hashes = {}
source_all = set()
for row in candidates:
    hashes.setdefault(row["workload_id"], set()).add(
        row.get("model_schedule_sha256", "")
    )
    if (
        row.get("candidate_role") == "searched"
        and row.get("source_anchor") == "1"
        and "ALL" in row.get("source_route", "").split("+")
        and "20" in row.get("source_core_cap", "").split(",")
        and len(row.get("source_raw_tiling_hex", "")) == 544
    ):
        source_all.add(row["workload_id"])
if (
    len(workloads) != 3
    or len(set(workload_ids)) != 3
    or len({
        (row["m"], row["n"], row["k"], row["dtype"],
         row["trans_a"], row["trans_b"])
        for row in workloads
    }) != 3
    or {row.get("search_family") for row in workloads} != {"source_route_frontier"}
    or any("target_kernel_family" in row for row in workloads)
    or len(candidates) != 2256
    or set(candidate_ids) != set(workload_ids)
    or len(searched) != 3
    or any(
        searched[row["workload_id"]]
        != int(row["required_successful_tilings"]) + 32
        for row in workloads
    )
    or len(hashes) != 3
    or any(len(hashes[row["workload_id"]]) != searched[row["workload_id"]] for row in workloads)
    or source_all != set(workload_ids)
    or not all_candidates
    or not all(row.get("global_model_rank", "").isdigit() for row in candidates)
    or not all(row.get("controlled_factor", "") for row in candidates)
    or not all(row.get("candidate_set_frozen_before_model_scoring") == "1" for row in candidates)
):
    raise SystemExit(1)
PY
}
if candidate_contract >/dev/null 2>&1
then
    export REUSE_VICTOR_FRONTIER_CANDIDATES=1
fi

SEARCH_LOG="${CAMPAIGN_DIR}/candidate_generation.log"
search_started_ns="$(date +%s%N)"
PROFILE_WORKLOADS="${WORKLOADS}"
set +e
source "${ROOT}/scripts/run_search.sh" "${CATALOG}" \
    > >(tee "${SEARCH_LOG}" | awk '
        /VICTOR_FRONTIER_CANDIDATES \[/ {
            split(substr($2,2,length($2)-2), a, "/");
            if (a[1] == 1 || a[1] == a[2] || a[1] % 20 == 0) print;
        }
        /MATMUL_VICTOR_FRONTIER_CANDIDATES|fatal:/ {print}
    ') 2>&1
search_rc=$?
set -e
WORKLOADS="${PROFILE_WORKLOADS}"
search_wall_ms=$(( ($(date +%s%N) - search_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=tiling_selection wall_ms=${search_wall_ms}" | tee -a "${SEARCH_LOG}"
if [[ "${search_rc}" -ne 0 ]]; then
    echo "CANDIDATE_GENERATION_FAILED log=${SEARCH_LOG}"
    exit "${search_rc}"
fi
export PLATFORM_AIC_CORES PLATFORM_L0A_BYTES PLATFORM_L0B_BYTES
export PLATFORM_L0C_BYTES PLATFORM_L1_BYTES PLATFORM_L2_BYTES
export PLATFORM_L2_BPC PLATFORM_HBM_BPC

if ! candidate_contract >/dev/null 2>&1; then
    echo "CANDIDATE_CONTRACT_FAILED log=${SEARCH_LOG}"
    exit 1
fi

direct_preflight_started_ns="$(date +%s%N)"
python3 tools/direct_matmul_tiling.py \
    --candidates "${CANDIDATES}" \
    --output-dir "${DETAILS_DIR}/direct_tilings" \
    --manifest "${DETAILS_DIR}/direct_manifest.csv" \
    --l2-bytes "${PLATFORM_L2_BYTES}" \
    --aic-cores "${PLATFORM_AIC_CORES}" \
    --include-reserves >/dev/null
direct_preflight_wall_ms=$(( ($(date +%s%N) - direct_preflight_started_ns) / 1000000 ))
echo "DIRECT_TILING_PREFLIGHT passed candidates=2256 wall_ms=${direct_preflight_wall_ms}"

echo "NPU_MEASUREMENT_BEGIN shapes=3 candidate_records=2160 official_baselines=3 records=2163"
export KEEP_DETAILS=1
export WARMUP=1
export REPEAT=1
export SAMPLES=3
export PROFILE_PROGRESS_EVERY=20

PROFILE_LOG="${CAMPAIGN_DIR}/measurement_progress.log"
profile_started_ns="$(date +%s%N)"
set +e
"${ROOT}/scripts/profile_npu.sh" \
    "${CANDIDATES}" "${OUT_STEM}" "${WORKLOADS}" \
    > >(awk '
        /OFFICIAL_BASELINE_|DIRECT_VARIANT_|DIRECT_MEASUREMENT_|NPU_RESULTS_READY|fatal:|Traceback/ {
            print; fflush();
        }
    ' | tee "${PROFILE_LOG}") 2>&1
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
python3 tools/analyze_matmul_hardware_calibration.py \
    --workloads "${WORKLOADS}" \
    --candidates "${CANDIDATES}" \
    --profile "${DETAILS_DIR}/profile.csv" \
    --official-profile "${DETAILS_DIR}/official_profile.csv" \
    --output "${ANALYSIS}" \
    --log-directory "${LOG_DIR}" \
    --expected-shapes 3 \
    --require-direct-tiling-applied
analysis_wall_ms=$(( ($(date +%s%N) - analysis_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
echo "analysis=${ANALYSIS}"
