#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-1}"
WARMUP="${OP_NPU_WARMUP:-2}"
SAMPLES="${OP_NPU_SAMPLES:-5}"
BUILD_JOBS=1
RECORD_TARGET=5000

usage() {
    cat <<'USAGE'
Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs only the GatherElements original-source tiling collection.

Formal output target: 5,000 real-NPU latency records. Every admitted shape
has at least 20 distinct source-rule tilings which each complete an installed
reference equality check and a device-event measurement. The source search
uses the original GatherElements dispatcher across bounded AIV caps 1..20;
only if that yields fewer than 20 raw identities does its declared UB-capacity
envelope (2/4/8) run. No MatMul, attention, ScatterElements, CCE table,
historical tiling, or historical latency is used.

All generated source, build, OPP-view, and logs stay below this profiling
checkout. The installed CANN built-in OPP is linked read-only; it is never
copied into or modified by this command.
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
if [[ ! -d "${CANN_ROOT}" || ! -f "${CANN_ROOT}/opp/version.info" ]]; then
    echo "fatal: CANN root or its OPP package is missing: ${CANN_ROOT}" >&2
    exit 1
fi
ENV_FILE=""
for candidate in "${CANN_ROOT}/set_env.sh" "$(dirname "${CANN_ROOT}")/set_env.sh"; do
    if [[ -f "${candidate}" ]]; then ENV_FILE="${candidate}"; break; fi
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
export TILINGKEY_PAR_COMPILE=1
export OMP_NUM_THREADS=1
unset ASCEND_CUSTOM_OPP_PATH ASCENDC_CPU_DEBUG

PRIVATE_SOURCE_CACHE="${ROOT}/.source_cache/repo_bundles_v1"
mkdir -p "${PRIVATE_SOURCE_CACHE}"

prepare_repo_source() {
    local kind="$1"
    local destination="$2"
    if [[ -f "${destination}/.source_bundle_attestation.json" ]]; then return; fi
    if [[ -e "${destination}" ]]; then
        echo "fatal: private repository-source cache is incomplete: ${destination}" >&2
        exit 2
    fi
    echo "SOURCE_PREPARE {\"source\":\"${kind}\",\"destination\":\"${destination}\",\"scope\":\"this_profiling_checkout_only\",\"network_calls\":0}"
    python3 "${ROOT}/source_adapter/materialize_repo_source_bundle.py" --kind "${kind}" --destination "${destination}"
}

OPS_SOURCE="${PRIVATE_SOURCE_CACHE}/cann_ops"
GATHER_EXTRACT_SOURCE="${PRIVATE_SOURCE_CACHE}/gather_elements_v2"
prepare_repo_source cann_ops "${OPS_SOURCE}"
prepare_repo_source gather_elements_v2 "${GATHER_EXTRACT_SOURCE}"

SOURCE_ID="$({
    sha256sum \
        "${ROOT}/run_npu.sh" \
        "${ROOT}/multi_op_bench/CMakeLists.txt" \
        "${ROOT}/multi_op_bench/runner.cpp" \
        "${ROOT}/source_adapter/non_matmul_source_lock.json" \
        "${ROOT}/source_adapter/non_matmul_candidate_catalog.py" \
        "${ROOT}/source_adapter/prepare_gather_elements_compat_overlay.py" \
        "${ROOT}/source_adapter/build_source_candidate_overlay.py" \
        "${ROOT}/source_adapter/materialize_repo_source_bundle.py" \
        "${ROOT}/source_adapter/materialize_installed_dynamic_opp.py" \
        "${ROOT}/source_adapter/find_reusable_source_tiler_cache.py" \
        "${ROOT}/source_adapter/run_source_tiler_smoke.py" \
        "${ROOT}/source_adapter/reset_incomplete_private_state.py" \
        "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py"
    sha256sum "${ROOT}/source_adapter/vendor_source/cann_ops_8_1rc1.tar.gz" \
        "${ROOT}/source_adapter/vendor_source/gather_elements_v2_source.zip"
    readlink -f "${CANN_ROOT}"
    sha256sum "${GATHER_EXTRACT_SOURCE}/op_host/gather_elements_v2_tiling.cpp"
} | sha256sum | cut -c1-20)"
STATE="${ROOT}/.benchmark_state/gather_elements_source_candidate_v1/${SOURCE_ID}"
OVERLAY_PARENT="${STATE}/overlays"
PACKAGE_BUILD_PARENT="${STATE}/package_builds"
CUSTOM_OPP_PARENT="${STATE}/custom_opp"
RUNNER_BUILD="${STATE}/runner_build"
RESULTS="${ROOT}/results/gather_elements_source_candidate_v1/${SOURCE_ID}"
LOGS="${RESULTS}/logs"
mkdir -p "${OVERLAY_PARENT}" "${PACKAGE_BUILD_PARENT}" "${CUSTOM_OPP_PARENT}" "${RUNNER_BUILD}" "${LOGS}"

echo "GatherElements original-source tiling campaign"
echo "  target:             physical NPU ${PHYSICAL_DEVICE} -> worker logical NPU 0"
echo "  operator:           GatherElements only"
echo "  formal target:      ${RECORD_TARGET} output-validated device-event latency records"
echo "  group gate:         at least 20 distinct successful source-rule tilings per shape"
echo "  source search:      original GatherElements dispatcher × AIV caps 1..20; declared UB 2/4/8 only below 20 identities"
echo "  build:              one GatherElements host tiler only; no FASG/FIAS/Scatter build or dynamic OPP materialization"
echo "  installed CANN:     private read-only built-in OPP link; no CANN files are written"
echo "  output:             ${LOGS}/1.log, 2.log, ... (JSONL records; each file <= 50 MiB)"

GATHER_OVERLAY="${OVERLAY_PARENT}/gather/gather_elements_v2_compat_source/source_candidate_overlay.json"
if [[ ! -f "${GATHER_OVERLAY}" ]]; then
    mkdir -p "${OVERLAY_PARENT}/gather"
    echo "SOURCE_OVERLAY_PREPARE_BEGIN operator=gather_elements"
    if ! python3 "${ROOT}/source_adapter/prepare_gather_elements_compat_overlay.py" \
        --parent-source-root "${OPS_SOURCE}" --extracted-source-root "${GATHER_EXTRACT_SOURCE}" \
        --output-parent "${OVERLAY_PARENT}/gather" >"${STATE}/gather_overlay_prepare.log" 2>&1; then
        tail -100 "${STATE}/gather_overlay_prepare.log" >&2 || true
        exit 1
    fi
    echo "SOURCE_OVERLAY_PREPARE_END operator=gather_elements"
fi

LABEL="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1], encoding="utf-8")); print((m["cmake_op_name"] + "__" + (m.get("strategy_class") or "dispatch")).lower())' "${GATHER_OVERLAY}")"
BUILD_DIR="${PACKAGE_BUILD_PARENT}/${LABEL}_host_tiler"
BUILD_MANIFEST="${BUILD_DIR}/source_candidate_build.json"
BUILD_LOG="${STATE}/${LABEL}_source_host_tiler_build.log"
CUSTOM_ROOT="${CUSTOM_OPP_PARENT}/${LABEL}"
CUSTOM_MANIFEST="${CUSTOM_ROOT}/source_candidate_package.json"

cmake -S "${ROOT}/multi_op_bench" -B "${RUNNER_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}"
cmake --build "${RUNNER_BUILD}" --target multi_op_npu_runner --parallel "${BUILD_JOBS}"

# A matching host-tiler artifact from the old private state remains usable.
# Its former OPP package is deliberately not reused: this release creates the
# correct private ASCEND_OPP_PATH view around the artifact without rebuilding.
if [[ -f "${CUSTOM_MANIFEST}" ]]; then
    REUSE_STATUS="local"
else
    REUSE_JSON="$(python3 "${ROOT}/source_adapter/find_reusable_source_tiler_cache.py" \
        --state-parent "${ROOT}/.benchmark_state/non_matmul_source_candidate_v5" \
        --overlay-manifest "${GATHER_OVERLAY}" --label "${LABEL}" \
        --installed-op-impl "${CANN_ROOT}/opp/built-in/op_impl")"
    REUSE_STATUS="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<<"${REUSE_JSON}")"
fi
if [[ "${REUSE_STATUS}" == "reused" || "${REUSE_STATUS}" == "repackage" ]]; then
    BUILD_MANIFEST="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["build_manifest"])' <<<"${REUSE_JSON}")"
fi

if [[ "${REUSE_STATUS}" == "local" ]]; then
    echo "SOURCE_HOST_TILER_CACHE_REUSE source=${LABEL} package=current_private_state"
elif [[ "${REUSE_STATUS}" == "reused" ]]; then
    CUSTOM_MANIFEST="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["custom_opp_manifest"])' <<<"${REUSE_JSON}")"
    echo "SOURCE_HOST_TILER_CACHE_REUSE source=${LABEL} package=complete"
elif [[ "${REUSE_STATUS}" == "repackage" ]]; then
    echo "SOURCE_HOST_TILER_CACHE_REUSE source=${LABEL} package=rematerialize_only"
elif [[ "${REUSE_STATUS}" == "absent" ]]; then
    if [[ ! -f "${BUILD_MANIFEST}" ]]; then
        if [[ -e "${BUILD_DIR}" ]]; then
            python3 "${ROOT}/source_adapter/reset_incomplete_private_state.py" \
                --parent "${PACKAGE_BUILD_PARENT}" --target "${BUILD_DIR}" \
                --required-absent "${BUILD_MANIFEST}" --kind host_tiler_build
        fi
        echo "SOURCE_HOST_TILER_BUILD_BEGIN source=${LABEL}"
        if ! python3 "${ROOT}/source_adapter/build_source_candidate_overlay.py" \
            --overlay "$(dirname "${GATHER_OVERLAY}")" --build-dir "${BUILD_DIR}" \
            --cann-root "${CANN_ROOT}" --target optiling --jobs "${BUILD_JOBS}" >"${BUILD_LOG}" 2>&1; then
            tail -100 "${BUILD_LOG}" >&2 || true
            exit 1
        fi
        echo "SOURCE_HOST_TILER_BUILD_END source=${LABEL}"
    else
        echo "SOURCE_HOST_TILER_CACHE_REUSE source=${LABEL} package=local_build"
    fi
else
    echo "fatal: invalid GatherElements host-tiler cache status: ${REUSE_STATUS}" >&2
    exit 1
fi

if [[ "${REUSE_STATUS}" != "reused" && "${REUSE_STATUS}" != "local" ]]; then
    if [[ -e "${CUSTOM_ROOT}" ]]; then
        python3 "${ROOT}/source_adapter/reset_incomplete_private_state.py" \
            --parent "${CUSTOM_OPP_PARENT}" --target "${CUSTOM_ROOT}" \
            --required-absent "${CUSTOM_MANIFEST}" --kind dynamic_opp_root
    fi
    mkdir -p "${CUSTOM_ROOT}"
    echo "SOURCE_DYNAMIC_OPP_MATERIALIZE_BEGIN source=${LABEL}"
    if ! python3 "${ROOT}/source_adapter/materialize_installed_dynamic_opp.py" \
        --build-manifest "${BUILD_MANIFEST}" --installed-op-impl "${CANN_ROOT}/opp/built-in/op_impl" \
        --destination "${CUSTOM_ROOT}" >"${STATE}/${LABEL}_dynamic_opp_materialize.log" 2>&1; then
        tail -100 "${STATE}/${LABEL}_dynamic_opp_materialize.log" >&2 || true
        exit 1
    fi
    echo "SOURCE_DYNAMIC_OPP_MATERIALIZE_END source=${LABEL}"
fi

# This executor-only smoke must emit the source audit before any semantic
# shape or timed device launch begins. A failure exits here with zero formal
# measurements instead of producing a large rejection log.
SMOKE_LOG="${LOGS}/preflight_gather_elements.log"
if ! python3 "${ROOT}/source_adapter/run_source_tiler_smoke.py" \
    --runner "${RUNNER_BUILD}/multi_op_npu_runner" --device 0 \
    --custom-opp-manifest "${CUSTOM_MANIFEST}" --work-dir "${STATE}/smoke" >"${SMOKE_LOG}" 2>&1; then
    tail -100 "${SMOKE_LOG}" >&2 || true
    exit 1
fi
grep '^SOURCE_TILER_EARLY_SMOKE ' "${SMOKE_LOG}"

PYTHONPATH="${ROOT}/multi_op_bench" python3 "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
    --runner "${RUNNER_BUILD}/multi_op_npu_runner" --log-dir "${LOGS}" --device 0 \
    --warmup "${WARMUP}" --samples "${SAMPLES}" --operator gather_elements \
    --record-target "${RECORD_TARGET}" --custom-opp-manifest "${CUSTOM_MANIFEST}"
