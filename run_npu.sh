#!/usr/bin/env bash
set -Eeuo pipefail

# This entry point intentionally runs one operator only: GatherElementsV2.
# It builds an isolated CANN-8.1 custom package below this checkout, then
# calls its generated API. The installed aclnnGather route is used only as
# the output reference. No installed CANN/OPP file is copied or modified.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-1}"
WARMUP="${OP_NPU_WARMUP:-2}"
SAMPLES="${OP_NPU_SAMPLES:-5}"
BUILD_JOBS=1
RECORD_TARGET=5000

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs only the complete GatherElementsV2 source package campaign.

- 5,000 formal real-NPU device-event latency records (250 groups x 20).
- A group counts only if 20 distinct source-generated raw tilings each launch
  the source kernel and exactly match an installed aclnnGather reference.
- The source dispatcher is tried at AIV caps 1..20; only then may its declared
  UB-capacity envelope 2/4/8 be tried. No raw tiling field is edited or
  replayed.
- All source cache, build state and JSONL logs remain under this checkout.
  Each numeric log is capped at 50 MiB. The installed CANN tree is read-only.
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
# Do not inherit a different user's custom operator tree into either the
# installed reference path or this private package.  Source workers add this
# run's exact private root explicitly.
unset ASCEND_CUSTOM_OPP_PATH ASCENDC_CPU_DEBUG

PRIVATE_SOURCE_CACHE="${ROOT}/.source_cache/gather_elements_complete_v2"
mkdir -p "${PRIVATE_SOURCE_CACHE}"
GATHER_SOURCE="${PRIVATE_SOURCE_CACHE}/gather_elements_v2"
if [[ ! -f "${GATHER_SOURCE}/.source_bundle_attestation.json" ]]; then
    if [[ -e "${GATHER_SOURCE}" ]]; then
        echo "fatal: incomplete private GatherElements source cache exists: ${GATHER_SOURCE}" >&2
        exit 2
    fi
    echo "SOURCE_PREPARE_BEGIN source=gather_elements_v2 scope=profiling_checkout_only network_calls=0"
    python3 "${ROOT}/source_adapter/materialize_repo_source_bundle.py" \
        --kind gather_elements_v2 --destination "${GATHER_SOURCE}"
    echo "SOURCE_PREPARE_END source=gather_elements_v2"
fi

SOURCE_ID="$({
    sha256sum \
        "${ROOT}/run_npu.sh" \
        "${ROOT}/multi_op_bench/CMakeLists.txt" \
        "${ROOT}/multi_op_bench/runner.cpp" \
        "${ROOT}/source_adapter/non_matmul_source_lock.json" \
        "${ROOT}/source_adapter/non_matmul_candidate_catalog.py" \
        "${ROOT}/source_adapter/prepare_gather_elements_custom_package.py" \
        "${ROOT}/source_adapter/build_gather_elements_complete_custom_package.py" \
        "${ROOT}/source_adapter/materialize_repo_source_bundle.py" \
        "${ROOT}/source_adapter/reset_incomplete_private_state.py" \
        "${ROOT}/source_adapter/run_source_tiler_smoke.py" \
        "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py"
    sha256sum "${ROOT}/source_adapter/vendor_source/gather_elements_v2_source.zip"
    sha256sum "${GATHER_SOURCE}/op_host/gather_elements_v2_tiling.cpp"
    readlink -f "${CANN_ROOT}"
    sha256sum "${CANN_ROOT}/opp/version.info"
} | sha256sum | cut -c1-20)"

STATE="${ROOT}/.benchmark_state/gather_elements_complete_custom_v2/${SOURCE_ID}"
OVERLAY_PARENT="${STATE}/overlays"
PACKAGE_PARENT="${STATE}/package_builds"
RUNNER_BUILD="${STATE}/runner_build"
RESULTS="${ROOT}/results/gather_elements_complete_custom_v2/${SOURCE_ID}"
LOGS="${RESULTS}/logs"
mkdir -p "${OVERLAY_PARENT}" "${PACKAGE_PARENT}" "${RUNNER_BUILD}" "${LOGS}"

echo "GatherElementsV2 complete original-source tiling campaign"
echo "  target:             physical NPU ${PHYSICAL_DEVICE} -> worker logical NPU 0"
echo "  formal target:      ${RECORD_TARGET} output-validated device-event latency records"
echo "  group gate:         at least 20 distinct successful source tilings per shape"
echo "  source execution:   generated aclnnGatherElementsV2 API + source host tiler + source kernel"
echo "  reference only:     installed aclnnGather"
echo "  search:             original dispatcher × AIV 1..20; declared UB 2/4/8 only below 20 identities"
echo "  installed CANN:     read-only; all new files are below ${ROOT}"
echo "  output:             ${LOGS}/1.log, 2.log, ... (JSONL, each <= 50 MiB)"

OVERLAY="${OVERLAY_PARENT}/gather_elements_v2_complete_custom"
OVERLAY_MANIFEST="${OVERLAY}/source_candidate_overlay.json"
if [[ ! -f "${OVERLAY_MANIFEST}" ]]; then
    if [[ -e "${OVERLAY}" ]]; then
        echo "fatal: incomplete private GatherElements custom overlay exists: ${OVERLAY}" >&2
        exit 2
    fi
    echo "SOURCE_COMPLETE_OVERLAY_PREPARE_BEGIN operator=gather_elements_v2"
    python3 "${ROOT}/source_adapter/prepare_gather_elements_custom_package.py" \
        --extracted-source-root "${GATHER_SOURCE}" \
        --template-root "${CANN_ROOT}/tools/op_project_templates/ascendc/customize" \
        --output-parent "${OVERLAY_PARENT}" >"${STATE}/overlay_prepare.log" 2>&1
    echo "SOURCE_COMPLETE_OVERLAY_PREPARE_END operator=gather_elements_v2"
fi

PACKAGE_BUILD="${PACKAGE_PARENT}/gather_elements_v2_complete_custom"
PACKAGE_MANIFEST="${PACKAGE_BUILD}/complete_custom_package.json"
if [[ ! -f "${PACKAGE_MANIFEST}" && -e "${PACKAGE_BUILD}" ]]; then
    python3 "${ROOT}/source_adapter/reset_incomplete_private_state.py" \
        --parent "${PACKAGE_PARENT}" --target "${PACKAGE_BUILD}" \
        --required-absent "${PACKAGE_MANIFEST}" --kind complete_custom_package
fi
if [[ ! -f "${PACKAGE_MANIFEST}" ]]; then
    echo "SOURCE_COMPLETE_PACKAGE_BUILD_BEGIN operator=gather_elements_v2"
    if ! python3 "${ROOT}/source_adapter/build_gather_elements_complete_custom_package.py" \
        --overlay "${OVERLAY}" --build-dir "${PACKAGE_BUILD}" --cann-root "${CANN_ROOT}" \
        --vendor gather_elements_v2_source --jobs "${BUILD_JOBS}" >"${STATE}/complete_package_build.log" 2>&1; then
        tail -120 "${STATE}/complete_package_build.log" >&2 || true
        exit 1
    fi
    echo "SOURCE_COMPLETE_PACKAGE_BUILD_END operator=gather_elements_v2"
else
    echo "SOURCE_COMPLETE_PACKAGE_CACHE_REUSE operator=gather_elements_v2"
fi

cmake -S "${ROOT}/multi_op_bench" -B "${RUNNER_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}"
cmake --build "${RUNNER_BUILD}" --target multi_op_npu_runner --parallel "${BUILD_JOBS}"

# The only NPU smoke is the source custom API itself. It first creates a
# source executor (which must emit the source audit), then the campaign makes
# one source-kernel viability launch before any semantic shape discovery.
SMOKE_LOG="${LOGS}/preflight_gather_elements.log"
if ! python3 "${ROOT}/source_adapter/run_source_tiler_smoke.py" \
    --runner "${RUNNER_BUILD}/multi_op_npu_runner" --device 0 \
    --custom-opp-manifest "${PACKAGE_MANIFEST}" --work-dir "${STATE}/smoke" >"${SMOKE_LOG}" 2>&1; then
    tail -120 "${SMOKE_LOG}" >&2 || true
    exit 1
fi
grep '^SOURCE_TILER_EARLY_SMOKE ' "${SMOKE_LOG}"

PYTHONPATH="${ROOT}/multi_op_bench" python3 "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
    --runner "${RUNNER_BUILD}/multi_op_npu_runner" --log-dir "${LOGS}" --device 0 \
    --warmup "${WARMUP}" --samples "${SAMPLES}" --operator gather_elements \
    --record-target "${RECORD_TARGET}" --custom-opp-manifest "${PACKAGE_MANIFEST}"
