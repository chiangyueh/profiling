#!/usr/bin/env bash
set -Eeuo pipefail

# One checkout-local campaign built entirely from the pinned official CANN
# 8.1 ScatterElementsV2 source. Installed CANN files are read-only inputs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP="${OP_NPU_WARMUP:-2}"
SAMPLES="${OP_NPU_SAMPLES:-5}"
RECORD_TARGET=5000

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs only official CANN-8.1 ScatterElementsV2. The first real-NPU shape must
pass installed-reference launch, private OpAPI load, executor planning,
launch-time host-tiler audit, private precompiled-kernel launch, and exact
output equality before the 5,000-record campaign starts. Successful data are
rotating JSONL logs <=50 MiB.
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

[[ "${MODE}" == "full" ]] || { echo "fatal: only --mode full is supported" >&2; exit 2; }
for value_name in PHYSICAL_DEVICE WARMUP SAMPLES; do
    value="${!value_name}"
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "fatal: ${value_name} must be a non-negative integer" >&2; exit 2; }
done
(( SAMPLES >= 1 )) || { echo "fatal: samples must be at least 1" >&2; exit 2; }
[[ -e "/dev/davinci${PHYSICAL_DEVICE}" ]] || { echo "fatal: missing /dev/davinci${PHYSICAL_DEVICE}" >&2; exit 1; }

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
VERSION_FILE="${CANN_ROOT}/opp/version.info"
for component in compiler toolkit runtime acllib opp; do
    component_version="${CANN_ROOT}/${component}/version.info"
    [[ -f "${component_version}" ]] || { echo "fatal: missing CANN ${component} version file" >&2; exit 1; }
    grep -q '^version_dir=8\.1\.RC1$' "${component_version}" || {
        echo "fatal: CANN ${component} is not 8.1.RC1" >&2
        exit 1
    }
done
ENV_FILE=""
for candidate in "${CANN_ROOT}/set_env.sh" "$(dirname "${CANN_ROOT}")/set_env.sh"; do
    [[ -f "${candidate}" ]] && { ENV_FILE="${candidate}"; break; }
done
[[ -n "${ENV_FILE}" ]] || { echo "fatal: CANN set_env.sh is absent" >&2; exit 1; }

set +u
source "${ENV_FILE}"
set -u
export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export ASCEND_RT_VISIBLE_DEVICES="${PHYSICAL_DEVICE}"
export TILINGKEY_PAR_COMPILE=1
export OMP_NUM_THREADS=1
export CANN_OPS_BUILD_JOBS=1
unset ASCEND_CUSTOM_OPP_PATH SCATTER_ELEMENTS_SOURCE_DISPATCH \
    SCATTER_ELEMENTS_SOURCE_OPAPI_LIBRARY SCATTER_ELEMENTS_TILING_AUDIT_PATH \
    SCATTER_ELEMENTS_SOURCE_AIV_CAP SCATTER_ELEMENTS_SOURCE_UB_DIVISOR ASCENDC_CPU_DEBUG

REQUIRED=(
    source_adapter/vendor_source/cann_ops_8_1rc1.tar.gz
    source_adapter/materialize_scatter_elements_v2_cann81_source.py
    source_adapter/prepare_scatter_source_overlay.py
    source_adapter/finalize_scatter_elements_v2_package.py
    source_adapter/run_non_matmul_candidate_campaign.py
    source_adapter/scatter_elements_v2_candidate_catalog.py
    source_adapter/scatter_elements_v2_cann81_lock.json
    multi_op_bench/CMakeLists.txt
    multi_op_bench/runner.cpp
)
for relative in "${REQUIRED[@]}"; do
    [[ -f "${ROOT}/${relative}" ]] || { echo "fatal: missing repository input: ${relative}" >&2; exit 1; }
done

PACKAGE_ID="$({
    sha256sum "${ROOT}/source_adapter/vendor_source/cann_ops_8_1rc1.tar.gz" \
        "${ROOT}/source_adapter/materialize_scatter_elements_v2_cann81_source.py" \
        "${ROOT}/source_adapter/prepare_scatter_source_overlay.py" \
        "${ROOT}/source_adapter/finalize_scatter_elements_v2_package.py" \
        "${ROOT}/source_adapter/scatter_elements_v2_cann81_lock.json" "${VERSION_FILE}"
    readlink -f "${CANN_ROOT}"
} | sha256sum | cut -c1-20)"
RUN_ID="$({
    printf '%s\n' "${PACKAGE_ID}"
    sha256sum "${ROOT}/run_npu.sh" "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
        "${ROOT}/source_adapter/scatter_elements_v2_candidate_catalog.py" \
        "${ROOT}/multi_op_bench/CMakeLists.txt" "${ROOT}/multi_op_bench/runner.cpp"
} | sha256sum | cut -c1-20)"

SOURCE_ROOT="${ROOT}/.source_cache/scatter_elements_v2_cann_ops_8_1rc1"
PACKAGE_STATE="${ROOT}/.benchmark_state/scatter_elements_v2_cann81_native_v2/${PACKAGE_ID}"
PROJECT_PARENT="${PACKAGE_STATE}/project"
PROJECT="${PROJECT_PARENT}/scatter_elements_v2_source"
BUILD="${PACKAGE_STATE}/build"
OUTPUT="${PACKAGE_STATE}/output"
PACKAGE_ROOT="${OUTPUT}/packages/vendors/scatter_elements_source"
MANIFEST="${PACKAGE_STATE}/scatter_elements_v2_package.json"
RUNNER_BUILD="${ROOT}/.benchmark_state/scatter_elements_v2_cann81_native_runner_v2/${RUN_ID}"
RESULTS="${ROOT}/results/scatter_elements_v2_cann81_native_v2/${RUN_ID}"
LOGS="${RESULTS}/logs"
mkdir -p "${ROOT}/.source_cache" "${PACKAGE_STATE}" "${PROJECT_PARENT}" "${RUNNER_BUILD}" "${LOGS}"
mkdir -p "${PACKAGE_STATE}/tmp" "${PACKAGE_STATE}/cache" "${PACKAGE_STATE}/work"
export TMPDIR="${PACKAGE_STATE}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export ASCEND_CACHE_PATH="${PACKAGE_STATE}/cache"
export ASCEND_WORK_PATH="${PACKAGE_STATE}/work"

run_logged() {
    local label="$1" log="$2"
    shift 2
    echo "${label} begin"
    if "$@" >"${log}" 2>&1; then echo "${label} passed"; return 0; fi
    echo "${label} failed; inspect=${log}" >&2
    return 1
}

CPU_ALLOWED_LIST="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
BUILD_CPU="${CPU_ALLOWED_LIST%%[-,]*}"
resource_limited() {
    if command -v taskset >/dev/null 2>&1 && [[ "${BUILD_CPU}" =~ ^[0-9]+$ ]]; then
        nice -n 15 taskset -c "${BUILD_CPU}" "$@"
    else
        nice -n 15 "$@"
    fi
}

echo "SCATTER_ELEMENTS_V2_CAMPAIGN device=${PHYSICAL_DEVICE} records=${RECORD_TARGET} logs=${LOGS}"
echo "source=official_cann_ops_cann81_native installed_cann=read_only no_reset_no_kill"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
    run_logged "SOURCE_CACHE" "${PACKAGE_STATE}/source_cache.log" \
        python3 "${ROOT}/source_adapter/materialize_scatter_elements_v2_cann81_source.py" \
        --destination "${SOURCE_ROOT}"
fi
[[ -f "${SOURCE_ROOT}/.scatter_elements_v2_cann81_attestation.json" ]] || { echo "fatal: incomplete dedicated CANN-8.1 source cache" >&2; exit 2; }

if [[ ! -f "${PROJECT}/source_candidate_overlay.json" ]]; then
    [[ ! -e "${PROJECT}" ]] || { echo "fatal: incomplete private ScatterElementsV2 project: ${PROJECT}" >&2; exit 2; }
    run_logged "SCATTER_PACKAGE_PREPARE" "${PACKAGE_STATE}/prepare.log" \
        python3 "${ROOT}/source_adapter/prepare_scatter_source_overlay.py" \
        --source-root "${SOURCE_ROOT}" --output-parent "${PROJECT_PARENT}"
fi

if [[ ! -f "${MANIFEST}" ]]; then
    if [[ ! -f "${BUILD}/CMakeCache.txt" ]]; then
        run_logged "SCATTER_PACKAGE_CONFIGURE" "${PACKAGE_STATE}/configure.log" \
            resource_limited cmake -S "${PROJECT}" -B "${BUILD}" -G "Unix Makefiles" \
            -DBUILD_OPEN_PROJECT=ON -DASCEND_COMPUTE_UNIT=ascend910b \
            -DASCEND_OP_NAME=scatter_elements_v2 -DVENDOR_NAME=scatter_elements_source \
            -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" -DCHECK_COMPATIBLE=OFF \
            -DENABLE_OPS_KERNEL=ON -DENABLE_OPS_HOST=ON -DPREPARE_BUILD=OFF \
            -DENABLE_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release
    fi
    run_logged "SCATTER_PACKAGE_BUILD" "${PACKAGE_STATE}/build.log" \
        resource_limited cmake --build "${BUILD}" --target package --parallel 1
    run_logged "SCATTER_PACKAGE_INSTALL" "${PACKAGE_STATE}/install.log" cmake --install "${BUILD}"
fi
run_logged "SCATTER_PACKAGE_VALIDATE" "${PACKAGE_STATE}/validate.log" \
    python3 "${ROOT}/source_adapter/finalize_scatter_elements_v2_package.py" \
    --project "${PROJECT}" --package-root "${PACKAGE_ROOT}" --cann-root "${CANN_ROOT}" --manifest "${MANIFEST}"

if [[ ! -f "${RUNNER_BUILD}/CMakeCache.txt" ]]; then
    run_logged "SCATTER_RUNNER_CONFIGURE" "${RUNNER_BUILD}/configure.log" \
        cmake -S "${ROOT}/multi_op_bench" -B "${RUNNER_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" \
        -DSCATTER_ELEMENTS_ONLY=ON
fi
run_logged "SCATTER_RUNNER_BUILD" "${RUNNER_BUILD}/build.log" \
    resource_limited cmake --build "${RUNNER_BUILD}" --target multi_op_npu_runner --parallel 1

python3 "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
    --runner "${RUNNER_BUILD}/multi_op_npu_runner" --log-dir "${LOGS}" --device 0 \
    --warmup "${WARMUP}" --samples "${SAMPLES}" --operator scatter_elements \
    --record-target "${RECORD_TARGET}" --source-package-manifest "${MANIFEST}"
