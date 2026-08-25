#!/usr/bin/env bash
set -Eeuo pipefail

# Build and run one remaining CANN-8.1 operator campaign entirely inside this
# checkout. Installed CANN is a read-only compiler/runtime input.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPERATOR=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP="${OP_NPU_WARMUP:-2}"
SAMPLES="${OP_NPU_SAMPLES:-5}"
RECORD_TARGET=5000

usage() {
    cat <<'USAGE'
Usage: profiling/run_remaining_npu.sh --operator OP [-d PHYSICAL_NPU_ID]

OP is one of:
  all
  gather_elements
  flash_attention_score_grad
  fused_infer_attention_score

Each selected operator produces exactly 5,000 validated device-event latency
records in rotating JSONL logs (each <=50 MiB). "all" runs only the three
operators listed above; ScatterElements is not part of this entry. No installed
CANN file or login-shell setting is modified.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --operator) OPERATOR="${2:?missing value for --operator}"; shift 2 ;;
        -d|--device) PHYSICAL_DEVICE="${2:?missing physical NPU ID}"; shift 2 ;;
        --warmup) WARMUP="${2:?missing warmup count}"; shift 2 ;;
        --samples) SAMPLES="${2:?missing sample count}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "fatal: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${OPERATOR}" in
    all|gather_elements|flash_attention_score_grad|fused_infer_attention_score) ;;
    *) echo "fatal: --operator is required" >&2; usage >&2; exit 2 ;;
esac
for value_name in PHYSICAL_DEVICE WARMUP SAMPLES; do
    value="${!value_name}"
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "fatal: ${value_name} must be a non-negative integer" >&2; exit 2; }
done
(( SAMPLES >= 1 )) || { echo "fatal: samples must be at least 1" >&2; exit 2; }
[[ -e "/dev/davinci${PHYSICAL_DEVICE}" ]] || { echo "fatal: missing /dev/davinci${PHYSICAL_DEVICE}" >&2; exit 1; }

if [[ "${OPERATOR}" == "all" ]]; then
    for selected in gather_elements flash_attention_score_grad fused_infer_attention_score; do
        "${ROOT}/run_remaining_npu.sh" --operator "${selected}" -d "${PHYSICAL_DEVICE}" \
            --warmup "${WARMUP}" --samples "${SAMPLES}"
    done
    exit 0
fi

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
unset ASCEND_CUSTOM_OPP_PATH ASCENDC_CPU_DEBUG \
    GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE GATHER_ELEMENTS_SOURCE_DISPATCH \
    GATHER_ELEMENTS_SOURCE_OPAPI_LIBRARY GATHER_ELEMENTS_TILING_AUDIT_PATH \
    GATHER_ELEMENTS_SOURCE_AIV_CAP GATHER_ELEMENTS_SOURCE_UB_DIVISOR \
    FASG_SOURCE_DISPATCH FASG_SOURCE_OPAPI_LIBRARY FASG_SOURCE_TILING_LIBRARY FASG_TILING_AUDIT_PATH FASG_SOURCE_AIV_CAP FASG_SOURCE_L2_DIVISOR \
    FIAS_SOURCE_DISPATCH FIAS_SOURCE_OPAPI_LIBRARY FIAS_SOURCE_TILING_LIBRARY FIAS_TILING_AUDIT_PATH FIAS_SOURCE_AIV_CAP FIAS_SOURCE_UB_DIVISOR

REQUIRED=(
    source_adapter/vendor_source/cann_ops_8_1rc1.tar.gz
    source_adapter/vendor_source/cann_ops_adv_8_1rc1.tar.gz
    source_adapter/vendor_source/gather_elements_v2_source.zip
    source_adapter/remaining_operators_cann81_lock.json
    source_adapter/materialize_remaining_operators_cann81_source.py
    source_adapter/prepare_gather_elements_v2_project.py
    source_adapter/finalize_gather_elements_v2_package.py
    source_adapter/prepare_remaining_attention_cann81_overlays.py
    source_adapter/prepare_fasg_strategy_overlays.py
    source_adapter/prepare_fias_source_overlay.py
    source_adapter/finalize_remaining_cann81_package.py
    source_adapter/remaining_operator_candidate_catalog.py
    source_adapter/run_remaining_operator_campaign.py
    source_adapter/run_non_matmul_candidate_campaign.py
    multi_op_bench/CMakeLists.txt
    multi_op_bench/runner.cpp
)
for relative in "${REQUIRED[@]}"; do
    [[ -f "${ROOT}/${relative}" ]] || { echo "fatal: missing repository input: ${relative}" >&2; exit 1; }
done

PACKAGE_ID="$({
    printf '%s\n' "${OPERATOR}"
    sha256sum "${ROOT}/source_adapter/vendor_source/cann_ops_8_1rc1.tar.gz" \
        "${ROOT}/source_adapter/vendor_source/cann_ops_adv_8_1rc1.tar.gz" \
        "${ROOT}/source_adapter/vendor_source/gather_elements_v2_source.zip" \
        "${ROOT}/source_adapter/remaining_operators_cann81_lock.json" \
        "${ROOT}/source_adapter/materialize_remaining_operators_cann81_source.py" \
        "${ROOT}/source_adapter/prepare_gather_elements_v2_project.py" \
        "${ROOT}/source_adapter/finalize_gather_elements_v2_package.py" \
        "${ROOT}/source_adapter/prepare_remaining_attention_cann81_overlays.py" \
        "${ROOT}/source_adapter/prepare_fasg_strategy_overlays.py" \
        "${ROOT}/source_adapter/prepare_fias_source_overlay.py" \
        "${ROOT}/source_adapter/finalize_remaining_cann81_package.py" "${VERSION_FILE}"
    readlink -f "${CANN_ROOT}"
} | sha256sum | cut -c1-20)"
RUN_ID="$({
    printf '%s\n' "${PACKAGE_ID}"
    sha256sum "${ROOT}/run_remaining_npu.sh" \
        "${ROOT}/source_adapter/remaining_operator_candidate_catalog.py" \
        "${ROOT}/source_adapter/run_remaining_operator_campaign.py" \
        "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
        "${ROOT}/multi_op_bench/CMakeLists.txt" "${ROOT}/multi_op_bench/runner.cpp"
} | sha256sum | cut -c1-20)"

SOURCE_BUNDLE_ID="$({
    sha256sum "${ROOT}/source_adapter/vendor_source/cann_ops_8_1rc1.tar.gz" \
        "${ROOT}/source_adapter/vendor_source/cann_ops_adv_8_1rc1.tar.gz" \
        "${ROOT}/source_adapter/remaining_operators_cann81_lock.json"
} | sha256sum | cut -c1-20)"
SOURCE_PARENT="${ROOT}/.source_cache/remaining_operators_cann81/${SOURCE_BUNDLE_ID}"
BASE_SOURCE="${SOURCE_PARENT}/cann_ops"
ADV_SOURCE="${SOURCE_PARENT}/cann_ops_adv"
STATE="${ROOT}/.benchmark_state/remaining_operators_cann81/${OPERATOR}/${PACKAGE_ID}"
PROJECT_PARENT="${STATE}/projects"
RUNNER_BUILD="${ROOT}/.benchmark_state/remaining_operators_cann81_runners/${OPERATOR}/${RUN_ID}"
RESULTS="${ROOT}/results/${OPERATOR}_cann81_native/${RUN_ID}"
LOGS="${RESULTS}/logs"
mkdir -p "${SOURCE_PARENT}" "${STATE}" "${PROJECT_PARENT}" "${RUNNER_BUILD}" "${LOGS}" \
    "${STATE}/tmp" "${STATE}/cache" "${STATE}/work"
export TMPDIR="${STATE}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export ASCEND_CACHE_PATH="${STATE}/cache"
export ASCEND_WORK_PATH="${STATE}/work"

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

materialize_source() {
    local kind="$1" destination="$2" label="$3"
    if [[ ! -d "${destination}" ]]; then
        run_logged "${label}" "${STATE}/${label}.log" \
            python3 "${ROOT}/source_adapter/materialize_remaining_operators_cann81_source.py" \
            --kind "${kind}" --destination "${destination}"
    fi
    [[ -f "${destination}/.remaining_operators_cann81_attestation.json" ]] || {
        echo "fatal: incomplete private ${kind} source cache" >&2
        exit 2
    }
}

configure_project() {
    local project="$1" build="$2" output="$3" cmake_op="$4" vendor="$5" log_prefix="$6"
    if [[ ! -f "${build}/CMakeCache.txt" ]]; then
        run_logged "${log_prefix}_CONFIGURE" "${build}.configure.log" \
            resource_limited cmake -S "${project}" -B "${build}" -G "Unix Makefiles" \
            -DBUILD_OPEN_PROJECT=ON -DASCEND_COMPUTE_UNIT=ascend910b \
            -DASCEND_OP_NAME="${cmake_op}" -DVENDOR_NAME="${vendor}" \
            -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" -DCHECK_COMPATIBLE=OFF \
            -DENABLE_OPS_KERNEL=ON -DENABLE_OPS_HOST=ON -DPREPARE_BUILD=OFF \
            -DENABLE_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${output}"
    fi
}

build_full_package() {
    local project="$1" build="$2" output="$3" cmake_op="$4" vendor="$5" log_prefix="$6"
    configure_project "${project}" "${build}" "${output}" "${cmake_op}" "${vendor}" "${log_prefix}"
    local package_root="${output}/packages/vendors/${vendor}"
    local kernel_root="${package_root}/op_impl/ai_core/tbe/kernel/ascend910b/${cmake_op}"
    local tiling_root="${package_root}/op_impl/ai_core/tbe/op_tiling/lib/linux"
    if [[ ! -f "${package_root}/op_api/lib/libcust_opapi.so" ]] || \
       ! find "${kernel_root}" -maxdepth 1 -name '*.o' -type f -print -quit 2>/dev/null | grep -q . || \
       ! find "${tiling_root}" -name libcust_opmaster_rt2.0.so -type f -print -quit 2>/dev/null | grep -q .; then
        run_logged "${log_prefix}_BUILD" "${build}.build.log" \
            resource_limited cmake --build "${build}" --target package --parallel 1
        run_logged "${log_prefix}_INSTALL" "${build}.install.log" cmake --install "${build}"
    fi
}

build_attention_package() {
    local project="$1" build="$2" output="$3" cmake_op="$4" vendor="$5" log_prefix="$6"
    configure_project "${project}" "${build}" "${output}" "${cmake_op}" "${vendor}" "${log_prefix}"
    if [[ ! -f "${build}/libcust_opmaster_rt2.0.so" ]]; then
        # The NPU server already has the matching official CANN 8.1 device
        # kernels. Build only the instrumented CANN 8.1 host tiler; compiling
        # the public package target would rebuild hundreds of unrelated device
        # objects without changing the measured kernel.
        run_logged "${log_prefix}_BUILD" "${build}.build.log" \
            resource_limited cmake --build "${build}" --target optiling --parallel 1
    fi
    [[ -f "${build}/libcust_opmaster_rt2.0.so" ]] || {
        echo "fatal: ${log_prefix} produced no CANN 8.1 host-tiler library" >&2
        exit 2
    }
}

MANIFESTS=()
materialize_source cann_ops "${BASE_SOURCE}" SOURCE_BASE_CACHE

if [[ "${OPERATOR}" == "gather_elements" ]]; then
    PROJECT="${PROJECT_PARENT}/gather_elements_v2_source"
    if [[ ! -f "${PROJECT}/gather_elements_v2_project.json" ]]; then
        [[ ! -e "${PROJECT}" ]] || { echo "fatal: incomplete GatherElementsV2 project" >&2; exit 2; }
        run_logged GATHER_PREPARE "${STATE}/prepare.log" \
            python3 "${ROOT}/source_adapter/prepare_gather_elements_v2_project.py" \
            --cann-ops-source "${BASE_SOURCE}" --cann-root "${CANN_ROOT}" --output "${PROJECT}"
    fi
    BUILD="${STATE}/build"
    OUTPUT="${STATE}/output"
    VENDOR=gather_elements_source
    PACKAGE_ROOT="${OUTPUT}/packages/vendors/${VENDOR}"
    MANIFEST="${STATE}/package.json"
    build_full_package "${PROJECT}" "${BUILD}" "${OUTPUT}" gather_elements_v2 "${VENDOR}" GATHER_PACKAGE
    run_logged GATHER_VALIDATE "${STATE}/validate.log" \
        python3 "${ROOT}/source_adapter/finalize_gather_elements_v2_package.py" \
        --project "${PROJECT}" --package-root "${PACKAGE_ROOT}" \
        --cann-root "${CANN_ROOT}" --manifest "${MANIFEST}"
    MANIFESTS+=("${MANIFEST}")
else
    materialize_source cann_ops_adv "${ADV_SOURCE}" SOURCE_ADV_CACHE
    if [[ "${OPERATOR}" == "fused_infer_attention_score" ]]; then
        PROJECT="${PROJECT_PARENT}/fias_source_dispatch"
        if [[ ! -f "${PROJECT}/source_candidate_overlay.json" ]]; then
            [[ ! -e "${PROJECT}" ]] || { echo "fatal: incomplete FIAS project" >&2; exit 2; }
            run_logged FIAS_PREPARE "${STATE}/prepare.log" \
                python3 "${ROOT}/source_adapter/prepare_remaining_attention_cann81_overlays.py" \
                --operator "${OPERATOR}" --source-root "${ADV_SOURCE}" \
                --harness-root "${BASE_SOURCE}" --output-parent "${PROJECT_PARENT}"
        fi
        BUILD="${STATE}/build"
        OUTPUT="${STATE}/output"
        VENDOR=fias_source
        MANIFEST="${STATE}/package.json"
        build_attention_package "${PROJECT}" "${BUILD}" "${OUTPUT}" fused_infer_attention_score "${VENDOR}" \
            FIAS_HOST_TILER
        run_logged FIAS_VALIDATE "${STATE}/validate.log" \
            python3 "${ROOT}/source_adapter/finalize_remaining_cann81_package.py" \
            --operator "${OPERATOR}" --project "${PROJECT}" --build-root "${BUILD}" \
            --cann-root "${CANN_ROOT}" --manifest "${MANIFEST}"
        MANIFESTS+=("${MANIFEST}")
    else
        FIRST_MANIFEST="${PROJECT_PARENT}/fasg_flashattentionscoregradtilings1s2bn2gs1s2/source_candidate_overlay.json"
        if [[ ! -f "${FIRST_MANIFEST}" ]]; then
            if find "${PROJECT_PARENT}" -mindepth 1 -maxdepth 1 -name 'fasg_*' -print -quit | grep -q .; then
                echo "fatal: incomplete FASG project set" >&2
                exit 2
            fi
            run_logged FASG_PREPARE "${STATE}/prepare.log" \
                python3 "${ROOT}/source_adapter/prepare_remaining_attention_cann81_overlays.py" \
                --operator "${OPERATOR}" --source-root "${ADV_SOURCE}" \
                --harness-root "${BASE_SOURCE}" --output-parent "${PROJECT_PARENT}"
        fi
        mapfile -t FASG_PROJECTS < <(find "${PROJECT_PARENT}" -mindepth 2 -maxdepth 2 \
            -name source_candidate_overlay.json -path '*/fasg_*/*' -printf '%h\n' | sort -u)
        (( ${#FASG_PROJECTS[@]} == 8 )) || { echo "fatal: FASG requires exactly eight source projects" >&2; exit 2; }
        BASE_VENDOR=fasg_source
        for project in "${FASG_PROJECTS[@]}"; do
            slug="$(basename "${project}")"
            variant_build="${STATE}/build_${slug}"
            variant_output="${STATE}/unused_${slug}"
            manifest="${STATE}/manifests/${slug}.json"
            build_attention_package "${project}" "${variant_build}" "${variant_output}" \
                flash_attention_score_grad "${BASE_VENDOR}" "FASG_HOST_${slug}"
            run_logged "FASG_VALIDATE_${slug}" "${STATE}/validate_${slug}.log" \
                python3 "${ROOT}/source_adapter/finalize_remaining_cann81_package.py" \
                --operator "${OPERATOR}" --project "${project}" --build-root "${variant_build}" \
                --cann-root "${CANN_ROOT}" --manifest "${manifest}"
            MANIFESTS+=("${manifest}")
        done
    fi
fi

if [[ ! -f "${RUNNER_BUILD}/CMakeCache.txt" ]]; then
    run_logged RUNNER_CONFIGURE "${RUNNER_BUILD}/configure.log" \
        cmake -S "${ROOT}/multi_op_bench" -B "${RUNNER_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" \
        -DREMAINING_OPERATOR="${OPERATOR}"
fi
run_logged RUNNER_BUILD "${RUNNER_BUILD}/build.log" \
    resource_limited cmake --build "${RUNNER_BUILD}" --target multi_op_npu_runner --parallel 1

CAMPAIGN_ARGS=()
for manifest in "${MANIFESTS[@]}"; do
    CAMPAIGN_ARGS+=(--source-package-manifest "${manifest}")
done
echo "CAMPAIGN_READY operator=${OPERATOR} records=${RECORD_TARGET} logs=${LOGS} device=${PHYSICAL_DEVICE}"
python3 "${ROOT}/source_adapter/run_remaining_operator_campaign.py" \
    --runner "${RUNNER_BUILD}/multi_op_npu_runner" --log-dir "${LOGS}" --device 0 \
    --warmup "${WARMUP}" --samples "${SAMPLES}" --operator "${OPERATOR}" \
    --record-target "${RECORD_TARGET}" "${CAMPAIGN_ARGS[@]}"
