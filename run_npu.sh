#!/usr/bin/env bash
set -Eeuo pipefail

# This campaign has one scope: a checkout-local CANN-8.1 custom package for
# GatherElementsV2 with a C++ host tiler and precompiled Ascend C kernels.
# The installed CANN files remain read-only inputs; every build, package,
# temporary artifact, and rotating JSONL log stays below this checkout.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP="${OP_NPU_WARMUP:-2}"
SAMPLES="${OP_NPU_SAMPLES:-5}"
RECORD_TARGET=5000

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs GatherElements only: 5,000 output-validated real-NPU device-event
records, from 250 shapes with exactly 20 distinct successful source tilings
per admitted shape. Results are append-only JSONL files below profiling/results
and each file is capped at 50 MiB.

The script reads the installed CANN package but does not modify it, reset an
NPU, kill a process, or persistently change an environment variable. Its CANN
and compiler environment is private to this script and its child processes.
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
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "fatal: ${value_name} must be a non-negative integer" >&2; exit 2; }
done
(( SAMPLES >= 1 )) || { echo "fatal: samples must be at least 1" >&2; exit 2; }
[[ -e "/dev/davinci${PHYSICAL_DEVICE}" ]] || { echo "fatal: physical NPU device node is absent: /dev/davinci${PHYSICAL_DEVICE}" >&2; exit 1; }

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
[[ -d "${CANN_ROOT}" && -f "${CANN_ROOT}/opp/version.info" ]] || { echo "fatal: CANN root or OPP package is missing: ${CANN_ROOT}" >&2; exit 1; }
grep -q '^version_dir=8\.1\.RC1$' "${CANN_ROOT}/opp/version.info" || {
    echo "fatal: this package must be built and run with CANN 8.1.RC1: ${CANN_ROOT}/opp/version.info" >&2
    exit 1
}
ENV_FILE=""
for candidate in "${CANN_ROOT}/set_env.sh" "$(dirname "${CANN_ROOT}")/set_env.sh"; do
    [[ -f "${candidate}" ]] && { ENV_FILE="${candidate}"; break; }
done
[[ -n "${ENV_FILE}" ]] || { echo "fatal: CANN environment script is missing under ${CANN_ROOT}" >&2; exit 1; }

# This script is a child shell: source's exports end when it exits.  No login
# shell, other process, device configuration, or installed CANN file changes.
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
unset ASCEND_CUSTOM_OPP_PATH GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE \
    GATHER_ELEMENTS_TILING_AUDIT_PATH GATHER_ELEMENTS_SOURCE_DISPATCH \
    GATHER_ELEMENTS_SOURCE_AIV_CAP GATHER_ELEMENTS_SOURCE_UB_DIVISOR ASCENDC_CPU_DEBUG

for input_path in \
    "${ROOT}/source_adapter/vendor_source/cann_ops_8_1rc1.tar.gz" \
    "${ROOT}/source_adapter/vendor_source/gather_elements_v2_source.zip" \
    "${ROOT}/source_adapter/materialize_repo_source_bundle.py" \
    "${ROOT}/source_adapter/prepare_gather_elements_v2_project.py" \
    "${ROOT}/source_adapter/finalize_gather_elements_v2_package.py" \
    "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
    "${ROOT}/source_adapter/non_matmul_candidate_catalog.py" \
    "${ROOT}/multi_op_bench/CMakeLists.txt" \
    "${ROOT}/multi_op_bench/runner.cpp"; do
    [[ -f "${input_path}" ]] || { echo "fatal: required repository input is absent: ${input_path}" >&2; exit 1; }
done

SOURCE_ID="$({
    sha256sum "${ROOT}/run_npu.sh" \
        "${ROOT}/source_adapter/vendor_source/cann_ops_8_1rc1.tar.gz" \
        "${ROOT}/source_adapter/vendor_source/gather_elements_v2_source.zip" \
        "${ROOT}/source_adapter/materialize_repo_source_bundle.py" \
        "${ROOT}/source_adapter/prepare_gather_elements_v2_project.py" \
        "${ROOT}/source_adapter/finalize_gather_elements_v2_package.py" \
        "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
        "${ROOT}/source_adapter/non_matmul_candidate_catalog.py" \
        "${ROOT}/multi_op_bench/CMakeLists.txt" "${ROOT}/multi_op_bench/runner.cpp"
    sha256sum "${CANN_ROOT}/opp/version.info"
    readlink -f "${CANN_ROOT}"
} | sha256sum | cut -c1-20)"

SOURCE_CACHE_PARENT="${ROOT}/.source_cache"
CANN_OPS_SOURCE="${SOURCE_CACHE_PARENT}/cann_ops_8_1rc1"
STATE_FAMILY="${ROOT}/.benchmark_state/gather_elements_v2_cann81_prebuilt_v10"
LEGACY_STATE_FAMILY="${ROOT}/.benchmark_state/gather_elements_v2_cann81_prebuilt_v9"
STATE="${STATE_FAMILY}/${SOURCE_ID}"
PROJECT="${STATE}/project"
PACKAGE_BUILD="${STATE}/package_build"
DEFAULT_PACKAGE_ROOT="${STATE}/output/packages/vendors/gather_elements_source"
PACKAGE_ROOT="${DEFAULT_PACKAGE_ROOT}"
PACKAGE_MANIFEST="${STATE}/gather_elements_v2_private_package.json"
RUNNER_BUILD="${STATE}/runner_build"
RESULTS="${ROOT}/results/gather_elements_v2_cann81_prebuilt_v10/${SOURCE_ID}"
LOGS="${RESULTS}/logs"
mkdir -p "${SOURCE_CACHE_PARENT}" "${STATE}" "${LOGS}"
# Keep temporary files made by tools that honor the standard variables under
# the checkout as well. These exports are limited to this script and its
# children and disappear when it exits.
mkdir -p "${STATE}/tmp" "${STATE}/cache" "${STATE}/work"
export TMPDIR="${STATE}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export ASCEND_CACHE_PATH="${STATE}/cache"
export ASCEND_WORK_PATH="${STATE}/work"

run_logged() {
    local label="$1"
    local log="$2"
    shift 2
    echo "${label} begin"
    if "$@" >"${log}" 2>&1; then
        echo "${label} passed"
        return 0
    fi
    echo "${label} failed; inspect=${log}" >&2
    return 1
}

# Apply low priority and one allowed CPU to the compiler and every child it
# creates. This is process-local resource control; it does not change system
# settings or another user's processes.
CPU_ALLOWED_LIST="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
BUILD_CPU="${CPU_ALLOWED_LIST%%[-,]*}"
resource_limited() {
    local command=("$@")
    if command -v taskset >/dev/null 2>&1 && [[ "${BUILD_CPU}" =~ ^[0-9]+$ ]]; then
        nice -n 15 taskset -c "${BUILD_CPU}" "${command[@]}"
    else
        nice -n 15 "${command[@]}"
    fi
}

echo "GATHER_ELEMENTS_CAMPAIGN begin device=${PHYSICAL_DEVICE} records=${RECORD_TARGET} logs=${LOGS}"
echo "GATHER_ELEMENTS_CAMPAIGN scope=private_checkout_package installed_cann_read_only no_reset_no_kill"

if [[ ! -d "${CANN_OPS_SOURCE}" ]]; then
    run_logged "GATHER_SOURCE_CACHE" "${STATE}/source_cache.log" \
        python3 "${ROOT}/source_adapter/materialize_repo_source_bundle.py" \
        --kind cann_ops --destination "${CANN_OPS_SOURCE}"
elif [[ ! -f "${CANN_OPS_SOURCE}/.source_bundle_attestation.json" ]]; then
    echo "fatal: incomplete private source cache exists: ${CANN_OPS_SOURCE}" >&2
    exit 2
fi

if [[ ! -f "${PROJECT}/gather_elements_v2_project.json" ]]; then
    [[ ! -e "${PROJECT}" ]] || { echo "fatal: incomplete private package project exists: ${PROJECT}" >&2; exit 2; }
    run_logged "GATHER_PACKAGE_PREPARE" "${STATE}/package_prepare.log" \
        python3 "${ROOT}/source_adapter/prepare_gather_elements_v2_project.py" \
        --cann-ops-source "${CANN_OPS_SOURCE}" --cann-root "${CANN_ROOT}" --output "${PROJECT}"
fi

PACKAGE_READY=0
if [[ -f "${PACKAGE_MANIFEST}" ]]; then
    PACKAGE_ROOT="$(python3 - "${PACKAGE_MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["package_root"])
PY
)"
    run_logged "GATHER_PACKAGE_VALIDATE" "${STATE}/package_validate.log" \
        python3 "${ROOT}/source_adapter/finalize_gather_elements_v2_package.py" \
        --project "${PROJECT}" --package-root "${PACKAGE_ROOT}" \
        --cann-root "${CANN_ROOT}" --manifest "${PACKAGE_MANIFEST}"
    PACKAGE_READY=1
else
    # Runner/controller changes must not trigger another expensive Ascend-C
    # package build.  Reuse a previously validated package only when its
    # instrumented source hash, CANN 8.1 build identity, OpAPI, host tiler,
    # and four precompiled 910B kernels all match; the finalizer revalidates
    # the candidate against this freshly prepared project before use.
    mapfile -t REUSE_MANIFESTS < <(python3 - "${PROJECT}" "${CANN_ROOT}" "${STATE_FAMILY}" "${LEGACY_STATE_FAMILY}" <<'PY'
import hashlib, json, sys
from pathlib import Path

project, cann_root = map(Path, sys.argv[1:3])
families = [Path(value) for value in sys.argv[3:]]
source = project / "src/index/gather_elements_v2/op_host/gather_elements_v2_tiling.cpp"
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
cann_hash = hashlib.sha256((cann_root / "opp/version.info").read_bytes()).hexdigest()
candidates = []
for family in families:
    if not family.is_dir():
        continue
    for manifest in family.glob("*/gather_elements_v2_private_package.json"):
        try:
            row = json.loads(manifest.read_text(encoding="utf-8"))
            if (row.get("schema") != "gather_elements_v2_cann81_prebuilt_package_v2" or
                    row.get("build_cann_version") != "8.1.RC1" or
                    row.get("source_file_sha256") != source_hash or
                    row.get("cann_version_file_sha256") != cann_hash or
                    len(row.get("precompiled_device_kernels", [])) != 4):
                continue
            required = [row.get("package_root"), row.get("op_api_library"),
                        row.get("op_tiling_library"), row.get("op_proto_library")]
            required += [item.get("object") for item in row["precompiled_device_kernels"]]
            required += [item.get("metadata") for item in row["precompiled_device_kernels"]]
            if not all(value and Path(value).exists() for value in required):
                continue
            hashed = [
                (row.get("op_api_library"), row.get("op_api_library_sha256")),
                (row.get("op_tiling_library"), row.get("op_tiling_library_sha256")),
                (row.get("op_proto_library"), row.get("op_proto_library_sha256")),
                (row.get("ops_config"), row.get("ops_config_sha256")),
                (row.get("kernel_binary_info_config"), row.get("kernel_binary_info_config_sha256")),
                (row.get("kernel_operator_config"), row.get("kernel_operator_config_sha256")),
            ]
            for item in row["precompiled_device_kernels"]:
                hashed += [(item.get("object"), item.get("object_sha256")),
                           (item.get("metadata"), item.get("metadata_sha256"))]
            if not all(path and expected and Path(path).is_file() and
                       hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
                       for path, expected in hashed):
                continue
            candidates.append((manifest.stat().st_mtime_ns, manifest))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
for _, manifest in sorted(candidates, reverse=True):
    print(manifest)
PY
)
    for reusable_manifest in "${REUSE_MANIFESTS[@]}"; do
        reusable_root="$(python3 - "${reusable_manifest}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["package_root"])
PY
)"
        echo "GATHER_PACKAGE_REUSE begin"
        if python3 "${ROOT}/source_adapter/finalize_gather_elements_v2_package.py" \
                --project "${PROJECT}" --package-root "${reusable_root}" \
                --cann-root "${CANN_ROOT}" --manifest "${PACKAGE_MANIFEST}" \
                >"${STATE}/package_reuse_validate.log" 2>&1; then
            PACKAGE_ROOT="${reusable_root}"
            PACKAGE_READY=1
            echo "GATHER_PACKAGE_REUSE passed"
            break
        fi
        echo "GATHER_PACKAGE_REUSE rejected"
    done
fi

if (( PACKAGE_READY == 0 )); then
    if [[ ! -f "${PACKAGE_BUILD}/CMakeCache.txt" ]]; then
        run_logged "GATHER_PACKAGE_CONFIGURE" "${STATE}/package_configure.log" \
            resource_limited cmake -S "${PROJECT}" -B "${PACKAGE_BUILD}" -G "Unix Makefiles" \
            -DBUILD_OPEN_PROJECT=ON -DASCEND_COMPUTE_UNIT=ascend910b \
            -DASCEND_OP_NAME=gather_elements_v2 -DVENDOR_NAME=gather_elements_source \
            -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" -DCHECK_COMPATIBLE=OFF \
            -DENABLE_OPS_KERNEL=ON -DENABLE_OPS_HOST=ON -DPREPARE_BUILD=OFF -DENABLE_CCACHE=OFF \
            -DCMAKE_BUILD_TYPE=Release
    fi
    run_logged "GATHER_PACKAGE_BUILD" "${STATE}/package_build.log" \
        resource_limited cmake --build "${PACKAGE_BUILD}" --parallel 1
    run_logged "GATHER_PACKAGE_INSTALL" "${STATE}/package_install.log" \
        cmake --install "${PACKAGE_BUILD}"
    run_logged "GATHER_PACKAGE_VALIDATE" "${STATE}/package_validate.log" \
        python3 "${ROOT}/source_adapter/finalize_gather_elements_v2_package.py" \
        --project "${PROJECT}" --package-root "${DEFAULT_PACKAGE_ROOT}" \
        --cann-root "${CANN_ROOT}" --manifest "${PACKAGE_MANIFEST}"
    PACKAGE_ROOT="${DEFAULT_PACKAGE_ROOT}"
fi

if [[ ! -f "${RUNNER_BUILD}/CMakeCache.txt" ]]; then
    run_logged "GATHER_RUNNER_CONFIGURE" "${STATE}/runner_configure.log" \
        cmake -S "${ROOT}/multi_op_bench" -B "${RUNNER_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}"
fi
run_logged "GATHER_RUNNER_BUILD" "${STATE}/runner_build.log" \
    resource_limited cmake --build "${RUNNER_BUILD}" --target multi_op_npu_runner --parallel 1

python3 "${ROOT}/source_adapter/run_non_matmul_candidate_campaign.py" \
    --runner "${RUNNER_BUILD}/multi_op_npu_runner" --log-dir "${LOGS}" --device 0 \
    --warmup "${WARMUP}" --samples "${SAMPLES}" --operator gather_elements \
    --record-target "${RECORD_TARGET}" --source-package-manifest "${PACKAGE_MANIFEST}"
