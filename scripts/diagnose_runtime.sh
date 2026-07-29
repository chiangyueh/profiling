#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${ROOT}/scripts/env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${ROOT}/scripts/env.sh"
fi

cd "$ROOT"

echo "Runtime diagnostic"
echo "arch=$(uname -m)"
echo "device_id=${DEVICE_ID:-0}"
echo "ASCEND_MATMUL_STRICT_ENV=${ASCEND_MATMUL_STRICT_ENV:-1}"
echo "CANN_ROOT=${CANN_ROOT:-<unset>}"
echo "CANN_PLATFORM_ROOT=${CANN_PLATFORM_ROOT:-<unset>}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<unset>}"
echo "ASCEND_LATEST_INSTALL_PATH=${ASCEND_LATEST_INSTALL_PATH:-<unset>}"
echo "ASCENDC_SOC_VERSION=${ASCENDC_SOC_VERSION:-<unset>}"
echo "DETECTED_CANN_ROOT=${DETECTED_CANN_ROOT:-<unset>}"
echo "DETECTED_ACL_SOC=${DETECTED_ACL_SOC:-<unset>}"
echo "DETECTED_NPU_SOC=${DETECTED_NPU_SOC:-<unset>}"
echo "DETECTED_NPU_SOC_SOURCE=${DETECTED_NPU_SOC_SOURCE:-<unset>}"
echo

echo "Path checks"
for path in \
    "${CANN_ROOT:-}" \
    "${CANN_PLATFORM_ROOT:-}" \
    "${CANN_PLATFORM_ROOT:-}/lib64" \
    "${CANN_PLATFORM_ROOT:-}/devlib" \
    "${CANN_ROOT:-}/runtime/lib64" \
    "${CANN_ROOT:-}/fwkacllib/lib64" \
    "${CANN_ROOT:-}/atc/lib64" \
    /usr/local/Ascend/driver/lib64 \
    /usr/local/Ascend/driver/lib64/common \
    /usr/local/Ascend/driver/lib64/driver; do
    [[ -n "$path" ]] || continue
    if [[ -e "$path" ]]; then
        printf 'exists: %s' "$path"
        if command -v realpath >/dev/null 2>&1; then
            printf ' -> %s' "$(realpath "$path" 2>/dev/null || true)"
        fi
        printf '\n'
    else
        echo "missing: $path"
    fi
done
echo

echo "libascendcl candidates"
find /usr/local/Ascend/ascend-toolkit -maxdepth 5 -name libascendcl.so -print 2>/dev/null | sort | while read -r lib; do
    if command -v realpath >/dev/null 2>&1; then
        echo "$lib -> $(realpath "$lib" 2>/dev/null || true)"
    else
        echo "$lib"
    fi
done
echo

echo "Selected LD_LIBRARY_PATH entries"
printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -E '/usr/local/Ascend|ascend-toolkit|driver' || true
echo

if [[ -x ./build/official_matmul_runner ]]; then
    echo "official_matmul_runner file"
    file ./build/official_matmul_runner || true
    echo
    echo "official_matmul_runner ldd"
    ldd ./build/official_matmul_runner 2>&1 \
        | grep -E 'ascend|driver|not found|libascendcl|libnnopbase|libopapi|libruntime|libdrv|libacl|libstdc|libgcc|libc\\.' \
        || true
else
    echo "official_matmul_runner missing or not executable"
fi
echo

if [[ -x ./build/tiling_bank_probe ]]; then
    echo "tiling_bank_probe file"
    file ./build/tiling_bank_probe || true
    echo
    echo "tiling_bank_probe ldd"
    ldd ./build/tiling_bank_probe 2>&1 \
        | grep -E 'ascend|not found|liboptiling|libstdc|libgcc|libc\\.' \
        || true
else
    echo "tiling_bank_probe missing or not executable"
fi
echo

echo "npu-smi"
if command -v npu-smi >/dev/null 2>&1; then
    command -v npu-smi
    set +e
    npu-smi info 2>&1 | sed -n '1,80p'
    smi_rc=${PIPESTATUS[0]}
    set -e
    echo "npu-smi_rc=${smi_rc}"
else
    echo "npu-smi not found"
fi
