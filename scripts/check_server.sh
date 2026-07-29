#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

LOG_DIR="$ROOT/results/logs"
WORK_DIR="$ROOT/build/server_check"
mkdir -p "$LOG_DIR" "$WORK_DIR"
LOG_FILE="$LOG_DIR/server_check_$(date +%Y%m%d_%H%M%S).log"
SERVER_CHECK_VERBOSE="${SERVER_CHECK_VERBOSE:-0}"
if [[ "$SERVER_CHECK_VERBOSE" == "1" ]]; then
    exec > >(tee "$LOG_FILE") 2>&1
else
    exec 3>&1
    exec >"$LOG_FILE" 2>&1
fi

run_cmd() {
    echo
    echo "+ $*"
    set +e
    "$@" 2>&1 | sed -n '1,180p'
    local rc=${PIPESTATUS[0]}
    set -e
    echo "rc=${rc}"
}

CAPTURED_OUTPUT=""
CAPTURED_RC=0
run_capture() {
    echo
    echo "+ $*"
    set +e
    CAPTURED_OUTPUT="$("$@" 2>&1)"
    CAPTURED_RC=$?
    set -e
    printf '%s\n' "$CAPTURED_OUTPUT" | sed -n '1,180p'
    echo "rc=${CAPTURED_RC}"
}

output_has_acl_507008() {
    grep -Eq 'aclInit rc=507008|aclInit failed, rc=507008|rc=507008' <<<"${1:-}"
}

output_has_acl_success() {
    grep -Eq 'aclInit rc=0|aclrtGetSocName=' <<<"${1:-}"
}

print_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        echo
        echo "== $path =="
        sed -n '1,120p' "$path"
    fi
}

echo "Ascend MatMul server check"
echo "log=$LOG_FILE"
echo "date=$(date -Is)"
echo "pwd=$PWD"
echo "user=$(id)"
echo "uname=$(uname -a)"
echo "arch=$(uname -m)"
echo "device_id=${DEVICE_ID:-0}"
echo
grep -E '^RUN_NPU_VERSION=' "$ROOT/run_npu.sh" || true
echo "CANN_ROOT=${CANN_ROOT:-<unset>}"
echo "CANN_PLATFORM_ROOT=${CANN_PLATFORM_ROOT:-<unset>}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<unset>}"
echo "ASCENDC_SOC_VERSION=${ASCENDC_SOC_VERSION:-<unset>}"
echo "DETECTED_NPU_SOC=${DETECTED_NPU_SOC:-<unset>}"
if command -v readlink >/dev/null 2>&1; then
    echo "CANN_ROOT_REAL=$(readlink -f "${CANN_ROOT:-/nonexistent}" 2>/dev/null || true)"
fi

print_file "/usr/local/Ascend/ascend-toolkit/latest/version.cfg"
print_file "/usr/local/Ascend/ascend-toolkit/latest/runtime/version.info"
print_file "/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"
print_file "/usr/local/Ascend/ascend-toolkit/latest/opp/version.info"
print_file "/usr/local/Ascend/driver/version.info"
print_file "/usr/local/Ascend/driver/driver_version.info"
print_file "/etc/ascend_install.info"

echo
echo "== selected LD_LIBRARY_PATH =="
printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -E '/usr/local/Ascend|ascend-toolkit|driver' || true

run_cmd bash -lc 'ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc 2>&1'

if command -v npu-smi >/dev/null 2>&1; then
    run_cmd npu-smi info
    run_cmd npu-smi info -t board
    run_cmd npu-smi info -t product
else
    echo
    echo "npu-smi not found in PATH"
    if [[ -x /usr/local/Ascend/driver/tools/npu-smi ]]; then
        run_cmd /usr/local/Ascend/driver/tools/npu-smi info
        run_cmd /usr/local/Ascend/driver/tools/npu-smi info -t board
        run_cmd /usr/local/Ascend/driver/tools/npu-smi info -t product
    fi
fi

ACL_MIN_CPP="$WORK_DIR/acl_min.cpp"
ACL_MIN_BIN="$WORK_DIR/acl_min"
cat >"$ACL_MIN_CPP" <<'CPP'
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdlib>
#include <iostream>

int main() {
    const char *dev = std::getenv("DEVICE_ID");
    const int deviceId = dev == nullptr ? 0 : std::atoi(dev);
    std::cout << "minimal acl check" << std::endl;
    std::cout << "device=" << deviceId << std::endl;
    aclError rc = aclInit(nullptr);
    std::cout << "aclInit rc=" << rc << std::endl;
    if (rc != ACL_SUCCESS) return static_cast<int>(rc);
    rc = aclrtSetDevice(deviceId);
    std::cout << "aclrtSetDevice rc=" << rc << std::endl;
    if (rc != ACL_SUCCESS) {
        aclFinalize();
        return static_cast<int>(rc);
    }
    const char *soc = aclrtGetSocName();
    std::cout << "aclrtGetSocName=" << (soc == nullptr ? "<null>" : soc) << std::endl;
    rc = aclrtResetDevice(deviceId);
    std::cout << "aclrtResetDevice rc=" << rc << std::endl;
    rc = aclFinalize();
    std::cout << "aclFinalize rc=" << rc << std::endl;
    return 0;
}
CPP

echo
echo "== build minimal ACL probe =="
set +e
g++ -std=c++17 -O2 "$ACL_MIN_CPP" \
    -I"$CANN_PLATFORM_ROOT/include" \
    -L"$CANN_PLATFORM_ROOT/lib64" \
    -L"$CANN_PLATFORM_ROOT/devlib" \
    -Wl,-rpath,"$CANN_PLATFORM_ROOT/lib64" \
    -Wl,-rpath-link,"$CANN_PLATFORM_ROOT/devlib" \
    -Wl,--allow-shlib-undefined \
    -lascendcl \
    -o "$ACL_MIN_BIN"
compile_rc=$?
set -e
echo "compile_rc=${compile_rc}"
MINIMAL_ACL_OUTPUT="compile failed"
MINIMAL_ACL_RC="$compile_rc"
if [[ "$compile_rc" -eq 0 ]]; then
    run_cmd ldd "$ACL_MIN_BIN"
    run_capture env DEVICE_ID="${DEVICE_ID:-0}" "$ACL_MIN_BIN"
    MINIMAL_ACL_OUTPUT="$CAPTURED_OUTPUT"
    MINIMAL_ACL_RC="$CAPTURED_RC"
fi

if output_has_acl_success "$MINIMAL_ACL_OUTPUT"; then
    echo
    echo "minimal ACL passed, building official MatMulV3 runner for linked ACL check"
    DETECT_ENV="$LOG_DIR/server_check_detected_soc_$(date +%Y%m%d_%H%M%S).env"
    run_capture "$ROOT/scripts/detect_soc.sh" "$DETECT_ENV"
    if [[ "$CAPTURED_RC" -eq 0 && -f "$DETECT_ENV" ]]; then
        # shellcheck disable=SC1090
        source "$DETECT_ENV"
        export ASCENDC_SOC_VERSION SOC_VERSION DETECTED_NPU_SOC DETECTED_NPU_SOC_SOURCE DETECTED_NPU_SOC_RAW
        run_capture "$ROOT/scripts/build_all.sh"
    else
        echo "runner build skipped because SoC detection failed"
    fi
fi

RUNNER_ACL_OUTPUT="runner missing"
RUNNER_ACL_RC="missing"
if [[ -x "$ROOT/build/official_matmul_runner" ]]; then
    echo
    echo "== official_matmul_runner =="
    run_cmd file "$ROOT/build/official_matmul_runner"
    run_cmd ldd "$ROOT/build/official_matmul_runner"
    run_capture env DEVICE_ID="${DEVICE_ID:-0}" \
        "$ROOT/build/official_matmul_runner" --acl-only --device "${DEVICE_ID:-0}"
    RUNNER_ACL_OUTPUT="$CAPTURED_OUTPUT"
    RUNNER_ACL_RC="$CAPTURED_RC"
else
    echo
    echo "build/official_matmul_runner missing; run ./scripts/build_all.sh first if needed"
fi

echo
SUMMARY="server_check_summary"$'\n'
SUMMARY+="log=${LOG_FILE}"$'\n'
SUMMARY+="minimal_acl_shell_rc=${MINIMAL_ACL_RC}"$'\n'
SUMMARY+="runner_acl_shell_rc=${RUNNER_ACL_RC}"$'\n'
if output_has_acl_507008 "$MINIMAL_ACL_OUTPUT"; then
    SUMMARY+="minimal_acl=aclInit_507008"$'\n'
else
    if output_has_acl_success "$MINIMAL_ACL_OUTPUT"; then
        SUMMARY+="minimal_acl=ok"$'\n'
    else
        SUMMARY+="minimal_acl=not_ok_or_not_run"$'\n'
    fi
fi
if output_has_acl_507008 "$RUNNER_ACL_OUTPUT"; then
    SUMMARY+="runner_acl=aclInit_507008"$'\n'
else
    if output_has_acl_success "$RUNNER_ACL_OUTPUT"; then
        SUMMARY+="runner_acl=ok"$'\n'
    else
        SUMMARY+="runner_acl=not_ok_or_not_run"$'\n'
    fi
fi

if output_has_acl_507008 "$MINIMAL_ACL_OUTPUT"; then
    SUMMARY+="classification=server_runtime_driver_soc_init_failure"$'\n'
    SUMMARY+="next_step=check CANN/runtime/driver/firmware/container device mapping; MatMul code has not run yet"$'\n'
elif output_has_acl_success "$MINIMAL_ACL_OUTPUT" && output_has_acl_507008 "$RUNNER_ACL_OUTPUT"; then
    SUMMARY+="classification=linked_official_runner_runtime_failure"$'\n'
    SUMMARY+="next_step=inspect official runner linked CANN runtime libraries"$'\n'
elif output_has_acl_success "$MINIMAL_ACL_OUTPUT" && output_has_acl_success "$RUNNER_ACL_OUTPUT"; then
    SUMMARY+="classification=acl_init_ok"$'\n'
    SUMMARY+="next_step=rerun ./run_npu.sh --mode smoke and inspect MatMulV3 bank/profile result"$'\n'
else
    SUMMARY+="classification=inconclusive"$'\n'
    SUMMARY+="next_step=inspect this full server_check log"$'\n'
fi
printf '%s' "$SUMMARY"
if [[ "$SERVER_CHECK_VERBOSE" != "1" ]]; then
    printf '%s' "$SUMMARY" >&3
fi
echo
echo "server_check_log=$LOG_FILE"
