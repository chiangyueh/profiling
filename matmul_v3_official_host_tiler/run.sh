#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SOURCE_DIR/.." && pwd)"
CANN_ROOT="/usr/local/Ascend/ascend-toolkit/latest"
RUN_MODE="npu"
SOC_VERSION="Ascend910B3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--run-mode) RUN_MODE="${2:?missing run mode}"; shift 2 ;;
        -v|--soc-version) SOC_VERSION="${2:?missing SoC version}"; shift 2 ;;
        *) exit 2 ;;
    esac
done
[[ "$RUN_MODE" == "npu" && "$SOC_VERSION" == "Ascend910B3" ]] || exit 2

signature="$(sha256sum \
    "$SOURCE_DIR/CMakeLists.txt" \
    "$SOURCE_DIR/main_v3.cpp" \
    "$SOURCE_DIR/build.sh" \
    "$SOURCE_DIR/run.sh" \
    "$SOURCE_DIR/scripts/gen_data.py" \
    "$SOURCE_DIR/scripts/verify_result.py" | sha256sum | cut -c1-20)"
STATE_DIR="$REPO_ROOT/.benchmark_state/matmul_v3_official_host_tiler/$signature"
LOG_DIR="$REPO_ROOT/results/matmul_v3_official_host_tiler/$signature/logs"
BUILD_DIR="$STATE_DIR/build"
OUT_DIR="$STATE_DIR/out"
RUN_DIR="$STATE_DIR/run"
mkdir -p "$LOG_DIR" "$RUN_DIR/input" "$RUN_DIR/output"

printf 'MATMUL_OFFICIAL_HOST_TILER_BEGIN shape=%sx%sx%s logs=%s\n' \
    "${MM_M:-512}" "${MM_N:-512}" "${MM_K:-512}" "$LOG_DIR"

: >"$LOG_DIR/1.log"
set +e
set +u
source "$CANN_ROOT/bin/setenv.bash" >>"$LOG_DIR/1.log" 2>&1
setenv_rc=$?
set -u
set -e
printf 'CANN_SETENV_RETURN_CODE=%d\n' "$setenv_rc" >>"$LOG_DIR/1.log"

printf 'MATMUL_OFFICIAL_HOST_TILER_BUILD begin\n'
MM_BUILD_DIR="$BUILD_DIR" MM_OUT_DIR="$OUT_DIR" \
    bash "$SOURCE_DIR/build.sh" >>"$LOG_DIR/1.log" 2>&1
printf 'MATMUL_OFFICIAL_HOST_TILER_BUILD passed\n'

export MM_M="${MM_M:-512}"
export MM_N="${MM_N:-512}"
export MM_K="${MM_K:-512}"

printf 'MATMUL_OFFICIAL_HOST_TILER_INPUT begin\n'
(
    cd "$RUN_DIR"
    python3 "$SOURCE_DIR/scripts/gen_data.py"
) >"$LOG_DIR/2.log" 2>&1
printf 'MATMUL_OFFICIAL_HOST_TILER_INPUT passed\n'

printf 'MATMUL_OFFICIAL_HOST_TILER_NPU begin\n'
(
    cd "$RUN_DIR"
    export LD_LIBRARY_PATH="$OUT_DIR/lib:$OUT_DIR/lib64:$CANN_ROOT/lib64:${LD_LIBRARY_PATH:-}"
    msprof op "$OUT_DIR/bin/ascendc_kernels_bbit"
    python3 "$SOURCE_DIR/scripts/verify_result.py" output/output.bin output/golden.bin
) >"$LOG_DIR/3.log" 2>&1
printf 'MATMUL_OFFICIAL_HOST_TILER_NPU passed log=%s\n' "$LOG_DIR/3.log"
