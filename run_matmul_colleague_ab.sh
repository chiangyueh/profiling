#!/usr/bin/env bash
set -Eeuo pipefail

PHYSICAL_DEVICE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--device)
            PHYSICAL_DEVICE="${2:?missing device ID}"
            shift 2
            ;;
        *)
            exit 2
            ;;
    esac
done
[[ "$PHYSICAL_DEVICE" =~ ^[0-9]+$ ]] || exit 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$ROOT_DIR/colleague_matmul_v3"
CANN_ROOT="/usr/local/Ascend/ascend-toolkit/latest"
SOC_VERSION="Ascend910B3"
SOURCE_SIGNATURE="$(sha256sum \
    "$ROOT_DIR/run_matmul_colleague_ab.sh" \
    "$SOURCE_DIR/CMakeLists.txt" \
    "$SOURCE_DIR/cmake/npu_lib.cmake" \
    "$SOURCE_DIR/matmul_v3_launch.cpp" \
    "$SOURCE_DIR/main_v3.cpp" \
    "$SOURCE_DIR/op_kernel/mat_mul_base_kernel.h" | sha256sum | cut -c1-20)"
STATE_DIR="$ROOT_DIR/.benchmark_state/matmul_v3_colleague_ab/$SOURCE_SIGNATURE"
RESULT_DIR="$ROOT_DIR/results/matmul_v3_colleague_ab/$SOURCE_SIGNATURE"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$STATE_DIR" "$LOG_DIR"

report_unhandled_error() {
    local rc=$?
    local line="${BASH_LINENO[0]:-unknown}"
    trap - ERR
    printf '{"status":"failed","stage":"shell","return_code":%d,"line":"%s","logs":"%s"}\n' \
        "$rc" "$line" "$LOG_DIR" >&2
    exit "$rc"
}
trap report_unhandled_error ERR

printf '{"status":"begin","shape":"512x512x512","original_tiling":{"base":"16x32x96","single":"176x192x512","used_cores":9},"prediction":{"computed_elements":4608,"total_elements":262144,"error_ratio":0.982421875},"device":%s,"workdir":"%s","logs":"%s"}\n' \
    "$PHYSICAL_DEVICE" "$STATE_DIR" "$LOG_DIR"

if [[ ! -f "$CANN_ROOT/bin/setenv.bash" ]]; then
    printf '{"status":"failed","stage":"environment","reason":"CANN setenv.bash not found"}\n'
    exit 1
fi
if ! grep -q 'toolkit_running_version=.*8\.1\.RC1' "$CANN_ROOT/version.cfg"; then
    printf '{"status":"failed","stage":"environment","reason":"this reproduction requires CANN 8.1.RC1"}\n'
    exit 1
fi

# CANN's setenv scripts contain optional probes whose intermediate nonzero
# statuses are harmless but would make an errexit caller terminate silently.
printf '{"stage":"cann_environment","status":"begin"}\n'
trap - ERR
set +e
set +u
source "$CANN_ROOT/bin/setenv.bash" >>"$LOG_DIR/1.log" 2>&1
setenv_rc=$?
set -u
set -e
trap report_unhandled_error ERR
printf 'CANN_SETENV_RETURN_CODE=%d\n' "$setenv_rc" >>"$LOG_DIR/1.log"

for required_tool in cmake python3 msprof; do
    if ! command -v "$required_tool" >/dev/null 2>&1; then
        printf '{"status":"failed","stage":"environment","reason":"required tool not found","tool":"%s","logs":"%s"}\n' \
            "$required_tool" "$LOG_DIR"
        exit 1
    fi
done
printf '{"stage":"cann_environment","status":"passed","setenv_return_code":%d}\n' "$setenv_rc"

build_kernel() {
    local build_dir="$STATE_DIR/build"
    local out_dir="$STATE_DIR/out"
    local log_file="$LOG_DIR/1.log"

    if [[ ! -x "$out_dir/bin/ascendc_kernels_bbit" ]]; then
        printf '{"stage":"configure","status":"begin"}\n'
        : >"$log_file"
        if ! cmake -S "$SOURCE_DIR" -B "$build_dir" \
            -DRUN_MODE=npu \
            -DSOC_VERSION="$SOC_VERSION" \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_INSTALL_PREFIX="$out_dir" \
            -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
            -DMMV3_OP_KERNEL_DIR="$SOURCE_DIR/op_kernel" \
            -DMM_ASCEND_KERNEL_LAUNCH_ONLY=ON \
            -DMM_SHARED_SERVER_SAFE=ON >>"$log_file" 2>&1; then
            printf '{"status":"failed","stage":"configure","log":"%s"}\n' "$log_file"
            return 1
        fi
        printf '{"stage":"configure","status":"passed"}\n'
        printf '{"stage":"device_build","status":"begin","parallel":1}\n'
        if ! cmake --build "$build_dir" --parallel 1 >>"$log_file" 2>&1; then
            printf '{"status":"failed","stage":"build","log":"%s"}\n' "$log_file"
            return 1
        fi
        printf '{"stage":"device_build","status":"passed"}\n'
        printf '{"stage":"install","status":"begin"}\n'
        if ! cmake --install "$build_dir" >>"$log_file" 2>&1; then
            printf '{"status":"failed","stage":"install","log":"%s"}\n' "$log_file"
            return 1
        fi
        printf '{"stage":"install","status":"passed"}\n'
    else
        printf '{"stage":"device_build","status":"cached"}\n'
    fi
}

run_variant() {
    local variant="$1"
    local base_m="$2"
    local base_n="$3"
    local base_k="$4"
    local single_m="$5"
    local single_n="$6"
    local log_number="$7"
    local out_dir="$STATE_DIR/out"
    local run_dir="$STATE_DIR/run_$variant"
    local log_file="$LOG_DIR/$log_number.log"
    local summary

    mkdir -p "$run_dir/input" "$run_dir/output"
    cp "$STATE_DIR/input/x1_gm.bin" "$run_dir/input/x1_gm.bin"
    cp "$STATE_DIR/input/x2_gm.bin" "$run_dir/input/x2_gm.bin"
    cp "$STATE_DIR/output/golden.bin" "$run_dir/output/golden.bin"
    rm -f "$run_dir/output/output.bin"
    : >"$log_file"
    printf '{"variant":"%s","stage":"npu_run","status":"begin"}\n' "$variant"

    set +e
    (
        cd "$run_dir"
        export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
        export LD_LIBRARY_PATH="$out_dir/lib:$out_dir/lib64:$CANN_ROOT/lib64:${LD_LIBRARY_PATH:-}"
        export MM_M=512 MM_N=512 MM_K=512
        export MM_BASE_M="$base_m" MM_BASE_N="$base_n" MM_BASE_K="$base_k"
        export MM_SINGLE_M="$single_m" MM_SINGLE_N="$single_n" MM_SINGLE_K=256
        export MM_STEP_M=1 MM_STEP_N=1 MM_STEP_Ka=4 MM_STEP_Kb=4
        export MM_ITER_ORDER=0 MM_OP_TILING=0
        msprof op "$out_dir/bin/ascendc_kernels_bbit"
    ) >>"$log_file" 2>&1
    local run_rc=$?

    if (( run_rc == 0 )); then
        printf '{"variant":"%s","stage":"npu_run","status":"passed"}\n' "$variant"
    else
        printf '{"variant":"%s","stage":"npu_run","status":"failed","return_code":%d,"log":"%s"}\n' \
            "$variant" "$run_rc" "$log_file"
    fi
    printf '{"variant":"%s","stage":"numeric_verification","status":"begin"}\n' "$variant"
    python3 "$SOURCE_DIR/scripts/verify_result.py" \
        "$run_dir/output/output.bin" "$run_dir/output/golden.bin" >>"$log_file" 2>&1
    local verify_rc=$?
    summary="$(python3 "$SOURCE_DIR/scripts/summarize_result.py" \
        --variant "$variant" \
        --output "$run_dir/output/output.bin" \
        --golden "$run_dir/output/golden.bin")"
    local summary_rc=$?
    set -e

    if (( run_rc != 0 )); then
        return 1
    fi
    printf '%s\n' "$summary"
    [[ $summary_rc -eq 0 && $verify_rc -eq 0 ]]
}

build_kernel

if [[ ! -f "$STATE_DIR/output/golden.bin" ]]; then
    printf '{"stage":"input_generation","status":"begin"}\n'
    mkdir -p "$STATE_DIR/input" "$STATE_DIR/output"
    (
        cd "$STATE_DIR"
        export MM_M=512 MM_N=512 MM_K=512
        python3 "$SOURCE_DIR/scripts/gen_data.py"
    ) >>"$LOG_DIR/2.log" 2>&1
    printf '{"stage":"input_generation","status":"passed"}\n'
else
    printf '{"stage":"input_generation","status":"cached"}\n'
fi

original_passed=0
legal_passed=0
run_variant original 16 32 96 176 192 3 && original_passed=1 || true
if (( original_passed == 0 )) && [[ -f "$STATE_DIR/run_original/output/output.bin" ]]; then
    python3 "$SOURCE_DIR/scripts/validate_base_tiling.py" \
        --base-m 16 --base-n 32 --single-m 176 --single-n 192 --step-m 1 --step-n 1
fi
run_variant legal_base 128 256 64 128 256 4 && legal_passed=1 || true

if (( original_passed == 0 && legal_passed == 1 )); then
    printf '{"status":"root_cause_confirmed","cause":"BASE singleCoreM/singleCoreN exceed stepM*baseM/stepN*baseN; one Iterate/GetTensorC produces one scheduled base region per core","validator":{"status":"rejected","rules":["BASE_SINGLE_CORE_M_EXCEEDS_STEP_M_BASE_M","BASE_SINGLE_CORE_N_EXCEEDS_STEP_N_BASE_N"]}}\n'
    exit 0
fi
if (( original_passed == 1 && legal_passed == 1 )); then
    printf '{"status":"not_reproduced","cause":"both variants passed"}\n'
    exit 3
fi
printf '{"status":"inconclusive","cause":"legal BASE control did not pass","logs":"%s"}\n' "$LOG_DIR"
exit 1
