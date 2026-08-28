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
OFFICIAL_SOURCE_DIR="$ROOT_DIR/src/matmul/mat_mul_v3"
CANN_ROOT="/usr/local/Ascend/ascend-toolkit/latest"
SOC_VERSION="Ascend910B3"
SOURCE_SIGNATURE="$(sha256sum \
    "$ROOT_DIR/run_matmul_colleague_ab.sh" \
    "$ROOT_DIR/build.sh" \
    "$OFFICIAL_SOURCE_DIR/CMakeLists.txt" \
    "$OFFICIAL_SOURCE_DIR/op_host/mat_mul_v3_tiling.cpp" \
    "$OFFICIAL_SOURCE_DIR/op_host/mat_mul_v3_base_tiling.cpp" \
    "$OFFICIAL_SOURCE_DIR/op_host/mat_mul_v3_l2_cache.cpp" \
    "$OFFICIAL_SOURCE_DIR/op_host/aclnn_matmul.cpp" \
    "$OFFICIAL_SOURCE_DIR/op_kernel/mat_mul_v3.cpp" \
    "$SOURCE_DIR/CMakeLists.txt" \
    "$SOURCE_DIR/cmake/npu_lib.cmake" \
    "$SOURCE_DIR/matmul_v3_launch.cpp" \
    "$SOURCE_DIR/main_v3.cpp" \
    "$SOURCE_DIR/op_kernel/mat_mul_base_kernel.h" | sha256sum | cut -c1-20)"
STATE_DIR="$ROOT_DIR/.benchmark_state/matmul_v3_colleague_ab/$SOURCE_SIGNATURE"
RESULT_DIR="$ROOT_DIR/results/matmul_v3_colleague_ab/$SOURCE_SIGNATURE"
LOG_DIR="$RESULT_DIR/logs"
PRIVATE_OPP_DIR="$STATE_DIR/private_opp"
# +++ BEGIN: direct package extraction keeps the exact packaged vendor tree
# below this run's private state; no installer writes to a system OPP path.
PRIVATE_VENDOR_ROOT="$PRIVATE_OPP_DIR/packages/vendors/customize"
# +++ END: run-local packaged vendor root.

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

printf '{"status":"begin","shape":"512x512x512","log_3":"unmodified_gitee_MatMulV3_host_tiler_and_numeric_verification","log_4":"colleague_hand_filled_tiling_comparison","device":%s,"workdir":"%s","logs":"%s"}\n' \
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
: >"$LOG_DIR/1.log"
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

# +++ BEGIN: build and privately install the complete unmodified Gitee MatMulV3 package.
build_official_package() {
    local log_file="$LOG_DIR/1.log"
    local package_file

    if [[ -f "$PRIVATE_VENDOR_ROOT/op_api/lib/libcust_opapi.so" ]] &&
       find "$PRIVATE_VENDOR_ROOT/op_impl/ai_core/tbe/op_tiling" -type f -name 'libcust_opmaster_rt2.0.so' -print -quit |
           grep -q .; then
        printf '{"stage":"official_operator_package","status":"cached"}\n'
        return 0
    fi

    printf '{"stage":"official_operator_package","status":"begin","source":"%s","parallel":1}\n' \
        "$OFFICIAL_SOURCE_DIR"
    # Host-only packaging retains the official dynamic kernel source.  The NPU
    # compiler therefore builds only the one key actually selected for this
    # 512x512x512 launch instead of prebuilding every dtype/key combination.
    if ! CANN_OPS_BUILD_JOBS=1 bash "$ROOT_DIR/build.sh" \
        -n mat_mul_v3 -c ascend910b -p "$CANN_ROOT" -b host >>"$log_file" 2>&1; then
        printf '{"stage":"official_operator_package","status":"failed","log":"%s"}\n' "$log_file"
        return 1
    fi

    package_file="$(find "$ROOT_DIR/output" -maxdepth 1 -type f -name 'CANN-custom_ops-*.run' -print -quit)"
    if [[ -z "$package_file" ]]; then
        printf '{"stage":"official_operator_package","status":"failed","reason":"package installer missing","log":"%s"}\n' \
            "$log_file"
        return 1
    fi
    mkdir -p "$PRIVATE_OPP_DIR"
    # --- BEGIN: package installer route retained for reference.
    # "$package_file" --quiet --install-path="$PRIVATE_OPP_DIR"
    # --- END: package installer route retained for reference.

    # +++ BEGIN: extract the exact package payload without running install.sh.
    # The archive is confined to PRIVATE_OPP_DIR, and ASCEND_CUSTOM_OPP_PATH is
    # set only in the child process that produces 3.log.
    if ! "$package_file" --tar xf -C "$PRIVATE_OPP_DIR" >>"$log_file" 2>&1; then
        printf '{"stage":"official_operator_package","status":"failed","reason":"private package extraction failed","log":"%s"}\n' \
            "$log_file"
        return 1
    fi
    # +++ END: exact run-local package extraction.
    if [[ ! -f "$PRIVATE_VENDOR_ROOT/op_api/lib/libcust_opapi.so" ]]; then
        printf '{"stage":"official_operator_package","status":"failed","reason":"private cust_opapi missing","log":"%s"}\n' \
            "$log_file"
        return 1
    fi
    printf '{"stage":"official_operator_package","status":"passed","private_vendor":"%s"}\n' \
        "$PRIVATE_VENDOR_ROOT"
}
# +++ END: private unmodified Gitee MatMulV3 package.

build_runners() {
    local build_dir="$STATE_DIR/build"
    local out_dir="$STATE_DIR/out"
    local log_file="$LOG_DIR/1.log"

    build_official_package
    if [[ ! -x "$out_dir/bin/ascendc_kernels_bbit" ||
          ! -x "$out_dir/bin/ascendc_kernels_bbit_official" ]]; then
        printf '{"stage":"configure","status":"begin"}\n'
        if ! cmake -S "$SOURCE_DIR" -B "$build_dir" \
            -DRUN_MODE=npu \
            -DSOC_VERSION="$SOC_VERSION" \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_INSTALL_PREFIX="$out_dir" \
            -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
            -DMMV3_OP_KERNEL_DIR="$SOURCE_DIR/op_kernel" \
            -DMM_LOCAL_VENDOR_ROOT="$PRIVATE_VENDOR_ROOT" \
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

# +++ BEGIN: run the unmodified Gitee operator route and write its result to 3.log.
run_official_variant() {
    local out_dir="$STATE_DIR/out"
    local run_dir="$STATE_DIR/run_official_gitee"
    local log_file="$LOG_DIR/3.log"
    local tiling_lib_dir
    local summary

    tiling_lib_dir="$(find "$PRIVATE_VENDOR_ROOT/op_impl/ai_core/tbe/op_tiling/lib/linux" \
        -mindepth 1 -maxdepth 1 -type d -print -quit)"
    mkdir -p "$run_dir/input" "$run_dir/output"
    cp "$STATE_DIR/input/x1_gm.bin" "$run_dir/input/x1_gm.bin"
    cp "$STATE_DIR/input/x2_gm.bin" "$run_dir/input/x2_gm.bin"
    cp "$STATE_DIR/output/golden_fp16.bin" "$run_dir/output/golden.bin"
    rm -f "$run_dir/output/output.bin"
    : >"$log_file"
    printf '{"variant":"official_gitee_host_tiler","stage":"npu_run","status":"begin","log":"3.log"}\n'

    set +e
    (
        cd "$run_dir"
        export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
        export ASCEND_CUSTOM_OPP_PATH="$PRIVATE_VENDOR_ROOT${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
        export LD_LIBRARY_PATH="$PRIVATE_VENDOR_ROOT/op_api/lib:$tiling_lib_dir:$out_dir/lib:$out_dir/lib64:$CANN_ROOT/lib64:${LD_LIBRARY_PATH:-}"
        export MM_M=512 MM_N=512 MM_K=512
        msprof op "$out_dir/bin/ascendc_kernels_bbit_official"
    ) >>"$log_file" 2>&1
    local run_rc=$?

    if (( run_rc != 0 )); then
        set -e
        printf '{"variant":"official_gitee_host_tiler","stage":"npu_run","status":"failed","return_code":%d,"log":"%s"}\n' \
            "$run_rc" "$log_file"
        return 1
    fi
    printf '{"variant":"official_gitee_host_tiler","stage":"npu_run","status":"passed"}\n'
    printf '{"variant":"official_gitee_host_tiler","stage":"numeric_verification","status":"begin"}\n'
    python3 "$OFFICIAL_SOURCE_DIR/examples/AclNNInvocationNaive/verify_result.py" \
        "$run_dir/output/output.bin" "$run_dir/output/golden.bin" >>"$log_file" 2>&1
    summary="$(python3 "$SOURCE_DIR/scripts/summarize_result.py" \
        --variant official_gitee_host_tiler \
        --dtype fp16 \
        --output "$run_dir/output/output.bin" \
        --golden "$run_dir/output/golden.bin")"
    local summary_rc=$?
    printf '%s\n' "$summary" >>"$log_file"
    set -e
    printf '%s\n' "$summary"
    return "$summary_rc"
}
# +++ END: official route recorded in 3.log.

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

build_runners

if [[ ! -f "$STATE_DIR/output/golden.bin" || ! -f "$STATE_DIR/output/golden_fp16.bin" ]]; then
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

official_passed=0
colleague_passed=0
run_official_variant && official_passed=1 || true

# --- BEGIN: retained colleague hand-filled tiling path, now isolated in 4.log.
run_variant colleague_hand_filled 16 32 96 176 192 4 && colleague_passed=1 || true
if (( colleague_passed == 0 )) && [[ -f "$STATE_DIR/run_colleague_hand_filled/output/output.bin" ]]; then
    python3 "$SOURCE_DIR/scripts/validate_base_tiling.py" \
        --base-m 16 --base-n 32 --single-m 176 --single-n 192 --step-m 1 --step-n 1 >>"$LOG_DIR/4.log" 2>&1
fi
# --- END: retained colleague hand-filled tiling comparison.

if (( official_passed == 1 )); then
    printf '{"status":"completed","official_gitee_numeric_result":"passed","official_log":"%s","colleague_hand_filled_result":"%s","comparison_log":"%s"}\n' \
        "$LOG_DIR/3.log" "$([[ $colleague_passed -eq 1 ]] && printf passed || printf wrong_output)" "$LOG_DIR/4.log"
    exit 0
fi
printf '{"status":"failed","cause":"unmodified Gitee MatMulV3 did not pass numeric verification","official_log":"%s","comparison_log":"%s"}\n' \
    "$LOG_DIR/3.log" "$LOG_DIR/4.log"
exit 1
