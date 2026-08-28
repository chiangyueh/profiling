#!/usr/bin/env bash
set -Eeuo pipefail

MODE=""
PHYSICAL_DEVICE="2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?missing value for --mode}"
            shift 2
            ;;
        -d|--device)
            PHYSICAL_DEVICE="${2:?missing device ID}"
            shift 2
            ;;
        *)
            exit 2
            ;;
    esac
done

[[ "$MODE" == "full" && "$PHYSICAL_DEVICE" =~ ^[0-9]+$ ]] || exit 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_DIR="$ROOT_DIR/colleague_matmul_v3"
FIXED_DIR="$ROOT_DIR/colleague_matmul_v3_minimal_fix"
SOURCE_SIGNATURE="$({
    find "$FIXED_DIR" -type f \
        ! -path '*/build/*' ! -path '*/out/*' ! -path '*/input/*' ! -path '*/output/*' \
        ! -name ascendc_kernels_bbit -print0
} | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-20)"
LOG_DIR="$ROOT_DIR/results/colleague_matmul_v3_minimal_fix/$SOURCE_SIGNATURE/logs"
STATE_DIR="$ROOT_DIR/.benchmark_state/colleague_matmul_v3_minimal_fix/$SOURCE_SIGNATURE"
mkdir -p "$LOG_DIR" "$STATE_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE"

printf 'MATMUL_COLLEAGUE_MINIMAL_FIX_BEGIN shape=512x512x512 device=%s\n' "$PHYSICAL_DEVICE"

if [[ ! -f "$STATE_DIR/build_complete" || ! -x "$FIXED_DIR/ascendc_kernels_bbit" ]]; then
    printf 'MATMUL_BUILD begin\n'
    bash "$FIXED_DIR/build.sh" -r npu -v Ascend910B3 >"$LOG_DIR/1.log" 2>&1
    touch "$STATE_DIR/build_complete"
    printf 'MATMUL_BUILD passed\n'
else
    printf 'MATMUL_BUILD cached\n'
fi

{
    printf 'original_directory=%s\n' "$ORIGINAL_DIR"
    printf 'modified_directory=%s\n' "$FIXED_DIR"
    printf 'functional_change=MM_SINGLE_M:176->16,MM_SINGLE_N:192->32\n'
    printf 'build_safety_change=cmake_parallel:unbounded->1\n'
} >"$LOG_DIR/2.log"

printf 'MATMUL_NPU_AND_VERIFY begin\n'
set +e
MM_M=512 MM_N=512 MM_K=512 \
MM_BASE_M=16 MM_BASE_N=32 MM_BASE_K=96 \
MM_SINGLE_M=16 MM_SINGLE_N=32 MM_SINGLE_K=256 \
MM_STEP_M=1 MM_STEP_N=1 MM_STEP_Ka=4 MM_STEP_Kb=4 \
MM_ITER_ORDER=0 MM_OP_TILING=0 \
    bash "$FIXED_DIR/run.sh" -r npu -v Ascend910B3 >"$LOG_DIR/3.log" 2>&1
run_rc=$?
if (( run_rc == 0 )); then
    python3 "$FIXED_DIR/scripts/verify_result.py" \
        "$FIXED_DIR/output/output.bin" "$FIXED_DIR/output/golden.bin" >>"$LOG_DIR/3.log" 2>&1
    verify_rc=$?
else
    verify_rc=1
fi
set -e

{
    diff -u "$ORIGINAL_DIR/run.sh" "$FIXED_DIR/run.sh" || true
    diff -u "$ORIGINAL_DIR/build.sh" "$FIXED_DIR/build.sh" || true
} >"$LOG_DIR/4.log"

if (( run_rc != 0 )); then
    printf 'MATMUL_NPU_AND_VERIFY failed stage=npu return_code=%d log=%s\n' "$run_rc" "$LOG_DIR/3.log"
    exit "$run_rc"
fi
if (( verify_rc != 0 )); then
    printf 'MATMUL_NPU_AND_VERIFY wrong_output log=%s\n' "$LOG_DIR/3.log"
    exit "$verify_rc"
fi

printf 'MATMUL_NPU_AND_VERIFY passed log=%s\n' "$LOG_DIR/3.log"
printf 'MATMUL_EXACT_DIFF log=%s\n' "$LOG_DIR/4.log"
