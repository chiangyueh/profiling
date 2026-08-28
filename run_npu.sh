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

cd /home/spbgu_jointlab/matmul_v3
(
    unset MM_M MM_N MM_K
    unset MM_BASE_M MM_BASE_N MM_BASE_K
    unset MM_SINGLE_M MM_SINGLE_N MM_SINGLE_K
    unset MM_STEP_M MM_STEP_N MM_STEP_Ka MM_STEP_Kb
    unset MM_ITER_ORDER MM_OP_TILING
    export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
    bash run.sh -r npu -v Ascend910B3
)
python3 scripts/verify_result.py output/output.bin output/golden.bin
