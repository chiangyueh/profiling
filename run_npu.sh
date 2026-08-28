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
ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE" bash run.sh -r npu -v Ascend910B3
python3 scripts/verify_result.py output/output.bin output/golden.bin
