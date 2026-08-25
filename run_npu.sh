#!/usr/bin/env bash
set -Eeuo pipefail

# Stable public entry: full always runs every currently selected operator.
# The operator-specific driver is an internal implementation detail.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP="${OP_NPU_WARMUP:-2}"
SAMPLES="${OP_NPU_SAMPLES:-5}"

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs the complete CANN-8.1 campaign for GatherElementsV2,
FlashAttentionScoreGrad, and FusedInferAttentionScore. ScatterElements is not
included. Each operator must produce exactly 5,000 output-validated NPU
device-event latency records in rotating JSONL logs of at most 50 MiB.
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

[[ "${MODE}" == "full" ]] || {
    echo "fatal: only --mode full is supported" >&2
    usage >&2
    exit 2
}
for value_name in PHYSICAL_DEVICE WARMUP SAMPLES; do
    value="${!value_name}"
    [[ "${value}" =~ ^[0-9]+$ ]] || {
        echo "fatal: ${value_name} must be a non-negative integer" >&2
        exit 2
    }
done
(( SAMPLES >= 1 )) || { echo "fatal: samples must be at least 1" >&2; exit 2; }

exec "${ROOT}/run_remaining_npu.sh" --operator all -d "${PHYSICAL_DEVICE}" \
    --warmup "${WARMUP}" --samples "${SAMPLES}"
