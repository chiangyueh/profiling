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
PROJECT_DIR="$ROOT_DIR/colleague_matmul_v3"
SOURCE_SIGNATURE="$({
    git -C "$ROOT_DIR" ls-files -z colleague_matmul_v3 run_npu.sh
} | sort -z | while IFS= read -r -d '' relative; do
    sha256sum "$ROOT_DIR/$relative"
done | sha256sum | cut -c1-20)"
RUN_ID="${MATMUL_AUDIT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
RESULT_DIR="$ROOT_DIR/results/matmul_cost_filter_shape_audit/$SOURCE_SIGNATURE/$RUN_ID"
LOG_DIR="$RESULT_DIR/logs"
BUILD_STATE_DIR="$ROOT_DIR/.benchmark_state/matmul_cost_filter_shape_audit/$SOURCE_SIGNATURE"
RUN_STATE_DIR="$BUILD_STATE_DIR/$RUN_ID"
BUILD_LOG="$BUILD_STATE_DIR/build.log"
CAMPAIGN_LOG="$RUN_STATE_DIR/campaign.log"
mkdir -p "$LOG_DIR" "$RUN_STATE_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
export MATMUL_AUDIT_RUN_ID="$RUN_ID"
export MATMUL_AUDIT_LOG_DIR="$LOG_DIR"
export MATMUL_AUDIT_SEED="${MATMUL_AUDIT_SEED:-20260828}"
export MATMUL_AUDIT_PROFILE_ATTEMPTS="${MATMUL_AUDIT_PROFILE_ATTEMPTS:-2}"
export MM_DATA_SEED="${MM_DATA_SEED:-20260828}"
export MM_REUSE_GOLDEN=1

printf 'MATMUL_FILTER_AUDIT_READY run_id=%s unique_shapes=200 device=%s logs=%s\n' \
    "$RUN_ID" "$PHYSICAL_DEVICE" "$LOG_DIR"

if [[ ! -x "$PROJECT_DIR/ascendc_kernels_bbit" ]]; then
    printf 'MATMUL_FILTER_AUDIT_BUILD begin\n'
    if ! bash "$PROJECT_DIR/build.sh" -r npu -v Ascend910B3 >"$BUILD_LOG" 2>&1; then
        printf 'MATMUL_FILTER_AUDIT_BUILD failed log=%s\n' "$BUILD_LOG"
        exit 1
    fi
    touch "$BUILD_STATE_DIR/build_complete"
    printf 'MATMUL_FILTER_AUDIT_BUILD passed\n'
else
    printf 'MATMUL_FILTER_AUDIT_BUILD cached\n'
fi

printf 'MATMUL_FILTER_AUDIT_RUN begin\n'
set +e
(
    cd "$PROJECT_DIR"
    python3 shape_audit.py
) 2>&1 | tee "$CAMPAIGN_LOG"
run_rc=${PIPESTATUS[0]}
set -e
if (( run_rc != 0 )); then
    printf 'MATMUL_FILTER_AUDIT_RUN failed log=%s\n' "$CAMPAIGN_LOG"
    exit "$run_rc"
fi

printf 'MATMUL_FILTER_AUDIT_RUN passed summary=%s\n' "$RESULT_DIR/summary.json"
