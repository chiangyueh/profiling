#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE_ID=2

while getopts ":d:h" option; do
    case "$option" in
        d) DEVICE_ID="$OPTARG" ;;
        h)
            echo "Usage: $0 [-d PHYSICAL_NPU_ID]"
            exit 0
            ;;
        :) echo "fatal: -$OPTARG requires a value" >&2; exit 2 ;;
        \?) echo "fatal: unknown option -$OPTARG" >&2; exit 2 ;;
    esac
done

if [[ ! "$DEVICE_ID" =~ ^[0-9]+$ ]]; then
    echo "fatal: device ID must be a non-negative integer" >&2
    exit 2
fi

# All environment changes are scoped to this shell and its children.  The
# private RuntimeKb bank, CMake build and ACL cache stay under this repository.
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" >/dev/null
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_ID"

STATE_ROOT="$ROOT/.benchmark_state/matmul_runtimekb_validator_replay"
BUILD_DIR="$STATE_ROOT/build"
RUN_DIR="$STATE_ROOT/run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
BUILD_LOG="$STATE_ROOT/build.log"
mkdir -p "$STATE_ROOT" "$RUN_DIR"

echo "MATMUL_RUNTIMEKB_REPLAY_BUILD begin"
if ! cmake -S "$ROOT/cmake_npu" -B "$BUILD_DIR" \
        -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
        -DCMAKE_BUILD_TYPE=Release >"$BUILD_LOG" 2>&1 || \
   ! cmake --build "$BUILD_DIR" \
        --target official_matmul_runner tiling_bank_probe \
        --parallel 1 >>"$BUILD_LOG" 2>&1; then
    echo "MATMUL_RUNTIMEKB_REPLAY_BUILD failed log=$BUILD_LOG" >&2
    tail -30 "$BUILD_LOG" >&2
    exit 1
fi
echo "MATMUL_RUNTIMEKB_REPLAY_BUILD passed"

python3 "$ROOT/tools/replay_matmul_runtimekb_validator_case.py" \
    --runner "$BUILD_DIR/official_matmul_runner" \
    --bank-probe "$BUILD_DIR/tiling_bank_probe" \
    --cann-root "$CANN_ROOT" \
    --state-dir "$RUN_DIR" \
    --soc Ascend910B3 \
    --aic-cores 20
