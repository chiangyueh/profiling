#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
REFERENCE_SOURCE="${MATMUL_V3_REFERENCE_DIR:-/home/spbgu_jointlab/matmul_v3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
        -d|--device) PHYSICAL_DEVICE="${2:?missing device ID}"; shift 2 ;;
        -h|--help)
            printf 'Usage: %s --mode full [-d PHYSICAL_NPU_ID]\n' "$0"
            exit 0
            ;;
        *) exit 2 ;;
    esac
done

[[ "$MODE" == "full" && "$PHYSICAL_DEVICE" =~ ^[0-9]+$ ]] || exit 2

fail() {
    python3 - "$1" <<'PY'
import json
import sys

print(json.dumps({"status": "failed", "error": sys.argv[1]}, ensure_ascii=False, separators=(",", ":")))
PY
    exit 1
}

[[ -d "$REFERENCE_SOURCE" ]] || fail "reference MatMulV3 directory is absent: $REFERENCE_SOURCE"
[[ -f "$REFERENCE_SOURCE/run.sh" ]] || fail "reference run.sh is absent"
[[ -f "$REFERENCE_SOURCE/run_all.sh" ]] || fail "reference run_all.sh is absent"
[[ -f "$REFERENCE_SOURCE/scripts/gen_data.py" ]] || fail "reference scripts/gen_data.py is absent"
[[ -f "$REFERENCE_SOURCE/scripts/verify_result.py" ]] || fail "reference scripts/verify_result.py is absent"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
STATE="$ROOT/.benchmark_state/matmul_v3_reference_replay/$RUN_ID"
SOURCE_COPY="$STATE/source"
LOG_DIR="$ROOT/results/matmul_v3_reference_replay/$RUN_ID/logs"
mkdir -p "$SOURCE_COPY" "$LOG_DIR"

# Do not build, clean, or write in the shared reference directory. The launch
# is performed from a private copy that retains the reference scripts/source.
tar -C "$REFERENCE_SOURCE" \
    --exclude='./.git' \
    --exclude='./build' \
    --exclude='./out' \
    --exclude='./input' \
    --exclude='./output' \
    --exclude='./profiler' \
    --exclude='./profiling' \
    -cf - . | tar -C "$SOURCE_COPY" -xf -

SOURCE_MANIFEST="$LOG_DIR/source_manifest.sha256"
(
    cd "$SOURCE_COPY"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$SOURCE_MANIFEST"

export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
export MM_M=512
export MM_N=512
export MM_K=512
export MM_KERNEL=0
export RUN_WITH_TOOLCHAIN=1

LOG_FILE="$LOG_DIR/1.log"
set +e
(
    cd "$SOURCE_COPY"
    bash run.sh -r npu -v Ascend910B3
    python3 scripts/verify_result.py output/output.bin output/golden.bin
) 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

if (( STATUS != 0 )); then
    exit "$STATUS"
fi
