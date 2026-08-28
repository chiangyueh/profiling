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
exec "$ROOT_DIR/run_matmul_colleague_ab.sh" -d "$PHYSICAL_DEVICE"
