#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN_ROOT="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
BUILD_DIR="${MM_BUILD_DIR:-$ROOT_DIR/build}"
OUT_DIR="${MM_OUT_DIR:-$ROOT_DIR/out}"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
    -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT"
cmake --build "$BUILD_DIR" --parallel 1
cmake --install "$BUILD_DIR"
