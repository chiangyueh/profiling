#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
mkdir -p "$BUILD"

source "$ROOT/scripts/env.sh"

run_logged() {
    local log_file="$1"
    local step="$2"
    shift 2

    if "$@" >>"$log_file" 2>&1; then
        return 0
    else
        local rc=$?
        echo "fatal: build step failed: ${step}, rc=${rc}" >&2
        echo "build_error_log: ${log_file}" >&2
        if [[ -s "$log_file" ]]; then
            tail -60 "$log_file" | sed 's/^/build_error: /' >&2
        else
            echo "build_error: log is empty" >&2
        fi
        return "$rc"
    fi
}

if [[ "${BUILD_MEMORY_KIB:-0}" -gt 0 ]]; then
    ulimit -v "${BUILD_MEMORY_KIB}"
fi

ARCH="$(uname -m)"
PLATFORM_ROOT="$CANN_ROOT/${ARCH}-linux"
CANN_INCLUDE="$PLATFORM_ROOT/include"
CANN_LIB="$PLATFORM_ROOT/lib64"
HOST_TILING_INCLUDE="$CANN_INCLUDE/ascendc/host_api/tiling"
HOST_HIGHLEVEL_INCLUDE="$CANN_INCLUDE/ascendc/highlevel_api"

ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-${SOC_VERSION:-Ascend910B}}"
echo "ASCENDC_SOC_VERSION=${ASCENDC_SOC_VERSION}"

COMMON_HOST=(
    -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic
    -I"$ROOT/host/include"
    -I"$ROOT/compat"
    -I"$HOST_TILING_INCLUDE"
    -I"$HOST_HIGHLEVEL_INCLUDE"
    -I"$CANN_INCLUDE"
)

HOST_SOURCES=(
    "$ROOT/host/src/search_types.cpp"
    "$ROOT/host/src/official_tiling.cpp"
    "$ROOT/host/src/proxy_model.cpp"
    "$ROOT/host/src/beam_lns_search.cpp"
    "$ROOT/host/src/csv_io.cpp"
    "$ROOT/host/src/main.cpp"
)

echo "[1/2] Building official-CANN tiling search host"
HOST_OBJ_DIR="$BUILD/host_obj"
mkdir -p "$HOST_OBJ_DIR"
: >"$BUILD/host_build.log"
HOST_OBJECTS=()
for source_file in "${HOST_SOURCES[@]}"; do
    object_file="$HOST_OBJ_DIR/$(basename "${source_file%.cpp}").o"
    run_logged "$BUILD/host_build.log" "compile $(basename "$source_file")" \
        g++ "${COMMON_HOST[@]}" -c "$source_file" -o "$object_file"
    HOST_OBJECTS+=("$object_file")
done
run_logged "$BUILD/host_build.log" "link matmul_tiling_search" \
    g++ "${HOST_OBJECTS[@]}" \
        -L"$CANN_LIB" -Wl,-rpath,"$CANN_LIB" \
        -ltiling_api -lplatform -lregister -lgraph -lgraph_base \
        -lascendcl -lascendalog -lc_sec -ldl -lpthread \
        -o "$BUILD/matmul_tiling_search"

echo "[2/2] Building official MatMulV3 runner and tuning-bank probe"
SOC_BUILD_NAME="$(printf '%s' "$ASCENDC_SOC_VERSION" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')"
NPU_BUILD="$BUILD/npu_cmake_${SOC_BUILD_NAME}"
if [[ -d "$NPU_BUILD" ]]; then
    find "$NPU_BUILD" -mindepth 1 -delete
fi
: >"$BUILD/kernel_build.log"
run_logged "$BUILD/kernel_build.log" "configure official runner and bank probe" \
    cmake -S "$ROOT/cmake_npu" -B "$NPU_BUILD" \
        -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
        -DCMAKE_BUILD_TYPE=Release

run_logged "$BUILD/kernel_build.log" "compile official runner and bank probe" \
    cmake --build "$NPU_BUILD" --target official_matmul_runner tiling_bank_probe \
        --parallel "${BUILD_JOBS:-1}"
cp "$NPU_BUILD/official_matmul_runner" "$BUILD/official_matmul_runner"
cp "$NPU_BUILD/tiling_bank_probe" "$BUILD/tiling_bank_probe"
: >"$BUILD/runner_build.log"

file "$BUILD/matmul_tiling_search" "$BUILD/official_matmul_runner" "$BUILD/tiling_bank_probe"
echo "Build completed: $BUILD"
