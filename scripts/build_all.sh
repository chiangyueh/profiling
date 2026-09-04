#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
mkdir -p "$BUILD"
BUILD_COMPONENTS="${BUILD_COMPONENTS:-all}"

if [[ "${BUILD_COMPONENTS}" != "all" && "${BUILD_COMPONENTS}" != "host" && \
      "${BUILD_COMPONENTS}" != "runner" ]]; then
    echo "fatal: BUILD_COMPONENTS must be 'all', 'host', or 'runner'" >&2
    exit 2
fi

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
    "$ROOT/host/src/hardware_cost_model.cpp"
    "$ROOT/host/src/hardware_profiles.cpp"
    "$ROOT/host/src/hardware_path_builders.cpp"
    "$ROOT/host/src/indexed_read_path.cpp"
    "$ROOT/host/src/indexed_update_path.cpp"
    "$ROOT/host/src/proxy_model.cpp"
    "$ROOT/host/src/beam_lns_search.cpp"
    "$ROOT/host/src/csv_io.cpp"
    "$ROOT/host/src/main.cpp"
)

if [[ "${BUILD_COMPONENTS}" == "all" || "${BUILD_COMPONENTS}" == "host" ]]; then
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
    run_logged "$BUILD/host_build.log" "compile indexed_update_cost" \
        g++ "${COMMON_HOST[@]}" -c "$ROOT/tools/indexed_update_cost.cpp" \
        -o "$HOST_OBJ_DIR/indexed_update_cost.o"
    run_logged "$BUILD/host_build.log" "link indexed_update_cost" \
        g++ "$HOST_OBJ_DIR/indexed_update_cost.o" \
        "$HOST_OBJ_DIR/hardware_cost_model.o" \
        "$HOST_OBJ_DIR/hardware_profiles.o" \
        "$HOST_OBJ_DIR/hardware_path_builders.o" \
        "$HOST_OBJ_DIR/indexed_update_path.o" \
        -o "$BUILD/indexed_update_cost"
    run_logged "$BUILD/host_build.log" "compile indexed_read_cost" \
        g++ "${COMMON_HOST[@]}" -c "$ROOT/tools/indexed_read_cost.cpp" \
        -o "$HOST_OBJ_DIR/indexed_read_cost.o"
    run_logged "$BUILD/host_build.log" "link indexed_read_cost" \
        g++ "$HOST_OBJ_DIR/indexed_read_cost.o" \
        "$HOST_OBJ_DIR/hardware_cost_model.o" \
        "$HOST_OBJ_DIR/hardware_profiles.o" \
        "$HOST_OBJ_DIR/hardware_path_builders.o" \
        "$HOST_OBJ_DIR/indexed_read_path.o" \
        -o "$BUILD/indexed_read_cost"
elif [[ "${BUILD_COMPONENTS}" == "runner" ]]; then
    echo "[1/2] Skipping tiling search host (runner-only build)"
fi

if [[ "${BUILD_COMPONENTS}" == "host" ]]; then
    file "$BUILD/matmul_tiling_search"
    echo "Build completed: $BUILD"
    exit 0
fi

echo "[2/2] Building official baseline and direct MatMulV3 kernels (jobs=${BUILD_JOBS:-1})"
SOC_BUILD_NAME="$(printf '%s' "$ASCENDC_SOC_VERSION" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')"
NPU_BUILD="$BUILD/npu_cmake_${SOC_BUILD_NAME}"
mkdir -p "$NPU_BUILD"
: >"$BUILD/kernel_build.log"
run_logged "$BUILD/kernel_build.log" "configure official runner" \
    cmake -S "$ROOT/cmake_npu" -B "$NPU_BUILD" \
        -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
        -DCMAKE_BUILD_TYPE=Release

DIRECT_KERNEL_TARGETS=(
    direct_matmul_kernel_fp16_0 direct_matmul_kernel_fp16_1
    direct_matmul_kernel_fp16_20 direct_matmul_kernel_fp16_21
    direct_matmul_kernel_fp16_30 direct_matmul_kernel_fp16_31
    direct_matmul_kernel_fp16_201
    direct_matmul_kernel_fp16_10201
    direct_matmul_kernel_bf16_0 direct_matmul_kernel_bf16_1
    direct_matmul_kernel_bf16_20 direct_matmul_kernel_bf16_21
    direct_matmul_kernel_bf16_30 direct_matmul_kernel_bf16_31
    direct_matmul_kernel_bf16_201
    direct_matmul_kernel_bf16_10201
    direct_matmul_kernel_fp32_1 direct_matmul_kernel_fp32_21
    direct_matmul_kernel_fp32_31
    direct_matmul_kernel_fp32_101 direct_matmul_kernel_fp32_201
    direct_matmul_kernel_fp32_10201 direct_matmul_kernel_fp32_20201
)
MATMUL_V3_KERNEL_DIR="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3"
DIRECT_KERNEL_BUILD_SIGNATURE="$({
    printf '%s\0' \
        "${ROOT}/direct_matmul/kernel_entry.cpp" \
        "${ROOT}/direct_matmul/mat_mul_v3_tiling_data.h" \
        "${ROOT}/cmake_npu/CMakeLists.txt"
    find "${MATMUL_V3_KERNEL_DIR}" -type f -print0
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
kernel_count="${#DIRECT_KERNEL_TARGETS[@]}"
kernel_index=0
for target in "${DIRECT_KERNEL_TARGETS[@]}"; do
    kernel_index=$((kernel_index + 1))
    target_stamp="${NPU_BUILD}/.${target}.sha256"
    target_library="${NPU_BUILD}/lib/lib${target}.a"
    target_include="${NPU_BUILD}/include/${target}"
    if [[ -s "${target_library}" && -d "${target_include}" && \
          -f "${target_stamp}" && \
          "$(cat "${target_stamp}" 2>/dev/null || true)" == \
              "${DIRECT_KERNEL_BUILD_SIGNATURE}" ]] && \
       ar t "${target_library}" >/dev/null 2>&1; then
        echo "DIRECT_KERNEL_BUILD ${kernel_index}/${kernel_count} ${target} cached"
        continue
    fi
    # CANN 8.1's merge_obj_text.sh rewrites its input in place and cannot be
    # invoked twice.  An interrupted ExternalProject therefore has to replay
    # only this target's private preprocess directory from source.
    target_preprocess="${NPU_BUILD}/${target}_preprocess-prefix"
    if [[ -d "${target_preprocess}" ]]; then
        cmake -E remove_directory "${target_preprocess}"
    fi
    echo "DIRECT_KERNEL_BUILD ${kernel_index}/${kernel_count} ${target} begin"
    run_logged "$BUILD/kernel_build.log" "compile ${target}" \
        cmake --build "$NPU_BUILD" --target "${target}" \
        --parallel "${BUILD_JOBS:-1}"
    printf '%s\n' "${DIRECT_KERNEL_BUILD_SIGNATURE}" >"${target_stamp}"
    echo "DIRECT_KERNEL_BUILD ${kernel_index}/${kernel_count} ${target} passed"
done
echo "DIRECT_RUNNER_LINK begin"
run_logged "$BUILD/kernel_build.log" "compile official runner" \
    cmake --build "$NPU_BUILD" --target official_matmul_runner \
    --parallel "${BUILD_JOBS:-1}"
run_logged "$BUILD/kernel_build.log" "link direct runner" \
    cmake --build "$NPU_BUILD" --target direct_matmul_runner \
    --parallel "${BUILD_JOBS:-1}"
echo "DIRECT_RUNNER_LINK passed"
cp "$NPU_BUILD/official_matmul_runner" "$BUILD/official_matmul_runner"
cp "$NPU_BUILD/direct_matmul_runner" "$BUILD/direct_matmul_runner"
: >"$BUILD/runner_build.log"

if [[ "${BUILD_COMPONENTS}" == "all" ]]; then
    file "$BUILD/matmul_tiling_search" "$BUILD/official_matmul_runner" "$BUILD/direct_matmul_runner"
else
    file "$BUILD/official_matmul_runner" "$BUILD/direct_matmul_runner"
fi
echo "Build completed: $BUILD"
