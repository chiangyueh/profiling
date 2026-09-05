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
CANN_DEVLIB="$PLATFORM_ROOT/devlib"
CANN_ROOT_LIB="$CANN_ROOT/lib64"
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
    sha256sum \
        "${ROOT}/direct_matmul/kernel_entry.cpp" \
        "${ROOT}/direct_matmul/mat_mul_v3_tiling_data.h"
    find "${MATMUL_V3_KERNEL_DIR}" -type f -print0 | \
        sort -z | xargs -0 sha256sum
    printf '%s\n' \
        "cann81-direct-kernel-v2:${ASCENDC_SOC_VERSION}:${DIRECT_KERNEL_TARGETS[*]}"
} | sha256sum | cut -d' ' -f1)"
kernel_count="${#DIRECT_KERNEL_TARGETS[@]}"

cleanup_kernel_intermediates() {
    local cleanup_target="$1"
    case "${cleanup_target}" in
        direct_matmul_kernel_*) ;;
        *) echo "fatal: invalid cleanup target ${cleanup_target}" >&2; return 2 ;;
    esac
    local cleanup_path
    for cleanup_path in \
        "${NPU_BUILD}/${cleanup_target}_precompile-prefix" \
        "${NPU_BUILD}/${cleanup_target}_preprocess-prefix" \
        "${NPU_BUILD}/${cleanup_target}_host-prefix" \
        "${NPU_BUILD}/${cleanup_target}_aic_device-prefix" \
        "${NPU_BUILD}/${cleanup_target}_aiv_device-prefix" \
        "${NPU_BUILD}/${cleanup_target}_aic_device_dir" \
        "${NPU_BUILD}/${cleanup_target}_aiv_device_dir" \
        "${NPU_BUILD}/${cleanup_target}_merge_obj_dir" \
        "${NPU_BUILD}/${cleanup_target}_host_dir" \
        "${NPU_BUILD}/auto_gen/${cleanup_target}" \
        "${NPU_BUILD}/CMakeFiles/${cleanup_target}_host_stub_obj.dir"; do
        [[ -d "${cleanup_path}" ]] && cmake -E remove_directory "${cleanup_path}"
    done
}

kernel_identity() {
    local identity_target="$1"
    local identity="${identity_target#direct_matmul_kernel_}"
    local dtype="${identity%%_*}"
    local suffix="${identity#*_}"
    printf 'aclrtlaunch_direct_matmul_%s_k%s\n' "${dtype}" "${suffix}"
}

kernel_archive_valid() {
    local valid_target="$1"
    local valid_library="${NPU_BUILD}/lib/lib${valid_target}.a"
    local valid_symbol
    valid_symbol="$(kernel_identity "${valid_target}")"
    local valid_header="${NPU_BUILD}/include/${valid_target}/${valid_symbol}.h"
    [[ -s "${valid_library}" && -f "${valid_header}" ]] || return 1
    ar t "${valid_library}" >/dev/null 2>&1 || return 1
    nm -g --defined-only "${valid_library}" 2>/dev/null | \
        grep -E "[[:space:]]${valid_symbol}$" >/dev/null
}

recover_kernel_archive() {
    local recover_target="$1"
    local recover_library="${NPU_BUILD}/lib/lib${recover_target}.a"
    local recover_stub recover_host
    recover_stub="$(find \
        "${NPU_BUILD}/CMakeFiles/${recover_target}_host_stub_obj.dir" \
        -type f -name 'host_stub.cpp.o' -print -quit 2>/dev/null || true)"
    recover_host="$(find "${NPU_BUILD}/${recover_target}_host_dir" \
        -type f -name 'kernel_entry.cpp.o' -print -quit 2>/dev/null || true)"
    [[ -n "${recover_stub}" && -n "${recover_host}" && \
       -f "${NPU_BUILD}/elf_tool.c.o" && \
       -f "${NPU_BUILD}/ascendc_runtime.cpp.o" ]] || return 1
    mkdir -p "${NPU_BUILD}/lib"
    ar rcs "${recover_library}" \
        "${recover_stub}" "${recover_host}" \
        "${NPU_BUILD}/elf_tool.c.o" "${NPU_BUILD}/ascendc_runtime.cpp.o"
    kernel_archive_valid "${recover_target}"
}

# A completed archive contains the packed device binary.  Its ExternalProject
# trees are no longer needed and can consume many GiB across 23 variants.
# Remove only those private generated trees before recovering/building any
# missing archive; libraries, launch headers and stamps remain intact.
for target in "${DIRECT_KERNEL_TARGETS[@]}"; do
    if kernel_archive_valid "${target}"; then
        cleanup_kernel_intermediates "${target}"
    fi
done

build_free_kib="$(df -Pk "${BUILD}" | awk 'END {print $4}')"
if [[ ! "${build_free_kib}" =~ ^[0-9]+$ || "${build_free_kib}" -lt 524288 ]]; then
    echo "fatal: less than 512 MiB remains after private kernel cleanup" >&2
    exit 1
fi
echo "DIRECT_BUILD_STORAGE free_mib=$((build_free_kib / 1024))"

run_logged "$BUILD/kernel_build.log" "configure official runner" \
    cmake -S "$ROOT/cmake_npu" -B "$NPU_BUILD" \
        -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
        -DCMAKE_BUILD_TYPE=Release

kernel_index=0
for target in "${DIRECT_KERNEL_TARGETS[@]}"; do
    kernel_index=$((kernel_index + 1))
    target_stamp="${NPU_BUILD}/.${target}.sha256"
    target_library="${NPU_BUILD}/lib/lib${target}.a"
    target_include="${NPU_BUILD}/include/${target}"
    target_identity="${target#direct_matmul_kernel_}"
    target_dtype="${target_identity%%_*}"
    target_suffix="${target_identity#*_}"
    target_symbol="aclrtlaunch_direct_matmul_${target_dtype}_k${target_suffix}"
    target_header="${target_include}/${target_symbol}.h"
    if [[ ! -s "${target_library}" ]]; then
        recover_kernel_archive "${target}" || true
    fi
    kernel_cache_valid=0
    if [[ -s "${target_library}" && -f "${target_header}" && \
          -f "${target_stamp}" ]] && \
       ar t "${target_library}" >/dev/null 2>&1 && \
       nm -g --defined-only "${target_library}" 2>/dev/null | \
           grep -Eq "[[:space:]]${target_symbol}$"; then
        if [[ "$(cat "${target_stamp}" 2>/dev/null || true)" == \
              "${DIRECT_KERNEL_BUILD_SIGNATURE}" ]]; then
            kernel_cache_valid=1
        elif [[ ! "${ROOT}/direct_matmul/kernel_entry.cpp" -nt "${target_library}" && \
                ! "${ROOT}/direct_matmul/mat_mul_v3_tiling_data.h" -nt "${target_library}" ]] && \
             [[ -z "$(find "${MATMUL_V3_KERNEL_DIR}" -type f \
                 -newer "${target_library}" -print -quit)" ]]; then
            # One-time migration from the former stamp, whose hash included
            # link-only CMake text.  The archive symbol and every actual
            # kernel input are checked before retaining the expensive build.
            printf '%s\n' "${DIRECT_KERNEL_BUILD_SIGNATURE}" >"${target_stamp}"
            kernel_cache_valid=1
        fi
    fi
    if [[ "${kernel_cache_valid}" -eq 1 ]]; then
        cleanup_kernel_intermediates "${target}"
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
    kernel_archive_valid "${target}" || {
        echo "fatal: ${target} archive/header/symbol validation failed" >&2
        exit 1
    }
    printf '%s\n' "${DIRECT_KERNEL_BUILD_SIGNATURE}" >"${target_stamp}"
    cleanup_kernel_intermediates "${target}"
    echo "DIRECT_KERNEL_BUILD ${kernel_index}/${kernel_count} ${target} passed"
done
echo "DIRECT_RUNNER_LINK begin"
run_logged "$BUILD/kernel_build.log" "compile official runner" \
    cmake --build "$NPU_BUILD" --target official_matmul_runner \
    --parallel "${BUILD_JOBS:-1}"

DIRECT_RUNNER_COMMAND=(
    g++ -std=c++17 -O3 -DNDEBUG
    "-I${ROOT}/direct_matmul" "-I${CANN_INCLUDE}"
)
for target in "${DIRECT_KERNEL_TARGETS[@]}"; do
    DIRECT_RUNNER_COMMAND+=("-I${NPU_BUILD}/include/${target}")
done
DIRECT_RUNNER_COMMAND+=("${ROOT}/direct_matmul/runner.cpp")
for target in "${DIRECT_KERNEL_TARGETS[@]}"; do
    DIRECT_RUNNER_COMMAND+=("${NPU_BUILD}/lib/lib${target}.a")
done
DIRECT_RUNNER_COMMAND+=(
    "-L${CANN_LIB}" "-L${CANN_ROOT_LIB}"
    "-L${CANN_ROOT}/tools/simulator/${ASCENDC_SOC_VERSION}/lib"
    "-Wl,-rpath-link,${CANN_LIB}" "-Wl,-rpath-link,${CANN_DEVLIB}"
    -lascendcl -lruntime -lregister -lerror_manager -lprofapi
    -lge_common_base -lascendalog -lmmpa -lascend_dump -lc_sec
    -ldl -lpthread -o "${BUILD}/direct_matmul_runner"
)
run_logged "$BUILD/kernel_build.log" "link direct runner without kernel rebuild" \
    "${DIRECT_RUNNER_COMMAND[@]}"
echo "DIRECT_RUNNER_LINK passed"
cp "$NPU_BUILD/official_matmul_runner" "$BUILD/official_matmul_runner"
: >"$BUILD/runner_build.log"

if [[ "${BUILD_COMPONENTS}" == "all" ]]; then
    file "$BUILD/matmul_tiling_search" "$BUILD/official_matmul_runner" "$BUILD/direct_matmul_runner"
else
    file "$BUILD/official_matmul_runner" "$BUILD/direct_matmul_runner"
fi
echo "Build completed: $BUILD"
