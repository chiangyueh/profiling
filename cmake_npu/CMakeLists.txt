cmake_minimum_required(VERSION 3.16)
project(ascend_matmul_npu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()

if(NOT DEFINED CANN_ARCH)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE CANN_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

set(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(PLATFORM_ROOT "${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH}-linux")
set(CANN_INCLUDE "${PLATFORM_ROOT}/include")
set(CANN_LIB "${PLATFORM_ROOT}/lib64")
set(CANN_DEVLIB "${PLATFORM_ROOT}/devlib")
set(CANN_ROOT_LIB "${ASCEND_CANN_PACKAGE_PATH}/lib64")
set(MATMUL_V3_KERNEL_DIR
    "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3"
)
add_executable(official_matmul_runner
    "${PROJECT_ROOT}/runner/official_matmul_runner.cpp"
)

set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC version")
set(RUN_MODE "npu")
set(ASCENDC_CMAKE
    "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
)
if(NOT EXISTS "${ASCENDC_CMAKE}")
    set(ASCENDC_CMAKE
        "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    )
endif()
if(NOT EXISTS "${ASCENDC_CMAKE}")
    set(ASCENDC_CMAKE
        "${PLATFORM_ROOT}/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    )
endif()
if(NOT EXISTS "${ASCENDC_CMAKE}")
    message(FATAL_ERROR "CANN AscendC CMake module was not found")
endif()
if(NOT EXISTS "${MATMUL_V3_KERNEL_DIR}/mat_mul_v3.cpp")
    message(FATAL_ERROR "installed CANN 8.1 MatMulV3 kernel source was not found")
endif()
include("${ASCENDC_CMAKE}")

function(add_direct_matmul_kernel target kernel dtype suffix)
    ascendc_library(${target} STATIC
        "${PROJECT_ROOT}/direct_matmul/kernel_entry.cpp"
    )
    ascendc_include_directories(${target} PRIVATE
        "${PROJECT_ROOT}/direct_matmul"
        "${MATMUL_V3_KERNEL_DIR}"
    )
    ascendc_compile_definitions(${target} PRIVATE
        MATMUL_DIRECT_KERNEL=${kernel}
        DTYPE_X1=${dtype}
        DTYPE_X2=${dtype}
        DTYPE_Y=${dtype}
        DTYPE_BIAS=${dtype}
        MATMUL_DIRECT_TILING_KEY=100000000000000${suffix}UL
    )
endfunction()

set(DIRECT_MATMUL_KERNEL_TARGETS "")
function(register_direct_matmul dtype_name dtype suffix suffix_padded)
    set(target "direct_matmul_kernel_${dtype_name}_${suffix}")
    set(kernel "direct_matmul_${dtype_name}_k${suffix}")
    add_direct_matmul_kernel(
        ${target} ${kernel} ${dtype} ${suffix_padded}
    )
    set(DIRECT_MATMUL_KERNEL_TARGETS
        ${DIRECT_MATMUL_KERNEL_TARGETS} ${target} PARENT_SCOPE
    )
endfunction()

# Exact dtype/suffix combinations emitted by the fixed 70-shape campaign.
# Each library compiles one original CANN branch; this keeps compiler memory
# bounded and avoids compiling impossible dtype/family combinations.
register_direct_matmul(fp16 half 0 00000)
register_direct_matmul(fp16 half 1 00001)
register_direct_matmul(fp16 half 20 00020)
register_direct_matmul(fp16 half 21 00021)
register_direct_matmul(fp16 half 30 00030)
register_direct_matmul(fp16 half 31 00031)
register_direct_matmul(fp16 half 201 00201)
register_direct_matmul(fp16 half 10201 10201)

register_direct_matmul(bf16 bfloat16_t 0 00000)
register_direct_matmul(bf16 bfloat16_t 1 00001)
register_direct_matmul(bf16 bfloat16_t 20 00020)
register_direct_matmul(bf16 bfloat16_t 21 00021)
register_direct_matmul(bf16 bfloat16_t 30 00030)
register_direct_matmul(bf16 bfloat16_t 31 00031)
register_direct_matmul(bf16 bfloat16_t 201 00201)
register_direct_matmul(bf16 bfloat16_t 10201 10201)

register_direct_matmul(fp32 float 1 00001)
register_direct_matmul(fp32 float 21 00021)
register_direct_matmul(fp32 float 31 00031)
register_direct_matmul(fp32 float 101 00101)
register_direct_matmul(fp32 float 201 00201)
register_direct_matmul(fp32 float 10201 10201)
register_direct_matmul(fp32 float 20201 20201)

# direct_matmul_runner is intentionally linked outside this CMake graph by
# build_all.sh.  Depending on any ascendc_library target here would retraverse
# CANN 8.1 BUILD_ALWAYS ExternalProjects after their archives already exist.

target_include_directories(official_matmul_runner PRIVATE
    "${CANN_INCLUDE}"
)

target_link_directories(official_matmul_runner PRIVATE
    "${CANN_LIB}"
)

target_link_libraries(official_matmul_runner PRIVATE
    ascendcl
    acl_op_compiler
    nnopbase
    opapi
    dl
    pthread
)

target_link_options(official_matmul_runner PRIVATE
    "-Wl,-rpath-link,${CANN_LIB}"
    "-Wl,-rpath-link,${CANN_DEVLIB}"
)

set_target_properties(official_matmul_runner PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH ""
)
