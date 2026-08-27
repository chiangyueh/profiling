cmake_minimum_required(VERSION 3.16)
project(matmul_validator_probe LANGUAGES CXX)

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
if(NOT DEFINED MATMUL_V3_KERNEL_DIR)
    set(MATMUL_V3_KERNEL_DIR
        "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3"
    )
endif()

if(NOT EXISTS "${MATMUL_V3_KERNEL_DIR}/mat_mul_base_kernel.h")
    message(FATAL_ERROR "original CANN MatMulV3 BASE kernel was not found: ${MATMUL_V3_KERNEL_DIR}")
endif()

set(PLATFORM_ROOT "${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH}-linux")
set(CANN_INCLUDE "${PLATFORM_ROOT}/include")
set(CANN_LIB "${PLATFORM_ROOT}/lib64")
set(CANN_DEVLIB "${PLATFORM_ROOT}/devlib")

set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC version")
set(RUN_MODE "npu")
set(ASCENDC_CMAKE "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
if(NOT EXISTS "${ASCENDC_CMAKE}")
    set(ASCENDC_CMAKE "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
endif()
if(NOT EXISTS "${ASCENDC_CMAKE}")
    set(ASCENDC_CMAKE "${PLATFORM_ROOT}/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
endif()
if(NOT EXISTS "${ASCENDC_CMAKE}")
    message(FATAL_ERROR "CANN AscendC CMake module was not found")
endif()
include("${ASCENDC_CMAKE}")

ascendc_library(matmul_validator_probe_kernel STATIC kernel.cpp)
ascendc_include_directories(matmul_validator_probe_kernel PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${MATMUL_V3_KERNEL_DIR}"
)

add_executable(matmul_validator_probe runner.cpp)
target_include_directories(matmul_validator_probe PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CANN_INCLUDE}"
)
target_link_directories(matmul_validator_probe PRIVATE "${CANN_LIB}")
target_link_libraries(matmul_validator_probe PRIVATE
    matmul_validator_probe_kernel ascendcl runtime dl pthread
)
target_link_options(matmul_validator_probe PRIVATE
    "-Wl,-rpath-link,${CANN_LIB}"
    "-Wl,-rpath-link,${CANN_DEVLIB}"
)
