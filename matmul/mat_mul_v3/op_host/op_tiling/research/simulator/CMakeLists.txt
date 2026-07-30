cmake_minimum_required(VERSION 3.16)
project(matmul_v3_simulator_repro LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(RUN_MODE "sim" CACHE STRING "AscendC run mode")
set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend SoC")
set(
    ASCEND_CANN_PACKAGE_PATH
    "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE PATH "CANN toolkit root"
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(
    OP_KERNEL_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../op_kernel"
    ABSOLUTE
)
set(KERNEL_FILES "${CMAKE_CURRENT_LIST_DIR}/mat_mul_v3_base_fixed.cpp")

if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
    set(
        ASCENDC_CMAKE_DIR
        "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    )
elseif(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
    set(
        ASCENDC_CMAKE_DIR
        "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake"
    )
else()
    message(FATAL_ERROR "ascendc_kernel_cmake is not installed")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

ascendc_library(matmul_v3_sim_kernel SHARED "${KERNEL_FILES}")
ascendc_compile_definitions(
    matmul_v3_sim_kernel PRIVATE
    DTYPE_X1=half
    DTYPE_X2=half
    DTYPE_BIAS=half
    DTYPE_Y=half
)
ascendc_include_directories(
    matmul_v3_sim_kernel PRIVATE
    "${OP_KERNEL_DIR}"
)

add_executable(matmul_v3_simulator_repro simulator_main.cpp)
set(
    SIMULATOR_LIB_DIR
    "${ASCEND_CANN_PACKAGE_PATH}/tools/simulator/${SOC_VERSION}/lib"
)
target_compile_options(
    matmul_v3_simulator_repro PRIVATE
    -O2
    -D_GLIBCXX_USE_CXX11_ABI=0
)
target_link_directories(
    matmul_v3_simulator_repro PRIVATE
    "${SIMULATOR_LIB_DIR}"
)
target_link_options(
    matmul_v3_simulator_repro PRIVATE
    "-Wl,-rpath,${SIMULATOR_LIB_DIR}"
    "-Wl,-rpath-link,${SIMULATOR_LIB_DIR}"
)
target_link_libraries(
    matmul_v3_simulator_repro PRIVATE
    host_intf_pub
    matmul_v3_sim_kernel
    graph_base
    tiling_api
    register
    platform
    ascendalog
    npu_drv_camodel
    dl
)
