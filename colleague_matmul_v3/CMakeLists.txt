cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)

# +++ BEGIN: private official MatMulV3 package built from ../src/matmul/mat_mul_v3.
set(MM_LOCAL_VENDOR_ROOT "" CACHE PATH "private customize vendor root containing the official MatMulV3 package")
# +++ END: private official MatMulV3 package.

# Путь к исходникам кернела mat_mul_v3 (нужен ХОСТУ ради mat_mul_v3_tiling_data.h).
# Сам кернел подключается по абсолютному пути внутри matmul_v3_launch.cpp.
set(MMV3_OP_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel"
    CACHE STRING "path to mat_mul_v3 op_kernel sources"
)
option(MM_SHARED_SERVER_SAFE "Do not reset the selected physical device during host cleanup" ON)
option(MM_ASCEND_KERNEL_LAUNCH_ONLY "Build only the generated NPU launch stub on the host" ON)
set(ASCEND_KERNEL_LAUNCH_ONLY ${MM_ASCEND_KERNEL_LAUNCH_ONLY})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

# ${KERNEL_FILES} — исходники кернела (device-компиляция).
# Тут наша обёртка, которая инклюдит mat_mul_v3.cpp.
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/matmul_v3_launch.cpp)

set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()

# --- BEGIN: colleague standalone direct-launch target retained for 4.log.
# Хост: тайлинг заполняется прямо в main_v3.cpp, отдельного tiling.cpp нет.
add_executable(ascendc_kernels_bbit
    ${CMAKE_CURRENT_SOURCE_DIR}/main_v3.cpp
)

# Хосту нужен заголовок со структурой MatmulTilingData
target_include_directories(ascendc_kernels_bbit PRIVATE
    ${MMV3_OP_KERNEL_DIR}
)

target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_compile_definitions(ascendc_kernels_bbit PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    SOC_VERSION="${SOC_VERSION}"
    MM_KERNEL=0
    $<$<BOOL:${MM_SHARED_SERVER_SAFE}>:MM_SHARED_SERVER_SAFE>
)

target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE}
    tiling_api
    register
    platform
    ascendalog
    dl
)
# --- END: colleague standalone direct-launch target retained for 4.log.

# +++ BEGIN: official operator/host-tiler target used for 3.log.
if(NOT MM_LOCAL_VENDOR_ROOT)
    message(FATAL_ERROR "MM_LOCAL_VENDOR_ROOT must point to the private installed MatMulV3 vendor directory")
endif()

add_executable(ascendc_kernels_bbit_official
    ${CMAKE_CURRENT_SOURCE_DIR}/main_v3.cpp
)
target_compile_definitions(ascendc_kernels_bbit_official PRIVATE
    MM_USE_OFFICIAL_HOST_TILER=1
    MM_SHARED_SERVER_SAFE
)
target_compile_features(ascendc_kernels_bbit_official PRIVATE cxx_std_17)
target_compile_options(ascendc_kernels_bbit_official PRIVATE -O2 -Wall)
target_include_directories(ascendc_kernels_bbit_official PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/matmul/mat_mul_v3/op_host
)
target_link_directories(ascendc_kernels_bbit_official PRIVATE
    ${MM_LOCAL_VENDOR_ROOT}/op_api/lib
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64
)
target_link_libraries(ascendc_kernels_bbit_official PRIVATE
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    dl
)
# +++ END: official operator/host-tiler target used for 3.log.

install(TARGETS ascendc_kernels_bbit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# +++ BEGIN: install the official host-tiler runner beside the retained runner.
install(TARGETS ascendc_kernels_bbit_official
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
# +++ END: install the official host-tiler runner.
