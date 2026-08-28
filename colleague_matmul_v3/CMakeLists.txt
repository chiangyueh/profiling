cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)

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

install(TARGETS ascendc_kernels_bbit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
