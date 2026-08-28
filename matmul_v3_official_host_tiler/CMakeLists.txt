cmake_minimum_required(VERSION 3.16)
project(MatMulV3OfficialHostTiler LANGUAGES CXX)

set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE PATH "ASCEND CANN package installation directory")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE PATH "install path" FORCE)
endif()

# This project has one route only: the installed CANN MatMulV3 operator API.
# No device kernel source, hand-filled MatmulTilingData, MM_KERNEL, or launch stub
# is compiled here.
add_executable(ascendc_kernels_bbit
    ${CMAKE_CURRENT_SOURCE_DIR}/main_v3.cpp
)
target_compile_features(ascendc_kernels_bbit PRIVATE cxx_std_17)
target_compile_options(ascendc_kernels_bbit PRIVATE -O2 -Wall)
target_include_directories(ascendc_kernels_bbit PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
target_link_directories(ascendc_kernels_bbit PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64
)
target_link_libraries(ascendc_kernels_bbit PRIVATE
    ascendcl
    opapi
    acl_op_compiler
    nnopbase
    dl
)

install(TARGETS ascendc_kernels_bbit RUNTIME DESTINATION bin)
