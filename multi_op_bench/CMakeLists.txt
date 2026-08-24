cmake_minimum_required(VERSION 3.16)
project(multi_op_real_npu_bench LANGUAGES CXX)

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
    execute_process(COMMAND uname -m OUTPUT_VARIABLE CANN_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

set(CANN_PLATFORM "${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH}-linux")
set(CANN_INCLUDE "${CANN_PLATFORM}/include")
set(CANN_LIB "${CANN_PLATFORM}/lib64")
set(CANN_DEVLIB "${CANN_PLATFORM}/devlib")

add_executable(multi_op_npu_runner runner.cpp)
target_include_directories(multi_op_npu_runner PRIVATE "${CANN_INCLUDE}")
target_link_directories(multi_op_npu_runner PRIVATE "${CANN_LIB}")
# The installed reference is linked normally. The checkout-local CANN 8.1
# package's generated C++ ACLNN entry point is loaded by absolute path with
# dlopen, so no Python/TBE runtime compiler is part of the custom path.
target_link_libraries(multi_op_npu_runner PRIVATE ascendcl nnopbase opapi pthread dl)
target_link_options(
    multi_op_npu_runner PRIVATE
    "-Wl,-rpath-link,${CANN_LIB}"
    "-Wl,-rpath-link,${CANN_DEVLIB}"
)
set_target_properties(multi_op_npu_runner PROPERTIES SKIP_BUILD_RPATH TRUE BUILD_WITH_INSTALL_RPATH FALSE)
