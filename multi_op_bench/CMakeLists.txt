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
option(SCATTER_ELEMENTS_ONLY "Compile only the ScatterElementsV2 measurement path" OFF)
set(REMAINING_OPERATOR "" CACHE STRING "Compile one unfinished CANN-8.1 attention operator")
if(SCATTER_ELEMENTS_ONLY)
    target_compile_definitions(multi_op_npu_runner PRIVATE SCATTER_ELEMENTS_ONLY=1)
endif()
if(REMAINING_OPERATOR)
    if(NOT REMAINING_OPERATOR MATCHES "^(flash_attention_score_grad|fused_infer_attention_score)$")
        message(FATAL_ERROR "unsupported REMAINING_OPERATOR=${REMAINING_OPERATOR}")
    endif()
    target_compile_definitions(multi_op_npu_runner PRIVATE REMAINING_OPERATOR_ONLY=1)
    if(REMAINING_OPERATOR STREQUAL "flash_attention_score_grad")
        target_compile_definitions(multi_op_npu_runner PRIVATE FASG_ONLY=1)
    else()
        target_compile_definitions(multi_op_npu_runner PRIVATE FIAS_ONLY=1)
    endif()
endif()
target_include_directories(multi_op_npu_runner PRIVATE "${CANN_INCLUDE}")
target_link_directories(multi_op_npu_runner PRIVATE "${CANN_LIB}")
# Both the reference and candidate launch use the exact installed CANN 8.1
# OpAPI. Candidate workers additionally load a checkout-local host-tiling
# bridge that delegates to the installed final-RC1 tiler. No Python/TBE
# runtime compiler or alpha host implementation is involved.
target_link_libraries(multi_op_npu_runner PRIVATE ascendcl nnopbase opapi pthread dl)
target_link_options(
    multi_op_npu_runner PRIVATE
    "-Wl,-rpath-link,${CANN_LIB}"
    "-Wl,-rpath-link,${CANN_DEVLIB}"
)
set_target_properties(multi_op_npu_runner PROPERTIES SKIP_BUILD_RPATH TRUE BUILD_WITH_INSTALL_RPATH FALSE)
