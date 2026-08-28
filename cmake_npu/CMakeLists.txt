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
set(OP_TILING_LIB_DIR
    "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${CANN_ARCH}"
)

find_library(OP_TILING_LIBRARY
    NAMES optiling
    PATHS
        "${CANN_LIB}"
        "${OP_TILING_LIB_DIR}"
    NO_DEFAULT_PATH
)
if(NOT OP_TILING_LIBRARY)
    message(FATAL_ERROR
        "liboptiling.so was not found. Checked: ${CANN_LIB};${OP_TILING_LIB_DIR}"
    )
endif()

add_executable(official_matmul_runner
    "${PROJECT_ROOT}/runner/official_matmul_runner.cpp"
)

add_executable(tiling_bank_probe
    "${PROJECT_ROOT}/tools/tiling_bank_probe.cpp"
)

target_include_directories(tiling_bank_probe PRIVATE
    "${CANN_INCLUDE}"
    "${CANN_INCLUDE}/experiment/platform"
)

target_link_directories(tiling_bank_probe PRIVATE
    "${CANN_LIB}"
)

target_include_directories(official_matmul_runner PRIVATE
    "${CANN_INCLUDE}"
)

target_link_directories(official_matmul_runner PRIVATE
    "${CANN_LIB}"
)

target_link_libraries(official_matmul_runner PRIVATE
    ascendcl
    nnopbase
    opapi
    dl
    pthread
)

target_compile_definitions(tiling_bank_probe PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(tiling_bank_probe PRIVATE
    "${OP_TILING_LIBRARY}"
    platform
    dl
    pthread
)

target_link_options(official_matmul_runner PRIVATE
    "-Wl,-rpath-link,${CANN_LIB}"
    "-Wl,-rpath-link,${CANN_DEVLIB}"
)

target_link_options(tiling_bank_probe PRIVATE
    "-Wl,-rpath-link,${CANN_LIB}"
    "-Wl,-rpath-link,${CANN_DEVLIB}"
    "-Wl,-rpath-link,${CANN_ROOT_LIB}"
    "-Wl,-rpath-link,${OP_TILING_LIB_DIR}"
)

set_target_properties(official_matmul_runner PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH ""
)

set_target_properties(tiling_bank_probe PROPERTIES
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH ""
)
