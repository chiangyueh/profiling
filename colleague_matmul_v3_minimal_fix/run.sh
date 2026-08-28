#!/bin/bash
export MM_M=${MM_M:-512}
export MM_N=${MM_N:-512}
export MM_K=${MM_K:-512}
export MM_BASE_M=${MM_BASE_M:-16}
export MM_BASE_N=${MM_BASE_N:-32}
export MM_BASE_K=${MM_BASE_K:-96}
# +++ BEGIN: minimal legal BASE geometry fix.
export MM_SINGLE_M=${MM_SINGLE_M:-16}
export MM_SINGLE_N=${MM_SINGLE_N:-32}
# +++ END: minimal legal BASE geometry fix.
export MM_SINGLE_K=${MM_SINGLE_K:-256}
export MM_STEP_M=${MM_STEP_M:-1}
export MM_STEP_N=${MM_STEP_N:-1}
export MM_STEP_Ka=${MM_STEP_Ka:-4}
export MM_STEP_Kb=${MM_STEP_Kb:-4}
export MM_ITER_ORDER=${MM_ITER_ORDER:-0}
export MM_OP_TILING=${MM_OP_TILING:-0}

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd "$CURRENT_DIR"

SOC_VERSION="Ascend910B3"
RUN_MODE="sim"
SHORT=r:,v:
LONG=run-mode:,soc-version:
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR]: Unexpected option: $1"
        exit 1
        ;;
    esac
done

# bash build.sh -r ${RUN_MODE} -v ${SOC_VERSION}

_ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"

export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH

if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

if [ "${RUN_MODE}" = "npu" ]; then
    MM_M=$MM_M MM_N=$MM_N MM_K=$MM_K python3 scripts/gen_data.py
    msprof op ./ascendc_kernels_bbit
elif [ "${RUN_MODE}" = "sim" ]; then
    msprof op simulator --soc-version=${SOC_VERSION} ./ascendc_kernels_bbit
elif [ "${RUN_MODE}" = "cpu" ]; then
    MM_M=$MM_M MM_N=$MM_N MM_K=$MM_K python3 scripts/gen_data.py
    ./ascendc_kernels_bbit
fi
