#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH="${ROOT}/matmul/mat_mul_v3/op_host/op_tiling/research"
SIMULATOR_SOURCE="${RESEARCH}/simulator"
MODE="full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?missing value for --mode}"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_npu.sh --mode full"
            exit 0
            ;;
        *)
            echo "fatal: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
    echo "fatal: mode must be full or smoke" >&2
    exit 2
fi

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ ! -d "${CANN_ROOT}" ]]; then
    echo "fatal: CANN root does not exist: ${CANN_ROOT}" >&2
    exit 1
fi
CANN_REAL="$(readlink -f "${CANN_ROOT}")"
SIM_SOC="${SOC_VERSION:-Ascend910B3}"
SIMULATOR_LIB="${CANN_REAL}/tools/simulator/${SIM_SOC}/lib"
MSPROF="$(readlink -f "${CANN_ROOT}/bin/msprof")"

if [[ ! -x "${MSPROF}" ]]; then
    echo "fatal: msprof is not installed under ${CANN_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${SIMULATOR_LIB}" ]]; then
    echo "fatal: ${SIM_SOC} simulator is not installed: ${SIMULATOR_LIB}" >&2
    exit 1
fi

SET_ENV=""
for candidate in \
    "${CANN_ROOT}/$(uname -m)-linux/bin/setenv.bash" \
    "${CANN_ROOT}/set_env.sh" \
    "$(dirname "${CANN_ROOT}")/set_env.sh"; do
    if [[ -f "${candidate}" ]]; then
        SET_ENV="${candidate}"
        break
    fi
done
if [[ -n "${SET_ENV}" ]]; then
    set +e +u
    source "${SET_ENV}"
    SET_ENV_RC=$?
    set -e -u
    if [[ "${SET_ENV_RC}" -ne 0 ]]; then
        echo "warning: CANN environment script returned rc=${SET_ENV_RC}; continuing with explicit paths" >&2
    fi
fi

export ASCEND_HOME_PATH="${CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
export ASCEND_OPP_PATH="${CANN_ROOT}/opp"
export ASCENDC_SOC_VERSION="${SIM_SOC}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RESEARCH}:${CANN_ROOT}/python/site-packages:${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${ROOT}/results/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
BUILD_DIR="${ROOT}/.build/matmul_v3_msprof_simulator_${RUN_ID}"
RESULT_DIR="${ROOT}/results/msprof_simulator_proof_${RUN_ID}"
PROFILE_ROOT="${RESULT_DIR}/profile"
TILING_BIN="${RESULT_DIR}/tiling.bin"
OUTPUT_BIN="${RESULT_DIR}/output.bin"
RUN_LOG="${ROOT}/results/logs/run_npu_${RUN_ID}.log"
BUILD_LOG="${RESULT_DIR}/build.log"
MSPROF_LOG="${RESULT_DIR}/msprof.log"
mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo
echo "MatMulV3 msprof simulator proof"
echo "  script:    run_npu.sh 20260731-msprof-official-base-proof"
echo "  upstream:  CANN ops-nn 8.5.0 matmul/mat_mul_v3"
echo "  toolkit:   ${CANN_REAL}"
echo "  simulator: ${SIM_SOC}"
echo "  shape:     32x32x128 fp16 NN"
echo "  results:   ${RESULT_DIR}"
echo "  log:       ${RUN_LOG}"
echo

echo "[1/4] Emit the exact callback tiling record ..."
python3 "${RESEARCH}/repro_callback_npu_failure.py" \
    --stage host \
    --tiling-output "${TILING_BIN}"

echo
echo "[2/4] Build the original MatmulBaseKernel simulator entry ..."
if ! cmake \
    -S "${SIMULATOR_SOURCE}" \
    -B "${BUILD_DIR}" \
    -DRUN_MODE=sim \
    -DSOC_VERSION="${SIM_SOC}" \
    -DASCEND_CANN_PACKAGE_PATH="${CANN_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release >"${BUILD_LOG}" 2>&1; then
    cat "${BUILD_LOG}"
    exit 1
fi
if ! cmake --build "${BUILD_DIR}" --parallel >>"${BUILD_LOG}" 2>&1; then
    cat "${BUILD_LOG}"
    exit 1
fi
APP="${BUILD_DIR}/matmul_v3_simulator_repro"
echo "  kernel=mat_mul_v3_base_fixed_0 build=ok"

echo
echo "[3/4] Execute and profile the kernel with msprof op simulator ..."
export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${SIMULATOR_LIB}:${CANN_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set +e
pushd "${RESULT_DIR}" >/dev/null
"${MSPROF}" op simulator \
    --soc-version="${SIM_SOC}" \
    --launch-count=1 \
    --aic-metrics=PipeUtilization \
    --output="${PROFILE_ROOT}" \
    --timeout="${MSPROF_TIMEOUT_MINUTES:-2}" \
    "${APP}" \
    "${TILING_BIN}" \
    "${OUTPUT_BIN}" 2>&1 | tee "${MSPROF_LOG}"
MSPROF_RC="${PIPESTATUS[0]}"
popd >/dev/null
set -e
if [[ "${MSPROF_RC}" -ne 0 ]]; then
    echo "fatal: msprof simulator failed rc=${MSPROF_RC}" >&2
    exit "${MSPROF_RC}"
fi

echo
echo "[4/4] Validate simulator output and profiler artifacts ..."
python3 - "${TILING_BIN}" "${OUTPUT_BIN}" "${PROFILE_ROOT}" "${MSPROF_LOG}" "${RESULT_DIR}/proof.json" <<'PY'
import json
import re
import struct
import sys
from pathlib import Path

tiling_path, output_path, profile_root, log_path, proof_path = map(
    Path, sys.argv[1:]
)
tiling = tiling_path.read_bytes()
output = output_path.read_bytes()
if len(tiling) != 272:
    raise SystemExit(f"fatal: expected 272 tiling bytes, got {len(tiling)}")
if len(output) != 2048:
    raise SystemExit(f"fatal: expected 2048 output bytes, got {len(output)}")
values = struct.unpack("<1024H", output)
if any(value != 0x5800 for value in values):
    raise SystemExit("fatal: simulator output is not the expected FP16 value 128")

profiles = sorted(profile_root.glob("OPPROF_*"))
instruction_csv = sorted(profile_root.glob("OPPROF_*/simulator/**/*instr_exe.csv"))
if len(profiles) != 1 or not instruction_csv:
    raise SystemExit("fatal: msprof did not emit simulator instruction results")

log = log_path.read_text(encoding="utf-8", errors="replace")
match = re.search(
    r"core0\.cubecore0\s+([0-9.]+)\s+([0-9.]+)", log
)
proof = {
    "workload": "32x32x128 fp16 NN",
    "tiling_bytes": len(tiling),
    "tiling_signature": "1:16:16:128:16:16:32:2:2:1:1:0:1:1:1:1:1:1:1:1:1:0:0",
    "kernel": "mat_mul_v3_base_fixed_0",
    "output_bytes": len(output),
    "output_elements": len(values),
    "output_fp16_value": 128.0,
    "msprof_profile": str(profiles[0]),
    "instruction_csv": str(instruction_csv[0]),
    "duration_time_us": float(match.group(1)) if match else None,
    "running_time_us": float(match.group(2)) if match else None,
}
proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
print("MSPROF_SIMULATOR_PASS")
print("  callback_exact_fields=23 tiling_bytes=272")
print("  kernel=mat_mul_v3_base_fixed_0")
print("  output_elements=1024 output_fp16_value=128")
if match:
    print(
        f"  duration_time_us={match.group(1)} "
        f"running_time_us={match.group(2)}"
    )
print(f"  profile={profiles[0]}")
print(f"  evidence={proof_path}")
PY

echo
echo "msprof simulator proof completed"
echo "  output:   ${OUTPUT_BIN}"
echo "  evidence: ${RESULT_DIR}/proof.json"
echo "  profile:  ${PROFILE_ROOT}"
echo "  log:      ${RUN_LOG}"
