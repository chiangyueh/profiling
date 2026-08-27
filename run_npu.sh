#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs 100 deterministic MatMulV3 shapes through the installed original tiler.
Correct shapes are silent. Each wrong result produces one JSON object.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
        -d|--device) PHYSICAL_DEVICE="${2:?missing physical NPU ID}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) exit 2 ;;
    esac
done

[[ "${MODE}" == "full" ]] || exit 2
[[ "${PHYSICAL_DEVICE}" =~ ^[0-9]+$ ]] || exit 2

PRIVATE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/matmul-original-100.XXXXXX")"
cleanup() {
    [[ -n "${PRIVATE_ROOT:-}" && -d "${PRIVATE_ROOT}" ]] && rm -rf -- "${PRIVATE_ROOT}"
}
trap cleanup EXIT

CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
export CANN_ROOT
export ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-Ascend910B3}"
export SOC_VERSION="${SOC_VERSION:-${ASCENDC_SOC_VERSION}}"
export ASCEND_RT_VISIBLE_DEVICES="${PHYSICAL_DEVICE}"
export TUNE_BANK_PATH="${PRIVATE_ROOT}/empty_bank"
export ASCEND_CACHE_PATH="${PRIVATE_ROOT}/cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
mkdir -p "${TUNE_BANK_PATH}" "${ASCEND_CACHE_PATH}"
unset ASCEND_CUSTOM_OPP_PATH || true

if ! source "${ROOT}/scripts/env.sh" >"${PRIVATE_ROOT}/env.log" 2>&1; then
    printf '%s\n' '{"shape":"","錯誤的位置":"CANN環境載入失敗","validator攔截結果":"未執行"}'
    exit 1
fi

RUNNER="${ROOT}/build/official_matmul_runner"
if [[ ! -x "${RUNNER}" ]] ||
   ! "${RUNNER}" --help 2>/dev/null | grep -q -- '--structured-full-preflight'; then
    if ! env BUILD_COMPONENTS=runner BUILD_JOBS=1 \
        "${ROOT}/scripts/build_all.sh" >"${PRIVATE_ROOT}/build.log" 2>&1; then
        printf '%s\n' '{"shape":"","錯誤的位置":"MatMul runner建置失敗","validator攔截結果":"未執行"}'
        exit 1
    fi
fi

WORKLOADS="${PRIVATE_ROOT}/workloads.csv"
printf '%s\n' 'workload_id,m,n,k,dtype,trans_a,trans_b
matmul_001_skinny_n,3072,16,16383,fp16,0,0
matmul_002_skinny_n,3072,32,16384,bf16,0,1
matmul_003_skinny_n,3072,48,16385,fp16,1,0
matmul_004_skinny_n,3072,64,24575,bf16,1,1
matmul_005_skinny_n,3072,80,24576,fp16,0,0
matmul_006_skinny_n,3072,96,24577,bf16,0,1
matmul_007_skinny_n,4096,16,8191,fp16,1,0
matmul_008_skinny_n,4096,32,8192,bf16,1,1
matmul_009_skinny_n,4096,48,8193,fp16,0,0
matmul_010_skinny_n,4096,64,12287,bf16,0,1
matmul_011_skinny_n,4096,80,12288,fp16,1,0
matmul_012_skinny_n,4096,96,12289,bf16,1,1
matmul_013_skinny_n,5120,16,16383,fp16,0,0
matmul_014_skinny_n,5120,32,16384,bf16,0,1
matmul_015_skinny_n,5120,48,16385,fp16,1,0
matmul_016_skinny_n,3840,64,32767,bf16,1,1
matmul_017_skinny_n,3840,80,32768,fp16,0,0
matmul_018_skinny_n,3840,96,32769,bf16,0,1
matmul_019_skinny_n,6144,16,4095,fp16,1,0
matmul_020_skinny_n,6144,32,4096,bf16,1,1
matmul_021_skinny_n,6144,48,4097,fp16,0,0
matmul_022_skinny_n,6144,64,8191,bf16,0,1
matmul_023_skinny_n,6144,80,8192,fp16,1,0
matmul_024_skinny_n,6144,96,8193,bf16,1,1
matmul_025_skinny_m,16,3072,16383,fp16,0,0
matmul_026_skinny_m,32,3072,16384,bf16,0,1
matmul_027_skinny_m,48,3072,16385,fp16,1,0
matmul_028_skinny_m,64,3072,24575,bf16,1,1
matmul_029_skinny_m,80,3072,24576,fp16,0,0
matmul_030_skinny_m,96,3072,24577,bf16,0,1
matmul_031_skinny_m,16,4096,8191,fp16,1,0
matmul_032_skinny_m,32,4096,8192,bf16,1,1
matmul_033_skinny_m,48,4096,8193,fp16,0,0
matmul_034_skinny_m,64,4096,12287,bf16,0,1
matmul_035_skinny_m,80,4096,12288,fp16,1,0
matmul_036_skinny_m,96,4096,12289,bf16,1,1
matmul_037_skinny_m,16,5120,16383,fp16,0,0
matmul_038_skinny_m,32,5120,16384,bf16,0,1
matmul_039_skinny_m,48,5120,16385,fp16,1,0
matmul_040_skinny_m,64,3840,32767,bf16,1,1
matmul_041_skinny_m,80,3840,32768,fp16,0,0
matmul_042_skinny_m,96,3840,32769,bf16,0,1
matmul_043_split_k,16,16,8191,fp16,0,0
matmul_044_split_k,16,32,8192,bf16,0,1
matmul_045_split_k,32,16,8193,fp32,1,0
matmul_046_split_k,32,32,16383,fp16,1,1
matmul_047_split_k,48,64,16384,bf16,0,0
matmul_048_split_k,64,48,16385,fp32,0,1
matmul_049_split_k,64,64,24575,fp16,1,0
matmul_050_split_k,96,128,24576,bf16,1,1
matmul_051_split_k,128,96,24577,fp32,0,0
matmul_052_split_k,128,128,32767,fp16,0,1
matmul_053_split_k,160,160,32768,bf16,1,0
matmul_054_split_k,192,128,32769,fp32,1,1
matmul_055_split_k,128,192,49151,fp16,0,0
matmul_056_split_k,256,256,49152,bf16,0,1
matmul_057_split_k,16,48,12287,bf16,0,0
matmul_058_split_k,16,64,12288,fp16,1,0
matmul_059_split_k,32,48,12289,fp32,1,1
matmul_060_split_k,32,64,20479,bf16,0,1
matmul_061_split_k,48,16,20480,fp16,0,0
matmul_062_split_k,64,16,20481,fp32,1,0
matmul_063_split_k,48,48,28671,bf16,1,1
matmul_064_split_k,80,80,28672,fp16,0,1
matmul_065_split_k,96,96,28673,fp32,0,0
matmul_066_split_k,112,112,40959,bf16,1,0
matmul_067_split_k,144,144,40960,fp16,1,1
matmul_068_split_k,176,176,40961,fp32,0,1
matmul_069_split_k,224,224,57343,bf16,0,0
matmul_070_split_k,240,240,57344,fp16,1,0
matmul_071_tile_tail,112,240,4095,fp16,0,0
matmul_072_tile_tail,128,256,4096,bf16,0,1
matmul_073_tile_tail,144,272,4097,fp32,1,0
matmul_074_tile_tail,240,496,3071,fp16,1,1
matmul_075_tile_tail,256,512,3072,bf16,0,0
matmul_076_tile_tail,272,528,3073,fp32,0,1
matmul_077_tile_tail,496,752,6143,fp16,1,0
matmul_078_tile_tail,512,768,6144,bf16,1,1
matmul_079_tile_tail,528,784,6145,fp32,0,0
matmul_080_tile_tail,752,1008,8191,fp16,0,1
matmul_081_tile_tail,768,1024,8192,bf16,1,0
matmul_082_tile_tail,784,1040,8193,fp32,1,1
matmul_083_tile_tail,1008,1520,12287,fp16,0,0
matmul_084_tile_tail,1024,1536,12288,bf16,0,1
matmul_085_tile_tail,1040,1552,12289,fp32,1,0
matmul_086_tile_tail,1520,2032,4095,fp16,1,1
matmul_087_tile_tail,1536,2048,4096,bf16,0,0
matmul_088_tile_tail,1552,2064,4097,fp32,0,1
matmul_089_l2_grid,2544,1264,6161,fp16,0,0
matmul_090_l2_grid,2560,1280,6160,bf16,0,1
matmul_091_l2_grid,2576,1296,6159,fp32,1,0
matmul_092_l2_grid,3056,2032,12289,fp16,1,1
matmul_093_l2_grid,3072,2048,12288,bf16,0,0
matmul_094_l2_grid,3088,2064,12287,fp32,0,1
matmul_095_l2_grid,4080,2544,4097,fp16,1,0
matmul_096_l2_grid,4096,2560,4096,bf16,1,1
matmul_097_l2_grid,4112,2576,4095,fp32,0,0
matmul_098_l2_grid,5104,1008,8193,fp16,0,1
matmul_099_l2_grid,5120,1024,8192,bf16,1,0
matmul_100_l2_grid,5136,1040,8191,fp32,1,1' >"${WORKLOADS}"

REMAINING="${WORKLOADS}"
batch=0
while (( $(wc -l <"${REMAINING}") > 1 )); do
    batch=$((batch + 1))
    PROFILE="${PRIVATE_ROOT}/profile_${batch}.csv"
    SAMPLES="${PRIVATE_ROOT}/samples_${batch}.csv"
    RUN_LOG="${PRIVATE_ROOT}/runner_${batch}.log"
    set +e
    "${RUNNER}" \
        --candidates "${REMAINING}" \
        --output "${PROFILE}" \
        --samples-output "${SAMPLES}" \
        --device 0 \
        --warmup 0 \
        --repeat 1 \
        --samples 1 \
        --numeric-preflight-max-mib 256 \
        --structured-full-preflight \
        --preflight-only >"${RUN_LOG}" 2>&1
    runner_rc=$?
    set -e

    STATUS="${PRIVATE_ROOT}/status_${batch}.json"
    NEXT="${PRIVATE_ROOT}/remaining_${batch}.csv"
    python3 - "${REMAINING}" "${PROFILE}" "${runner_rc}" "${STATUS}" "${NEXT}" <<'PY' \
        >"${PRIVATE_ROOT}/parse_${batch}.log" 2>&1
import csv
import json
import sys
from pathlib import Path

remaining_path, profile_path, runner_rc, status_path, next_path = sys.argv[1:]
with open(remaining_path, newline="", encoding="utf-8") as stream:
    remaining = list(csv.DictReader(stream))
profile = []
if Path(profile_path).is_file():
    with open(profile_path, newline="", encoding="utf-8") as stream:
        profile = list(csv.DictReader(stream))

processed = len(profile)
status = {"kind": "complete"}
if processed:
    last = profile[-1]
    if str(last.get("success", "")).lower() not in {"1", "true"}:
        error = last.get("error", "")
        status = {
            "kind": "numeric" if "official structured numeric preflight failed at C index=" in error else "execution",
            "workload_id": last.get("workload_id", ""),
            "m": int(last.get("m", 0)),
            "n": int(last.get("n", 0)),
            "k": int(last.get("k", 0)),
            "dtype": last.get("dtype", ""),
            "trans_a": str(last.get("trans_a", "")).lower() in {"1", "true"},
            "trans_b": str(last.get("trans_b", "")).lower() in {"1", "true"},
            "error": error,
        }
    elif processed != len(remaining) or int(runner_rc) != 0:
        status = {"kind": "infrastructure"}
elif int(runner_rc) != 0:
    status = {"kind": "infrastructure"}

with open(status_path, "w", encoding="utf-8") as stream:
    json.dump(status, stream, ensure_ascii=False)

fieldnames = ["workload_id", "m", "n", "k", "dtype", "trans_a", "trans_b"]
with open(next_path, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(remaining[processed:])
PY

    kind="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["kind"])' "${STATUS}")"
    if [[ "${kind}" == "complete" ]]; then
        break
    fi
    if [[ "${kind}" == "infrastructure" ]]; then
        printf '%s\n' '{"shape":"","錯誤的位置":"NPU runner未產生結果","validator攔截結果":"未執行"}'
        exit 1
    fi

    RESULT="${PRIVATE_ROOT}/result_${batch}.json"
    python3 - "${ROOT}" "${STATUS}" "${RESULT}" <<'PY' \
        >"${PRIVATE_ROOT}/validator_${batch}.log" 2>&1
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
status = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
result_path = Path(sys.argv[3])
shape = (
    f"{status['m']}x{status['n']}x{status['k']} "
    f"{status['dtype']} trans={int(status['trans_a'])}{int(status['trans_b'])}"
)
error = status.get("error", "")

if status["kind"] != "numeric":
    result = {
        "shape": shape,
        "錯誤的位置": error[:400],
        "validator攔截結果": "未執行：沒有可比較的錯誤數值",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    raise SystemExit(0)

match = re.search(
    r"C index=(\d+), actual=([^,\s]+), expected=([^,\s]+)", error
)
if match:
    index, actual, expected = match.groups()
    position = f"C[{index}]"
    values = f"真實值={actual},計算值={expected}"
else:
    position = "C[未知]"
    values = error[:400]

sys.path.insert(0, str(root / "tools"))
try:
    import refine_matmul_v3_candidates as validator

    workload = validator.Workload(
        workload_id=status["workload_id"],
        m=status["m"], n=status["n"], k=status["k"],
        dtype=status["dtype"],
        trans_a=status["trans_a"], trans_b=status["trans_b"],
        max_cores=20,
    )
    callback = validator.invoke_official_callback(workload)
    hardware = validator.Hardware(
        aic_cores=20,
        l0a_bytes=64 * 1024,
        l0b_bytes=64 * 1024,
        l0c_bytes=256 * 1024,
        l1_bytes=512 * 1024,
        l2_bytes=192 * 1024 * 1024,
        l2_bytes_per_cycle_per_core=1.0,
        hbm_bytes_per_cycle_per_core=1.0,
    )
    last_line = {"value": None}

    def trace(frame, event, arg):
        if frame.f_code is validator.hard_legal.__code__ and event == "line":
            last_line["value"] = frame.f_lineno
        return trace

    sys.settrace(trace)
    try:
        accepted = validator.hard_legal(workload, callback.knowledge, hardware)
    finally:
        sys.settrace(None)

    rule_names = {
        602: "REQUIRED_FIELDS_MUST_BE_POSITIVE",
        604: "L2_TILE_BLOCK_MUST_BE_NONNEGATIVE",
        606: "TILING_ENABLE_TEMPLATE_CONTRACT",
        608: "USED_CORE_NUM_LIMIT",
        613: "ITERATE_ORDER_DOMAIN",
        615: "L0_DOUBLE_BUFFER_DOMAIN",
        623: "CUBE_BASE_ALIGNMENT",
        640: "L0_CAPACITY",
        645: "L1_DEPTH_STEP_DIVISIBILITY",
        647: "A1_BUFFER_COUNT",
        649: "B1_BUFFER_COUNT",
        659: "L1_CAPACITY",
        665: "K_STEP_COMPATIBILITY",
        682: "BASE_TEMPLATE_CONTRACT",
        693: "SINGLE_CORE_SPLIT_K_CONTRACT",
        699: "SINGLE_CORE_SPLIT_K_N_EXTENT",
        705: "DETERMINISTIC_SPLIT_K_BASE",
        725: "DETERMINISTIC_SPLIT_K_CONTRACT",
        751: "AL1_FULL_LOAD_CONTRACT",
        758: "AL1_FULL_LOAD_CAPACITY",
        778: "BL1_FULL_LOAD_CONTRACT",
        786: "BL1_RESIDENCY",
        789: "BL1_FIXPIPE_BOUND",
        796: "BL1_VEC_NZ2ND_CONTRACT",
        801: "BL1_SPECIAL_L2_BLOCK_ZERO",
        803: "SUPPORTED_TEMPLATE_CONTRACT",
        810: "SPLIT_K_L2_BLOCK_POSITIVE",
        817: "DETERMINISTIC_SPLIT_K_CORE_LIMIT",
    }
    if accepted:
        validator_result = "攔截失敗"
    else:
        knowledge = callback.knowledge
        line = last_line["value"]
        rule = rule_names.get(line, f"hard_legal_line_{line}")
        if line == 682:
            if knowledge["singleCoreK"] != workload.k:
                rule = "BASE_SINGLE_CORE_K_MUST_EQUAL_K"
            elif knowledge["singleCoreM"] > knowledge["baseM"]:
                rule = "BASE_SINGLE_CORE_M_EXCEEDS_BASE_M"
            elif knowledge["singleCoreN"] > knowledge["baseN"]:
                rule = "BASE_SINGLE_CORE_N_EXCEEDS_BASE_N"
            elif knowledge["baseM"] > validator.align_up(knowledge["singleCoreM"], 16):
                rule = "BASE_M_EXCEEDS_ALIGNED_SINGLE_CORE_M"
            elif knowledge["baseN"] > validator.align_up(knowledge["singleCoreN"], 16):
                rule = "BASE_N_EXCEEDS_ALIGNED_SINGLE_CORE_N"
            elif knowledge["baseK"] > validator.align_up(
                knowledge["singleCoreK"], validator.base_k_alignment(workload)
            ):
                rule = "BASE_K_EXCEEDS_ALIGNED_SINGLE_CORE_K"
            else:
                rule = "BASE_L2_SCHEDULE_CONTRACT"
        validator_result = f"攔截成功:{rule}"
except Exception as exception:
    validator_result = f"validator執行失敗:{type(exception).__name__}"

result = {
    "shape": shape,
    position: values,
    "validator攔截結果": validator_result,
}
result_path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
PY
    cat "${RESULT}"
    REMAINING="${NEXT}"
done
