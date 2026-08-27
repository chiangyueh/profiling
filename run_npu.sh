#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Runs 5,000 deterministic MatMulV3 boundary shapes through the installed original tiler.
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

PRIVATE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/matmul-original-5000.XXXXXX")"
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
{
    printf 'workload_id,m,n,k,dtype,trans_a,trans_b\n'
    dimensions=(
        16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
        272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512
        528 544 560 576 624 640 656 752 768 784 1008 1024 1040 1264
        1280 1296 1520 1536 1552 1776 1792 1808 2032 2048 2064 2288
        2304 2320 2544 2560 2576 3056 3072 3088 3568 3584 3600 4080
        4096 4112
    )
    k_boundaries=(
        63 64 65 127 128 129 255 256 257 511 512 513
        1023 1024 1025 2047 2048 2049 3071 3072 3073
        4095 4096 4097 6143 6144 6145 8191 8192 8193
        12287 12288 12289 16383 16384 16385 24575 24576 24577
        32767 32768 32769 40959 40960 40961 49151 49152 49153
        57343 57344 57345
    )
    dtypes=(fp16 bf16 fp32)
    trans_a=(0 0 1 1)
    trans_b=(0 1 0 1)
    dimension_count=${#dimensions[@]}
    pair_count=$((dimension_count * dimension_count))
    for ((sequence = 0; sequence < 5000; ++sequence)); do
        flat=$(((sequence * 5179) % pair_count))
        m_index=$((flat / dimension_count))
        n_index=$((flat % dimension_count))
        m=${dimensions[m_index]}
        n=${dimensions[n_index]}
        dtype_index=$(((m_index + n_index + sequence) % ${#dtypes[@]}))
        dtype=${dtypes[dtype_index]}
        if [[ "${dtype}" == "fp32" ]]; then
            element_bytes=4
        else
            element_bytes=2
        fi
        max_k=$(((192 * 1024 * 1024) / ((m + n) * element_bytes)))
        ((max_k > 60000)) && max_k=60000
        valid_k=()
        for k in "${k_boundaries[@]}"; do
            ((k <= max_k)) && valid_k+=("${k}")
        done
        k_index=$(((m_index * 17 + n_index * 29 + sequence * 7) % ${#valid_k[@]}))
        trans_index=$(((m_index + n_index * 2 + sequence) % 4))
        printf 'matmul_%04d_boundary_grid,%d,%d,%d,%s,%d,%d\n' \
            "$((sequence + 1))" "${m}" "${n}" "${valid_k[k_index]}" \
            "${dtype}" "${trans_a[trans_index]}" "${trans_b[trans_index]}"
    done
} >"${WORKLOADS}"

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
