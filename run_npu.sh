#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
        -d|--device) PHYSICAL_DEVICE="${2:?missing device ID}"; shift 2 ;;
        -h|--help)
            printf 'Usage: %s --mode full [-d PHYSICAL_NPU_ID]\n' "$0"
            exit 0
            ;;
        *) exit 2 ;;
    esac
done

[[ "$MODE" == full && "$PHYSICAL_DEVICE" =~ ^[0-9]+$ ]] || exit 2
[[ -f "$CANN_ROOT/version.cfg" ]] || {
    printf '%s\n' '{"shape":"","錯誤的位置":"CANN 8.1環境不存在","validator攔截結果":"未執行"}'
    exit 1
}

export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_OPP_PATH="$CANN_ROOT/opp"
export PATH="$CANN_ROOT/bin:$CANN_ROOT/compiler/ccec_compiler/bin:$PATH"
export LD_LIBRARY_PATH="$CANN_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CANN_ROOT/python/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MAX_COMPILE_CORE_NUMBER=1
export TBE_PARALLEL_COMPILER=1
export TILINGKEY_PAR_COMPILE=0

# The upstream install rule requires this common include directory even when
# this MatMul-only checkout intentionally contains none of its optional files.
mkdir -p "$ROOT/src/common/inc/kernel"

SOURCE_HASH="$({
    find "$ROOT/src/matmul/mat_mul_v3" "$ROOT/src/common" "$ROOT/cmake" "$ROOT/third_party" -type f -print0
    printf '%s\0' "$ROOT/CMakeLists.txt" "$ROOT/CMakePresets.json"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
STATE="$ROOT/.benchmark_state/matmul_original_${SOURCE_HASH:0:20}"
BUILD="$STATE/build"
INSTALL="$STATE/install"
LOGS="$STATE/logs"
mkdir -p "$BUILD" "$INSTALL" "$LOGS"

fail_infra() {
    local message="$1"
    printf '{"shape":"","錯誤的位置":"%s","validator攔截結果":"未執行"}\n' "$message"
    exit 1
}

if [[ ! -f "$INSTALL/.complete" ]]; then
    cmake -S "$ROOT" -B "$BUILD" \
        -DBUILD_OPEN_PROJECT=ON \
        -DASCEND_COMPUTE_UNIT=ascend910b \
        -DASCEND_OP_NAME=mat_mul_v3 \
        -DCUSTOM_ASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
        -DASCEND_THIRD_LIB_PATH="$ROOT/third_party" \
        -DCHECK_COMPATIBLE=FALSE \
        -DENABLE_CCACHE=OFF \
        -DCMAKE_INSTALL_PREFIX="$INSTALL" >"$LOGS/configure.log" 2>&1 \
        || fail_infra '官方MatMulV3 configure失敗'
    cmake --build "$BUILD" --target install -j1 >"$LOGS/build.log" 2>&1 \
        || fail_infra '官方MatMulV3 build失敗'
    touch "$INSTALL/.complete"
fi

CUSTOM_VENDOR="$INSTALL/packages/vendors/customize"
[[ -d "$CUSTOM_VENDOR" ]] || fail_infra '私有MatMulV3 package不存在'
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_VENDOR"
export LD_LIBRARY_PATH="$CUSTOM_VENDOR/op_api/lib:$CUSTOM_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -m):$LD_LIBRARY_PATH"
if [[ -f "$CUSTOM_VENDOR/bin/set_env.bash" ]]; then
    source "$CUSTOM_VENDOR/bin/set_env.bash" >"$LOGS/private_env.log" 2>&1
fi

RUNNER_CPP="$STATE/original_matmul_batch.cpp"
RUNNER="$STATE/original_matmul_batch"
cat >"$RUNNER_CPP" <<'CPP'
#include <acl/acl.h>
#include <aclnn_matmul.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static uint16_t FloatToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x7fffffu;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        mantissa += 0x0fffu + ((mantissa >> 13) & 1u);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    mantissa += 0x0fffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x800000u) {
        mantissa = 0;
        if (++exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

static float HalfToFloat(uint16_t value) {
    uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; --exponent; }
            mantissa &= 0x03ffu;
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t FloatToBf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

static float Bf16ToFloat(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static size_t ElementBytes(const std::string &dtype) { return dtype == "fp32" ? 4 : 2; }
static aclDataType AclType(const std::string &dtype) {
    if (dtype == "fp16") return ACL_FLOAT16;
    if (dtype == "bf16") return ACL_BF16;
    return ACL_FLOAT;
}
static void Store(std::vector<uint8_t> &buffer, size_t index, const std::string &dtype, float value) {
    if (dtype == "fp32") std::memcpy(buffer.data() + index * 4, &value, 4);
    else {
        uint16_t raw = dtype == "fp16" ? FloatToHalf(value) : FloatToBf16(value);
        std::memcpy(buffer.data() + index * 2, &raw, 2);
    }
}
static float Load(const std::vector<uint8_t> &buffer, size_t index, const std::string &dtype) {
    if (dtype == "fp32") { float value; std::memcpy(&value, buffer.data() + index * 4, 4); return value; }
    uint16_t raw; std::memcpy(&raw, buffer.data() + index * 2, 2);
    return dtype == "fp16" ? HalfToFloat(raw) : Bf16ToFloat(raw);
}
static float Quantize(float value, const std::string &dtype) {
    if (dtype == "fp16") return HalfToFloat(FloatToHalf(value));
    if (dtype == "bf16") return Bf16ToFloat(FloatToBf16(value));
    return value;
}

struct TensorBuffer {
    void *device = nullptr;
    aclTensor *tensor = nullptr;
};

static int MakeTensor(const std::vector<uint8_t> &host, const std::vector<int64_t> &logical,
                      const std::vector<int64_t> &strides, const std::vector<int64_t> &storage,
                      aclDataType dtype, TensorBuffer &result) {
    if (aclrtMalloc(&result.device, host.size(), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 1;
    if (aclrtMemcpy(result.device, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return 2;
    result.tensor = aclCreateTensor(logical.data(), logical.size(), dtype, strides.data(), 0,
                                    ACL_FORMAT_ND, storage.data(), storage.size(), result.device);
    return result.tensor == nullptr ? 3 : 0;
}

static void Release(TensorBuffer &buffer) {
    if (buffer.tensor) aclDestroyTensor(buffer.tensor);
    if (buffer.device) aclrtFree(buffer.device);
    buffer.tensor = nullptr;
    buffer.device = nullptr;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    std::ifstream shapes(argv[1]);
    std::ofstream failure(argv[2], std::ios::trunc);
    if (!shapes || !failure) return 2;
    if (aclInit(nullptr) != ACL_SUCCESS) return 10;
    if (aclrtSetDevice(0) != ACL_SUCCESS) return 11;
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) return 12;

    int64_t m, n, k;
    int ta, tb;
    std::string dtype;
    size_t line = 0;
    while (shapes >> m >> n >> k >> dtype >> ta >> tb) {
        ++line;
        const size_t bytes = ElementBytes(dtype);
        const size_t aCount = static_cast<size_t>(m) * static_cast<size_t>(k);
        const size_t bCount = static_cast<size_t>(k) * static_cast<size_t>(n);
        const size_t cCount = static_cast<size_t>(m) * static_cast<size_t>(n);
        std::vector<uint8_t> hostA(aCount * bytes), hostB(bCount * bytes), hostC(cCount * bytes, 0);
        for (int64_t i = 0; i < m; ++i) for (int64_t p = 0; p < k; ++p) {
            size_t index = ta ? static_cast<size_t>(p) * m + i : static_cast<size_t>(i) * k + p;
            float row = (i & 1) ? -1.0f : 1.0f;
            float reduction = (p & 1) ? 0.5f : 1.0f;
            Store(hostA, index, dtype, row * reduction);
        }
        for (int64_t p = 0; p < k; ++p) for (int64_t j = 0; j < n; ++j) {
            size_t index = tb ? static_cast<size_t>(j) * k + p : static_cast<size_t>(p) * n + j;
            float column = (j & 1) ? -1.0f : 1.0f;
            float reduction = (p % 3) ? 1.0f : 0.25f;
            Store(hostB, index, dtype, column * reduction);
        }

        float expectedMagnitude = 0.0f;
        for (int64_t p = 0; p < k; ++p) {
            expectedMagnitude += ((p & 1) ? 0.5f : 1.0f) * ((p % 3) ? 1.0f : 0.25f);
        }

        std::vector<int64_t> aLogical{m, k}, bLogical{k, n}, cLogical{m, n};
        std::vector<int64_t> aStorage = ta ? std::vector<int64_t>{k, m} : aLogical;
        std::vector<int64_t> bStorage = tb ? std::vector<int64_t>{n, k} : bLogical;
        std::vector<int64_t> aStrides = ta ? std::vector<int64_t>{1, m} : std::vector<int64_t>{k, 1};
        std::vector<int64_t> bStrides = tb ? std::vector<int64_t>{1, k} : std::vector<int64_t>{n, 1};
        std::vector<int64_t> cStrides{n, 1};
        TensorBuffer a, b, c;
        int stage = MakeTensor(hostA, aLogical, aStrides, aStorage, AclType(dtype), a);
        if (!stage) stage = MakeTensor(hostB, bLogical, bStrides, bStorage, AclType(dtype), b);
        if (!stage) stage = MakeTensor(hostC, cLogical, cStrides, cLogical, AclType(dtype), c);
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspace = nullptr;
        aclnnStatus status = ACL_SUCCESS;
        if (!stage) status = aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, c.tensor, 1, &workspaceSize, &executor);
        if (!stage && status == ACL_SUCCESS && workspaceSize) {
            if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) stage = 4;
        }
        if (!stage && status == ACL_SUCCESS) status = aclnnMatmul(workspace, workspaceSize, executor, stream);
        if (!stage && status == ACL_SUCCESS) status = aclrtSynchronizeStream(stream);
        if (!stage && status == ACL_SUCCESS) {
            status = aclrtMemcpy(hostC.data(), hostC.size(), c.device, hostC.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (stage || status != ACL_SUCCESS) {
            failure << "execution\t" << line << '\t' << m << '\t' << n << '\t' << k << '\t'
                    << dtype << '\t' << ta << '\t' << tb << "\t-1\t" << stage << '\t' << status << '\n';
            if (workspace) aclrtFree(workspace);
            Release(a); Release(b); Release(c);
            aclrtDestroyStream(stream); aclFinalize();
            return 4;
        }

        for (int64_t i = 0; i < m; ++i) for (int64_t j = 0; j < n; ++j) {
            size_t index = static_cast<size_t>(i) * n + j;
            float expected = Quantize(((i ^ j) & 1) ? -expectedMagnitude : expectedMagnitude, dtype);
            float actual = Load(hostC, index, dtype);
            float tolerance = dtype == "fp32" ? 1e-3f : 0.0f;
            if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
                failure << "numeric\t" << line << '\t' << m << '\t' << n << '\t' << k << '\t'
                        << dtype << '\t' << ta << '\t' << tb << '\t' << index << '\t'
                        << actual << '\t' << expected << '\n';
                if (workspace) aclrtFree(workspace);
                Release(a); Release(b); Release(c);
                aclrtDestroyStream(stream); aclFinalize();
                return 3;
            }
        }
        if (workspace) aclrtFree(workspace);
        Release(a); Release(b); Release(c);
    }
    aclrtDestroyStream(stream);
    aclFinalize();
    return 0;
}
CPP

if [[ ! -x "$RUNNER" || "$RUNNER_CPP" -nt "$RUNNER" ]]; then
    g++ -std=c++17 -O2 "$RUNNER_CPP" -o "$RUNNER" \
        -I"$CANN_ROOT/include" -I"$CUSTOM_VENDOR/op_api/include" \
        -L"$CUSTOM_VENDOR/op_api/lib" -L"$CANN_ROOT/lib64" \
        -Wl,-rpath,"$CUSTOM_VENDOR/op_api/lib" -Wl,-rpath,"$CANN_ROOT/lib64" \
        -lcust_opapi -lascendcl -lacl_op_compiler -lnnopbase -ldl \
        >"$LOGS/runner_build.log" 2>&1 || fail_infra '原始MatMulV3呼叫器建置失敗'
fi

SHAPES="$STATE/shapes_5000.txt"
python3 - "$SHAPES" <<'PY'
import itertools
import sys

out = sys.argv[1]
mn = [
    1, 2, 7, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65,
    95, 96, 97, 111, 112, 113, 127, 128, 129, 159, 160, 161,
    191, 192, 193, 207, 208, 209, 223, 224, 225, 239, 240, 241,
    255, 256, 257, 319, 320, 321, 383, 384, 385, 511, 512, 513,
    767, 768, 769, 1023, 1024, 1025, 1535, 1536, 1537, 2047,
    2048, 2049, 3071, 3072, 3073, 4095, 4096, 4097,
]
ks = [
    15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
    256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048,
    2049, 3071, 3072, 3073, 4095, 4096, 4097, 6143, 6144,
    6145, 8191, 8192, 8193, 12287, 12288, 12289, 16383, 16384,
    16385, 24575, 24576, 24577, 32767, 32768, 32769,
]
dtypes = ("fp16", "bf16", "fp32")
transposes = ((0, 0), (0, 1), (1, 0), (1, 1))

# The lists above are manually chosen around Cube/L1/L2 and transpose boundaries.
# A coprime traversal covers the product without random generation.
candidates = []
seen = set()
mn_pairs = len(mn) * len(mn)
for sequence in range(mn_pairs * len(ks) * len(dtypes) * 4):
    flat = (sequence * 5179) % mn_pairs
    m, n = mn[flat // len(mn)], mn[flat % len(mn)]
    k = ks[(sequence * 37 + flat) % len(ks)]
    dtype = dtypes[(sequence + flat) % len(dtypes)]
    ta, tb = transposes[(sequence * 3 + flat) % 4]
    key = (m, n, k, dtype, ta, tb)
    if key in seen:
        continue
    seen.add(key)
    element_bytes = 4 if dtype == "fp32" else 2
    tensor_elements = m * k + k * n + m * n
    memory = tensor_elements * element_bytes
    operations = 2 * m * n * k
    if tensor_elements > 200_000 or memory > 64 * 1024 * 1024 or operations > 100_000_000:
        continue
    candidates.append(key)
    if len(candidates) == 5000:
        break
if len(candidates) != 5000:
    raise SystemExit(f"only generated {len(candidates)} legal bounded shapes")
with open(out, "w", encoding="utf-8") as stream:
    for row in candidates:
        stream.write("%d %d %d %s %d %d\n" % row)
PY

validate_failure() {
    python3 - "$@" <<'PY'
import json
import struct
import sys

m, n, k = map(int, sys.argv[1:4])
dtype = sys.argv[4]
ta, tb = map(int, sys.argv[5:7])
index, actual, expected = sys.argv[7:10]

def desc(name, shape):
    names = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
    return {"name": name, "shape": shape, "ori_shape": shape, "format": "ND", "ori_format": "ND", "dtype": names[dtype]}

shape_a = [k, m] if ta else [m, k]
shape_b = [n, k] if tb else [k, n]
attrs = [
    {"name": "transpose_x1", "dtype": "bool", "value": bool(ta)},
    {"name": "transpose_x2", "dtype": "bool", "value": bool(tb)},
    {"name": "offset_x", "dtype": "int", "value": 0},
    {"name": "enable_hf32", "dtype": "bool", "value": False},
]
reason = None
try:
    from tbe.common.utils import op_tiling
    op_tiling._RT_BANK_CACHE = {}
    result = op_tiling.do_op_tiling(
        "MatMulV3", {}, [desc("x1", shape_a), desc("x2", shape_b), None, None],
        [desc("y", [m, n])], attrs=attrs,
    )
    blob = bytes(result["tiling_data"])
    cube = struct.unpack_from("<50i", blob, 0)
    l2 = struct.unpack_from("<5I", blob, 200)
    key = int(result["tiling_key"])
    suffix = key % 100000
    split = (suffix // 10) % 10
    full = (suffix // 100) % 10
    fix = (suffix // 10000) % 10
    q = {
        "usedCoreNum": cube[0], "singleCoreM": cube[5], "singleCoreN": cube[6], "singleCoreK": cube[7],
        "baseM": cube[8], "baseN": cube[9], "baseK": cube[10], "depthA1": cube[11], "depthB1": cube[12],
        "stepM": cube[13], "stepN": cube[14], "iterateOrder": cube[17], "stepKa": cube[26], "stepKb": cube[27],
        "dbL0A": cube[30], "dbL0B": cube[31], "dbL0C": cube[32],
        "l2MTileCnt": l2[0], "l2NTileCnt": l2[1], "l2MTileBlock": l2[2], "l2NTileBlock": l2[3],
        "l2IterateOrder": l2[4],
    }
    positive = ("usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK", "baseM", "baseN", "baseK",
                "depthA1", "depthB1", "stepM", "stepN", "stepKa", "stepKb", "dbL0A", "dbL0B", "dbL0C",
                "l2MTileCnt", "l2NTileCnt")
    if any(q[name] <= 0 for name in positive): reason = "NON_POSITIVE_TILING_FIELD"
    elif q["usedCoreNum"] > 20: reason = "USED_CORE_NUM_EXCEEDS_AIC"
    elif q["iterateOrder"] not in (0, 1) or q["l2IterateOrder"] not in (0, 1, 2): reason = "ITERATE_ORDER_OUT_OF_RANGE"
    elif any(q[name] not in (1, 2) for name in ("dbL0A", "dbL0B", "dbL0C")): reason = "DOUBLE_BUFFER_FLAG_OUT_OF_RANGE"
    else:
        in_bytes = 4 if dtype == "fp32" else 2
        k0 = 8 if dtype == "fp32" and not ta and tb else 16
        if q["baseM"] % 16 or q["baseN"] % 16 or q["baseK"] % k0: reason = "CUBE_BASE_ALIGNMENT"
        elif q["baseM"] * q["baseK"] * in_bytes * q["dbL0A"] > 64 * 1024: reason = "L0A_CAPACITY"
        elif q["baseN"] * q["baseK"] * in_bytes * q["dbL0B"] > 64 * 1024: reason = "L0B_CAPACITY"
        elif q["baseM"] * q["baseN"] * 4 * q["dbL0C"] > 256 * 1024: reason = "L0C_CAPACITY"
        elif q["depthA1"] % (q["stepM"] * q["stepKa"]): reason = "L1_A_DEPTH_DIVISIBILITY"
        elif q["depthB1"] % (q["stepN"] * q["stepKb"]): reason = "L1_B_DEPTH_DIVISIBILITY"
        elif q["baseM"] * q["baseK"] * q["depthA1"] * in_bytes + q["baseN"] * q["baseK"] * q["depthB1"] * in_bytes > 512 * 1024: reason = "L1_CAPACITY"
        elif q["stepKa"] % q["stepKb"] and q["stepKb"] % q["stepKa"]: reason = "L1_K_STEP_INCOMMENSURATE"
        elif split == 0 and full == 0:
            if q["singleCoreK"] != k: reason = "BASE_SINGLE_CORE_K_MUST_EQUAL_K"
            elif q["singleCoreM"] > q["baseM"]: reason = "BASE_SINGLE_CORE_M_EXCEEDS_BASE_M"
            elif q["singleCoreN"] > q["baseN"]: reason = "BASE_SINGLE_CORE_N_EXCEEDS_BASE_N"
            elif q["baseM"] > ((q["singleCoreM"] + 15) // 16) * 16: reason = "BASE_M_EXCEEDS_ALIGNED_SINGLE_CORE_M"
            elif q["baseN"] > ((q["singleCoreN"] + 15) // 16) * 16: reason = "BASE_N_EXCEEDS_ALIGNED_SINGLE_CORE_N"
            else:
                mt = (m + q["singleCoreM"] - 1) // q["singleCoreM"]
                nt = (n + q["singleCoreN"] - 1) // q["singleCoreN"]
                mb, nb = q["l2MTileBlock"], q["l2NTileBlock"]
                if mb == 0 or nb == 0:
                    if not (mb == nb == 0 and q["l2MTileCnt"] == q["l2NTileCnt"] == 1): reason = "L2_ZERO_BLOCK_CONTRACT"
                elif q["l2MTileCnt"] != (mt + mb - 1) // mb or q["l2NTileCnt"] != (nt + nb - 1) // nb: reason = "L2_TILE_COUNT_MISMATCH"
                elif not 1 <= mt - (q["l2MTileCnt"] - 1) * mb <= mb: reason = "L2_M_TAIL_OUT_OF_RANGE"
                elif not 1 <= nt - (q["l2NTileCnt"] - 1) * nb <= nb: reason = "L2_N_TAIL_OUT_OF_RANGE"
        elif split == 2:
            if q["singleCoreK"] != q["stepKa"] * q["baseK"]: reason = "SPLIT_K_SINGLE_CORE_K_MISMATCH"
            elif q["stepKa"] != q["stepKb"]: reason = "SPLIT_K_STEP_MISMATCH"
            elif q["singleCoreK"] >= k: reason = "SPLIT_K_NOT_SPLIT"
        elif split == 3:
            if (q["baseM"], q["baseN"], q["baseK"]) != (128, 128, 256 // in_bytes): reason = "DETERMINISTIC_SPLIT_K_BASE_CONTRACT"
            elif q["singleCoreK"] != 3 * q["baseK"]: reason = "DETERMINISTIC_SPLIT_K_CHUNK_CONTRACT"
        elif split == 0 and full == 1:
            if not (dtype == "fp32" and not ta and tb and m <= 16 and n > 16 and n <= 320 and k >= 4096): reason = "AL1_FULL_LOAD_DOMAIN"
        elif split == 0 and full == 2:
            if m <= 16 * max(k, n) or k > 256: reason = "BL1_FULL_LOAD_DOMAIN"
            elif fix == 2 and (dtype != "fp32" or ta or n > 192): reason = "BL1_VEC_NZ2ND_DOMAIN"
        else: reason = "UNKNOWN_TEMPLATE_MODE"
except Exception as exc:
    reason = "VALIDATOR_CALLBACK_FAILED:" + type(exc).__name__

shape = f"{m}x{n}x{k} {dtype} trans={ta}{tb}"
position = f"C[{index}]"
values = f"真實值={actual},計算值={expected}"
verdict = "攔截成功:" + reason if reason else "攔截失敗:validator未命中"
print(json.dumps({"shape": shape, "錯誤的位置": position + " " + values, "validator攔截結果": verdict}, ensure_ascii=False, separators=(",", ":")))
PY
}

REMAINING="$STATE/remaining.txt"
cp "$SHAPES" "$REMAINING"
while [[ -s "$REMAINING" ]]; do
    FAILURE="$STATE/failure.tsv"
    set +e
    "$RUNNER" "$REMAINING" "$FAILURE" >"$LOGS/run.log" 2>&1
    runner_rc=$?
    set -e
    if (( runner_rc == 0 )); then
        break
    fi
    [[ -s "$FAILURE" ]] || fail_infra 'NPU執行未留下錯誤位置'
    IFS=$'\t' read -r kind line m n k dtype ta tb index actual expected <"$FAILURE"
    if [[ "$kind" != numeric ]]; then
        printf '{"shape":"%sx%sx%s %s trans=%s%s","錯誤的位置":"NPU執行失敗(stage=%s,status=%s)","validator攔截結果":"未執行：沒有錯誤數值"}\n' \
            "$m" "$n" "$k" "$dtype" "$ta" "$tb" "$actual" "$expected"
        exit 1
    fi
    validate_failure "$m" "$n" "$k" "$dtype" "$ta" "$tb" "$index" "$actual" "$expected"
    tail -n "+$((line + 1))" "$REMAINING" >"$STATE/next_remaining.txt"
    mv "$STATE/next_remaining.txt" "$REMAINING"
done
