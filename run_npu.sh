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
    printf '%s\n' '{"shape":"","official_validator":"not_run","current_validator":"not_run","error":"CANN 8.1 environment is absent"}'
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

# The upstream install rule requires this directory even in a MatMul-only tree.
mkdir -p "$ROOT/src/common/inc/kernel"
: >"$ROOT/src/common/inc/kernel/matmul_only_placeholder.h"

SOURCE_HASH="$({
    find "$ROOT/src/matmul/mat_mul_v3" "$ROOT/src/common" "$ROOT/cmake" "$ROOT/third_party" \
        -type f ! -name matmul_only_placeholder.h -print0
    printf '%s\0' "$ROOT/CMakeLists.txt" "$ROOT/CMakePresets.json"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
# Reuse the already completed official MatMulV3 build from the preceding
# numeric campaign; run_npu.sh is deliberately excluded from SOURCE_HASH.
STATE="$ROOT/.benchmark_state/matmul_original_${SOURCE_HASH:0:20}"
BUILD="$STATE/build"
INSTALL="$STATE/install"
LOGS="$STATE/logs"
ARTIFACTS="$STATE/artifacts"
mkdir -p "$BUILD" "$INSTALL" "$LOGS" "$ARTIFACTS"

fail_infra() {
    python3 - "$1" <<'PY'
import json, sys
print(json.dumps({"shape":"", "official_validator":"not_run", "current_validator":"not_run", "error":sys.argv[1]}, separators=(",", ":")))
PY
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
        || fail_infra 'official MatMulV3 configure failed'
    cmake --build "$BUILD" --target install -j1 >"$LOGS/build.log" 2>&1 \
        || fail_infra 'official MatMulV3 build failed'
    touch "$INSTALL/.complete"
fi

CUSTOM_VENDOR="$INSTALL/packages/vendors/customize"
[[ -d "$CUSTOM_VENDOR" ]] || fail_infra 'private MatMulV3 package is absent'
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_VENDOR"
export LD_LIBRARY_PATH="$CUSTOM_VENDOR/op_api/lib:$CUSTOM_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -m):$LD_LIBRARY_PATH"
if [[ -f "$CUSTOM_VENDOR/bin/set_env.bash" ]]; then
    source "$CUSTOM_VENDOR/bin/set_env.bash" >"$LOGS/private_env.log" 2>&1
fi

RUNNER_CPP="$STATE/matmul_dual_numeric.cpp"
RUNNER="$STATE/matmul_dual_numeric"
cat >"$RUNNER_CPP" <<'CPP'
#include <acl/acl.h>
#include <aclnn_matmul.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
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
        ++exponent;
        if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

static bool WriteBinary(const std::string &path, const std::vector<uint16_t> &data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(uint16_t)));
    return static_cast<bool>(stream);
}

struct TensorBuffer {
    aclTensor *tensor = nullptr;
    void *device = nullptr;
};

static int MakeTensor(const std::vector<uint16_t> &host, const std::vector<int64_t> &logical,
                      const std::vector<int64_t> &strides, const std::vector<int64_t> &storage,
                      TensorBuffer &result) {
    size_t bytes = host.size() * sizeof(uint16_t);
    if (aclrtMalloc(&result.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 1;
    if (aclrtMemcpy(result.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return 2;
    result.tensor = aclCreateTensor(logical.data(), logical.size(), ACL_FLOAT16, strides.data(), 0,
                                    ACL_FORMAT_ND, storage.data(), storage.size(), result.device);
    return result.tensor ? 0 : 3;
}

static void Release(TensorBuffer &buffer) {
    if (buffer.tensor) aclDestroyTensor(buffer.tensor);
    if (buffer.device) aclrtFree(buffer.device);
    buffer.tensor = nullptr;
    buffer.device = nullptr;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    std::ifstream shapes(argv[1]);
    std::ofstream report(argv[2], std::ios::trunc);
    const std::string artifactDir = argv[3];
    if (!shapes || !report) return 2;
    if (aclInit(nullptr) != ACL_SUCCESS) return 10;
    if (aclrtSetDevice(0) != ACL_SUCCESS) return 11;
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) return 12;

    int64_t m, n, k;
    int ta, tb;
    size_t line = 0;
    while (shapes >> m >> n >> k >> ta >> tb) {
        ++line;
        const size_t aCount = static_cast<size_t>(m) * k;
        const size_t bCount = static_cast<size_t>(k) * n;
        const size_t cCount = static_cast<size_t>(m) * n;
        std::vector<uint16_t> hostA(aCount), hostB(bCount), hostC(cCount, 0);
        for (int64_t i = 0; i < m; ++i) for (int64_t p = 0; p < k; ++p) {
            size_t index = ta ? static_cast<size_t>(p) * m + i : static_cast<size_t>(i) * k + p;
            hostA[index] = FloatToHalf(((i & 1) ? -1.0f : 1.0f) * ((p & 1) ? 0.5f : 1.0f));
        }
        for (int64_t p = 0; p < k; ++p) for (int64_t j = 0; j < n; ++j) {
            size_t index = tb ? static_cast<size_t>(j) * k + p : static_cast<size_t>(p) * n + j;
            hostB[index] = FloatToHalf(((j & 1) ? -1.0f : 1.0f) * ((p % 3) ? 1.0f : 0.25f));
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
        int stage = MakeTensor(hostA, aLogical, aStrides, aStorage, a);
        if (!stage) stage = MakeTensor(hostB, bLogical, bStrides, bStorage, b);
        if (!stage) stage = MakeTensor(hostC, cLogical, cStrides, cLogical, c);
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspace = nullptr;
        aclnnStatus status = ACL_SUCCESS;
        if (!stage) status = aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, c.tensor, 1, &workspaceSize, &executor);
        if (!stage && status == ACL_SUCCESS && workspaceSize &&
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) stage = 4;
        if (!stage && status == ACL_SUCCESS) status = aclnnMatmul(workspace, workspaceSize, executor, stream);
        if (!stage && status == ACL_SUCCESS) status = aclrtSynchronizeStream(stream);
        if (!stage && status == ACL_SUCCESS) {
            status = aclrtMemcpy(hostC.data(), hostC.size() * sizeof(uint16_t), c.device,
                                 hostC.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (stage || status != ACL_SUCCESS) {
            report << line << '\t' << m << '\t' << n << '\t' << k << '\t' << ta << '\t' << tb
                   << "\texecution_failed\t" << stage << '\t' << status << '\n';
            if (workspace) aclrtFree(workspace);
            Release(a); Release(b); Release(c);
            continue;
        }

        bool analyticPass = true;
        size_t firstMismatch = cCount;
        for (int64_t i = 0; i < m; ++i) for (int64_t j = 0; j < n; ++j) {
            size_t index = static_cast<size_t>(i) * n + j;
            uint16_t expected = FloatToHalf(((i ^ j) & 1) ? -expectedMagnitude : expectedMagnitude);
            if (hostC[index] != expected && analyticPass) {
                analyticPass = false;
                firstMismatch = index;
            }
        }
        const std::string prefix = artifactDir + "/case_" + std::to_string(line);
        bool filesOk = WriteBinary(prefix + "_a.bin", hostA) && WriteBinary(prefix + "_b.bin", hostB) &&
                       WriteBinary(prefix + "_c.bin", hostC);
        report << line << '\t' << m << '\t' << n << '\t' << k << '\t' << ta << '\t' << tb << '\t'
               << (filesOk ? (analyticPass ? "passed" : "failed") : "artifact_write_failed") << '\t'
               << (firstMismatch == cCount ? -1 : static_cast<int64_t>(firstMismatch)) << "\t0\n";
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
        >"$LOGS/runner_build.log" 2>&1 || fail_infra 'MatMulV3 dual-validator runner build failed'
fi

SHAPES="$STATE/shapes.txt"
cat >"$SHAPES" <<'EOF'
16 16 32 0 0
17 31 33 0 0
31 17 65 0 1
64 65 127 1 0
97 113 129 1 1
257 193 257 0 0
EOF

REPORT="$STATE/npu_report.tsv"
rm -f "$ARTIFACTS"/case_*.bin "$REPORT"
if ! "$RUNNER" "$SHAPES" "$REPORT" "$ARTIFACTS" >"$LOGS/runner.log" 2>&1; then
    fail_infra 'MatMulV3 NPU runner failed before validation'
fi

OFFICIAL_VERIFY="$ROOT/src/matmul/mat_mul_v3/examples/AclNNInvocationNaive/verify_result.py"
[[ -f "$OFFICIAL_VERIFY" ]] || fail_infra 'official verify_result.py is absent'

python3 - "$REPORT" "$ARTIFACTS" "$OFFICIAL_VERIFY" <<'PY'
import json
import pathlib
import subprocess
import sys

import numpy as np

report_path = pathlib.Path(sys.argv[1])
artifacts = pathlib.Path(sys.argv[2])
official_verify = pathlib.Path(sys.argv[3])
rows = []
all_agree = True
all_pass = True

for raw in report_path.read_text(encoding="utf-8").splitlines():
    fields = raw.split("\t")
    line, m, n, k, ta, tb = map(int, fields[:6])
    current = fields[6]
    shape = f"{m}x{n}x{k} fp16 trans={ta}{tb}"
    if current == "execution_failed":
        row = {"shape":shape, "official_validator":"not_run", "current_validator":"not_run", "validators_agree":False,
               "error":f"NPU execution failed: stage={fields[7]}, status={fields[8]}"}
        rows.append(row)
        all_agree = False
        all_pass = False
        continue
    if current == "artifact_write_failed":
        rows.append({"shape":shape, "official_validator":"not_run", "current_validator":"not_run",
                     "validators_agree":False, "error":"artifact write failed"})
        all_agree = False
        all_pass = False
        continue

    prefix = artifacts / f"case_{line}"
    a_storage = np.fromfile(str(prefix) + "_a.bin", dtype=np.float16).reshape((k, m) if ta else (m, k))
    b_storage = np.fromfile(str(prefix) + "_b.bin", dtype=np.float16).reshape((n, k) if tb else (k, n))
    a = a_storage.T if ta else a_storage
    b = b_storage.T if tb else b_storage
    golden = a @ b
    golden_path = pathlib.Path(str(prefix) + "_golden.bin")
    golden.astype(np.float16, copy=False).tofile(golden_path)
    output_path = pathlib.Path(str(prefix) + "_c.bin")
    process = subprocess.run([sys.executable, str(official_verify), str(output_path), str(golden_path)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    official = "passed" if process.returncode == 0 and "test pass" in process.stdout else "failed"
    current_status = "passed" if current == "passed" else "failed"
    agree = official == current_status
    all_agree = all_agree and agree
    all_pass = all_pass and official == "passed" and current_status == "passed"
    rows.append({"shape":shape, "official_validator":official, "current_validator":current_status,
                 "validators_agree":agree})

for row in rows:
    print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
print(json.dumps({"summary":{"cases":len(rows), "validators_agree":all_agree,
                             "both_validators_passed_all_cases":all_pass}}, separators=(",", ":")))
raise SystemExit(0 if all_agree and all_pass else 1)
PY
