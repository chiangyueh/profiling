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
    printf '%s\n' '{"status":"failed","error":"CANN 8.1 environment is absent"}'
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

# Required by the unmodified upstream install rule in this MatMul-only tree.
mkdir -p "$ROOT/src/common/inc/kernel"
: >"$ROOT/src/common/inc/kernel/matmul_only_placeholder.h"

SOURCE_HASH="$({
    find "$ROOT/src/matmul/mat_mul_v3" "$ROOT/src/common" "$ROOT/cmake" "$ROOT/third_party" \
        -type f ! -name matmul_only_placeholder.h -print0
    printf '%s\0' "$ROOT/CMakeLists.txt" "$ROOT/CMakePresets.json"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
# The official source hash is unchanged, so reuse the completed prior build.
STATE="$ROOT/.benchmark_state/matmul_original_${SOURCE_HASH:0:20}"
BUILD="$STATE/build"
INSTALL="$STATE/install"
LOGS="$STATE/logs"
CASES="$STATE/official_golden_cases"
mkdir -p "$BUILD" "$INSTALL" "$LOGS" "$CASES"

fail_infra() {
    python3 - "$1" <<'PY'
import json, sys
print(json.dumps({"status":"failed", "error":sys.argv[1]}, separators=(",", ":")))
PY
    exit 1
}

if [[ ! -f "$INSTALL/.complete" ]]; then
    printf '%s\n' 'MATMUL_ORIGINAL_512_BUILD begin'
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
    printf '%s\n' 'MATMUL_ORIGINAL_512_BUILD passed'
else
    printf '%s\n' 'MATMUL_ORIGINAL_512_BUILD reused'
fi

CUSTOM_VENDOR="$INSTALL/packages/vendors/customize"
[[ -d "$CUSTOM_VENDOR" ]] || fail_infra 'private MatMulV3 package is absent'
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_VENDOR"
export LD_LIBRARY_PATH="$CUSTOM_VENDOR/op_api/lib:$CUSTOM_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -m):$LD_LIBRARY_PATH"
if [[ -f "$CUSTOM_VENDOR/bin/set_env.bash" ]]; then
    source "$CUSTOM_VENDOR/bin/set_env.bash" >"$LOGS/private_env.log" 2>&1
fi

OFFICIAL_EXAMPLE="$ROOT/src/matmul/mat_mul_v3/examples/AclNNInvocationNaive"
OFFICIAL_GENERATOR="$OFFICIAL_EXAMPLE/gen_data.py"
OFFICIAL_VERIFY="$OFFICIAL_EXAMPLE/verify_result.py"
EXPECTED_GENERATOR_SHA256="bb047fa1f8090493a3ad164107b83ad6fb1df2fb748badc0e1715f0722d0d7d4"
EXPECTED_VERIFY_SHA256="763fcd171b2e9704e5957ba9ae7f941c632b8fb1357e699ae21f0788671104de"
[[ "$(sha256sum "$OFFICIAL_GENERATOR" | cut -d' ' -f1)" == "$EXPECTED_GENERATOR_SHA256" ]] \
    || fail_infra 'official gen_data.py differs from Gitee master'
[[ "$(sha256sum "$OFFICIAL_VERIFY" | cut -d' ' -f1)" == "$EXPECTED_VERIFY_SHA256" ]] \
    || fail_infra 'official verify_result.py differs from Gitee master'

SHAPES="$STATE/official_golden_shape_512.txt"
python3 - "$OFFICIAL_GENERATOR" "$SHAPES" "$CASES" <<'PY'
import hashlib
import os
import pathlib
import sys

generator = pathlib.Path(sys.argv[1])
shape_path = pathlib.Path(sys.argv[2])
case_root = pathlib.Path(sys.argv[3])
source = generator.read_text(encoding="utf-8")
if hashlib.sha256(generator.read_bytes()).hexdigest() != "bb047fa1f8090493a3ad164107b83ad6fb1df2fb748badc0e1715f0722d0d7d4":
    raise SystemExit("official generator hash mismatch")

shapes = [(512, 512, 512)]

shape_path.write_text("".join(f"{m} {n} {k}\n" for m, n, k in shapes), encoding="utf-8")
original_lines = source.splitlines()
previous_cwd = os.getcwd()
for index, (m, n, k) in enumerate(shapes, 1):
    replacements = {
        "    self_shape = [16, 32]": f"    self_shape = [{m}, {k}]",
        "    mat2_shape = [32, 16]": f"    mat2_shape = [{k}, {n}]",
        "    output_shape = [16, 16]": f"    output_shape = [{m}, {n}]",
    }
    patched = source
    expected_changed = []
    for line_no, (old, new) in zip((18, 19, 20), replacements.items()):
        if patched.count(old) != 1:
            raise SystemExit(f"official generator shape anchor mismatch: {old}")
        patched = patched.replace(old, new)
        if old != new:
            expected_changed.append(line_no)
    changed = [line_no for line_no, (old, new) in enumerate(zip(original_lines, patched.splitlines()), 1) if old != new]
    if changed != expected_changed:
        raise SystemExit(f"generator changed outside the three shape lines: {changed}")
    if "    golden = self @ mat2" not in patched:
        raise SystemExit("official NumPy golden expression is absent")
    case_dir = case_root / f"case_{index}"
    case_dir.mkdir(parents=True, exist_ok=True)
    namespace = {"__name__": "official_gen_data_shape_only"}
    exec(compile(patched, str(generator), "exec"), namespace)
    try:
        os.chdir(case_dir)
        namespace["gen_golden_data_simple"]()
    finally:
        os.chdir(previous_cwd)
PY

RUNNER_CPP="$STATE/matmul_official_golden_512.cpp"
RUNNER="$STATE/matmul_official_golden_512"
cat >"$RUNNER_CPP" <<'CPP'
#include <acl/acl.h>
#include <aclnn_matmul.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

static bool ReadBinary(const std::string &path, std::vector<uint16_t> &data) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(uint16_t)));
    return stream.gcount() == static_cast<std::streamsize>(data.size() * sizeof(uint16_t));
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

static int MakeTensor(const std::vector<uint16_t> &host, const std::vector<int64_t> &shape, TensorBuffer &result) {
    size_t bytes = host.size() * sizeof(uint16_t);
    if (aclrtMalloc(&result.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 1;
    if (aclrtMemcpy(result.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return 2;
    result.tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, nullptr, 0,
                                    ACL_FORMAT_ND, shape.data(), shape.size(), result.device);
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
    const std::string caseRoot = argv[3];
    if (!shapes || !report) return 2;
    if (aclInit(nullptr) != ACL_SUCCESS) return 10;
    if (aclrtSetDevice(0) != ACL_SUCCESS) return 11;
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) return 12;

    int64_t m, n, k;
    size_t index = 0;
    while (shapes >> m >> n >> k) {
        ++index;
        const std::string caseDir = caseRoot + "/case_" + std::to_string(index);
        std::vector<uint16_t> hostA(static_cast<size_t>(m) * k);
        std::vector<uint16_t> hostB(static_cast<size_t>(k) * n);
        std::vector<uint16_t> hostC(static_cast<size_t>(m) * n, 0);
        if (!ReadBinary(caseDir + "/input/input_self.bin", hostA) ||
            !ReadBinary(caseDir + "/input/input_mat2.bin", hostB)) {
            report << index << '\t' << m << '\t' << n << '\t' << k << "\tinput_read_failed\t0\t0\n";
            continue;
        }
        TensorBuffer a, b, c;
        int stage = MakeTensor(hostA, {m, k}, a);
        if (!stage) stage = MakeTensor(hostB, {k, n}, b);
        if (!stage) stage = MakeTensor(hostC, {m, n}, c);
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
        bool outputOk = false;
        if (!stage && status == ACL_SUCCESS) {
            outputOk = WriteBinary(caseDir + "/output/output.bin", hostC);
        }
        report << index << '\t' << m << '\t' << n << '\t' << k << '\t'
               << ((!stage && status == ACL_SUCCESS && outputOk) ? "passed" : "execution_failed")
               << '\t' << stage << '\t' << status << '\n';
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
        >"$LOGS/runner_build.log" 2>&1 || fail_infra 'official-golden MatMulV3 runner build failed'
fi

REPORT="$STATE/official_golden_npu_report.tsv"
printf '%s\n' 'MATMUL_ORIGINAL_512_RUN begin'
if ! "$RUNNER" "$SHAPES" "$REPORT" "$CASES" >"$LOGS/runner.log" 2>&1; then
    fail_infra 'MatMulV3 NPU runner failed before official validation'
fi
printf '%s\n' 'MATMUL_ORIGINAL_512_RUN completed'

python3 - "$REPORT" "$CASES" "$OFFICIAL_VERIFY" <<'PY'
import contextlib
import hashlib
import importlib.util
import io
import json
import numpy as np
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
case_root = pathlib.Path(sys.argv[2])
verify_path = pathlib.Path(sys.argv[3])
if hashlib.sha256(verify_path.read_bytes()).hexdigest() != "763fcd171b2e9704e5957ba9ae7f941c632b8fb1357e699ae21f0788671104de":
    raise SystemExit("official verifier hash mismatch")
spec = importlib.util.spec_from_file_location("official_matmul_verify_result", verify_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

failures = []
passed = 0
rows = report_path.read_text(encoding="utf-8").splitlines()
for raw in rows:
    fields = raw.split("\t")
    index, m, n, k = map(int, fields[:4])
    shape = f"{m}x{n}x{k} fp16"
    if fields[4] != "passed":
        failures.append({"shape":shape, "official_validator":"not_run",
                         "error":f"NPU execution failed: stage={fields[5]}, status={fields[6]}"})
        continue
    case_dir = case_root / f"case_{index}"
    output = case_dir / "output/output.bin"
    golden = case_dir / "output/golden.bin"
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = bool(module.verify_result(str(output), str(golden)))
    if result:
        passed += 1
    else:
        actual = np.fromfile(output, dtype=np.float16)
        expected = np.fromfile(golden, dtype=np.float16)
        mismatch = np.flatnonzero(actual != expected)
        first = int(mismatch[0]) if mismatch.size else -1
        failures.append({
            "shape":shape,
            "official_validator":"failed",
            "first_mismatch":first,
            "position":f"C[{first // n},{first % n}]" if first >= 0 else None,
            "expected":float(expected[first]) if first >= 0 else None,
            "actual":float(actual[first]) if first >= 0 else None,
            "mismatch_count":int(mismatch.size),
            "element_count":int(expected.size),
            "mismatch_ratio":float(mismatch.size / expected.size),
        })

for failure in failures:
    print(json.dumps(failure, ensure_ascii=False, separators=(",", ":")))
print(json.dumps({"status":"passed" if not failures and passed == 1 else "failed",
                  "shapes":len(rows), "official_validator_passed":passed,
                  "golden":"unmodified official gen_data.py except shape lines",
                  "validator":"unmodified official verify_result.py"},
                 ensure_ascii=False, separators=(",", ":")))
raise SystemExit(0 if not failures and passed == 1 else 1)
PY
