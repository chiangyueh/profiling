#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 -d PHYSICAL_NPU_ID" >&2
}

device=""
while getopts ":d:h" option; do
    case "$option" in
        d) device="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
if [[ -z "$device" || ! "$device" =~ ^[0-9]+$ ]]; then
    usage
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cann_root="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
kernel_dir="$cann_root/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3"

if [[ ! -f "$kernel_dir/mat_mul_base_kernel.h" || ! -f "$kernel_dir/mat_mul_base_block.h" ]]; then
    echo "fatal: installed CANN MatMulV3 BASE source is missing: $kernel_dir" >&2
    exit 1
fi
probe_id="$(sha256sum \
    "$repo_root/matmul_validator_probe/CMakeLists.txt" \
    "$repo_root/matmul_validator_probe/kernel.cpp" \
    "$repo_root/matmul_validator_probe/runner.cpp" \
    "$repo_root/matmul_validator_probe/mat_mul_v3_tiling_data.h" \
    "$kernel_dir/mat_mul_base_kernel.h" \
    "$kernel_dir/mat_mul_base_block.h" | sha256sum | cut -c1-16)"
state_dir="$repo_root/.benchmark_state/matmul_validator_probe/$probe_id"
build_dir="$state_dir/build"
log_dir="$repo_root/results/matmul_validator_probe"
build_log="$state_dir/build.log"
run_log="$log_dir/latest.log"
mkdir -p "$build_dir" "$log_dir"
exec > >(tee "$run_log") 2>&1

# These variables affect this script and its child only.  No toolkit, OPP,
# driver, device, or other user's process is modified or reset.
export ASCEND_RT_VISIBLE_DEVICES="$device"
export ASCEND_HOME_PATH="$cann_root"
export LD_LIBRARY_PATH="$cann_root/$(uname -m)-linux/lib64:$cann_root/lib64:${LD_LIBRARY_PATH:-}"

echo "MATMUL_VALIDATOR_PROBE_BUILD begin"
if ! cmake \
    -S "$repo_root/matmul_validator_probe" \
    -B "$build_dir" \
    -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    -DMATMUL_V3_KERNEL_DIR="$kernel_dir" \
    -DSOC_VERSION=Ascend910B3 \
    -DCMAKE_BUILD_TYPE=Release >"$build_log" 2>&1; then
    tail -n 40 "$build_log" >&2
    exit 1
fi
if ! cmake --build "$build_dir" --parallel 1 >>"$build_log" 2>&1; then
    tail -n 60 "$build_log" >&2
    exit 1
fi
echo "MATMUL_VALIDATOR_PROBE_BUILD passed"
echo "MATMUL_VALIDATOR_PROBE_SOURCE kernel=$kernel_dir/mat_mul_base_kernel.h block=$kernel_dir/mat_mul_base_block.h"

"$build_dir/matmul_validator_probe"
echo "MATMUL_VALIDATOR_PROBE_LOG $run_log"
