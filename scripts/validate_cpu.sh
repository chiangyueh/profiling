#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"

SOC="${ASCENDC_SOC_VERSION:-Ascend910B3}"
export ASCENDC_SOC_VERSION="$SOC"
export SOC_VERSION="$SOC"
mkdir -p "$ROOT/build"
WORK_DIR="$(mktemp -d "$ROOT/build/tiling_validate.XXXXXX")"
trap 'find "$WORK_DIR" -mindepth 1 -delete; rmdir "$WORK_DIR" 2>/dev/null || true' EXIT

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    BUILD_JOBS="${BUILD_JOBS:-1}" ASCENDC_SOC_VERSION="$SOC" \
        SOC_VERSION="$SOC" ./scripts/build_all.sh
fi

ARCH="$(uname -m)"
CANN_INCLUDE="$CANN_ROOT/${ARCH}-linux/include"
g++ -std=c++17 -O2 -Wall -Wextra \
    -I"$ROOT/host/include" -I"$ROOT/compat" -I"$CANN_INCLUDE" \
    "$ROOT/tests/proxy_model_test.cpp" \
    "$ROOT/host/src/proxy_model.cpp" \
    "$ROOT/host/src/search_types.cpp" \
    -o "$WORK_DIR/proxy_model_test"
"$WORK_DIR/proxy_model_test"
python3 "$ROOT/tests/refine_model_test.py"
python3 "$ROOT/tests/rank_results_test.py"

GENERAL_CSV="$WORK_DIR/candidates.csv"
GENERAL_LOG="$WORK_DIR/search.log"
SEARCH_SCOPE=all_templates_validation \
    BEAM_WIDTH=16 TABU_ITERS=8 LNS_ROUNDS=2 TOP_K=4 \
    SEARCH_OUTPUT="$GENERAL_CSV" \
    SEARCH_ALL_OUTPUT="$WORK_DIR/all.csv" \
    SEARCH_TILING_DIR="$WORK_DIR/tilings" \
    ./scripts/run_search.sh config/workloads_validation.csv >"$GENERAL_LOG"

PLATFORM_LINE="$(sed -n '/^CANN platform=/{p;q;}' "$GENERAL_LOG")"
platform_field() {
    printf '%s\n' "$PLATFORM_LINE" |
        sed -n "s/.*[[:space:]]$1=\\([0-9][0-9.]*\\).*/\\1/p"
}
AIC_CORES="$(platform_field cores)"
L0A_BYTES="$(platform_field L0A)"
L0B_BYTES="$(platform_field L0B)"
L0C_BYTES="$(platform_field L0C)"
L1_BYTES="$(platform_field L1)"
test -n "$AIC_CORES"

python3 tools/profile_official_tilings.py \
    --runner build/official_matmul_runner \
    --bank-probe build/tiling_bank_probe \
    --candidates "$GENERAL_CSV" \
    --workloads config/workloads_validation.csv \
    --custom-output "$WORK_DIR/custom.csv" \
    --custom-samples-output "$WORK_DIR/custom_samples.csv" \
    --official-output "$WORK_DIR/official.csv" \
    --official-samples-output "$WORK_DIR/official_samples.csv" \
    --cann-root "$CANN_ROOT" \
    --soc "$SOC" \
    --aic-cores "$AIC_CORES" \
    --l0a-bytes "$L0A_BYTES" \
    --l0b-bytes "$L0B_BYTES" \
    --l0c-bytes "$L0C_BYTES" \
    --l1-bytes "$L1_BYTES" \
    --rank-limit 4 \
    --validate-only

python3 - \
    "$GENERAL_CSV" "$AIC_CORES" "$L0A_BYTES" "$L0B_BYTES" \
    "$L0C_BYTES" "$L1_BYTES" <<'PY'
import csv
import sys
from collections import Counter

sys.path.insert(0, "tools")
import profile_official_tilings
import refine_matmul_v3_candidates as refine

with open(sys.argv[1], newline="", encoding="utf-8") as source:
    general = list(csv.DictReader(source))

hardware = refine.Hardware(
    aic_cores=int(sys.argv[2]),
    l0a_bytes=int(sys.argv[3]),
    l0b_bytes=int(sys.argv[4]),
    l0c_bytes=int(sys.argv[5]),
    l1_bytes=int(sys.argv[6]),
    l2_bytes=192 * 1024 * 1024,
    l2_bytes_per_cycle_per_core=110.0,
    hbm_bytes_per_cycle_per_core=32.0,
)
ids = {row["workload_id"] for row in general}
assert "validation_skinny" in ids
assert "validation_skinny_n" in ids
assert "validation_trans_b" in ids
assert "validation_fp32_nt_splitk" in ids
assert "validation_int8" not in ids
assert all(row["valid"] == "1" for row in general)
assert len({
    (row["workload_id"], row["tiling_signature"]) for row in general
}) == len(general)
controls = [
    row for row in general
    if row.get("candidate_role") == "bank_seed_control"
]
assert len(controls) == 9
assert all(row["rank"] == "0" for row in controls)
assert all(len(row["callback_tiling_sha256"]) == 64 for row in controls)
skinny_control = next(
    row for row in controls
    if row["workload_id"] == "validation_skinny_n"
)
assert "baseBD:" in skinny_control["callback_derived_diff_vs_default"]
for row in general:
    if row.get("candidate_role") != "searched":
        continue
    assert len(row["callback_tiling_sha256"]) == 64
    assert float(row["search_model_cycles"]) > 0
    assert float(row["search_model_ratio_vs_bank_seed"]) > 0
    knowledge = profile_official_tilings.make_knowledge(row)
    workload = refine.Workload(
        workload_id=row["workload_id"],
        m=int(row["m"]),
        n=int(row["n"]),
        k=int(row["k"]),
        dtype=row["dtype"],
        trans_a=bool(int(row["trans_a"])),
        trans_b=bool(int(row["trans_b"])),
        max_cores=int(row["max_cores"]),
    )
    assert refine.hard_legal(workload, knowledge, hardware)
    assert knowledge["l2MTileCnt"] >= 1
    assert knowledge["l2NTileCnt"] >= 1
    if knowledge["l2MTileBlock"] == 0:
        assert row["execution_mode"] in {
            "bl1_full_load_fixpipe", "bl1_full_load_vec_nz2nd"
        }
        assert knowledge["l2NTileBlock"] == 0
    else:
        assert knowledge["l2NTileBlock"] >= 1
    suffix = int(row["callback_kernel_suffix"])
    assert suffix in refine.CANN81_KERNEL_VARIANTS
    assert row["callback_kernel_family"] == refine.template_name(knowledge)
    assert row["search_template"] == row["callback_kernel_family"]
    assert row["callback_kernel_variant"] == refine.kernel_variant(
        int(row["callback_tiling_key"])
    )
    if row["callback_kernel_family"] == "BASE":
        assert knowledge["singleCoreK"] == workload.k
        assert knowledge["singleCoreM"] <= knowledge["baseM"]
        assert knowledge["singleCoreN"] <= knowledge["baseN"]
modes = Counter(row["execution_mode"] for row in general)
print(
    f"tiling_validation passed general={len(general)} "
    + " ".join(f"{mode}={count}" for mode, count in sorted(modes.items()))
)
PY

TEMPLATE_CSV="$WORK_DIR/template_candidates.csv"
SEARCH_SCOPE=all_templates_validation \
    BEAM_WIDTH=16 TABU_ITERS=8 LNS_ROUNDS=2 TOP_K=8 \
    SEARCH_OUTPUT="$TEMPLATE_CSV" \
    SEARCH_ALL_OUTPUT="$WORK_DIR/template_all.csv" \
    SEARCH_TILING_DIR="$WORK_DIR/template_tilings" \
    ./scripts/run_search.sh \
    config/workloads_template_validation.csv \
    >"$WORK_DIR/template_search.log"

python3 tools/profile_official_tilings.py \
    --runner build/official_matmul_runner \
    --bank-probe build/tiling_bank_probe \
    --candidates "$TEMPLATE_CSV" \
    --workloads config/workloads_template_validation.csv \
    --custom-output "$WORK_DIR/template_custom.csv" \
    --custom-samples-output "$WORK_DIR/template_custom_samples.csv" \
    --official-output "$WORK_DIR/template_official.csv" \
    --official-samples-output "$WORK_DIR/template_official_samples.csv" \
    --cann-root "$CANN_ROOT" \
    --soc "$SOC" \
    --aic-cores "$AIC_CORES" \
    --l0a-bytes "$L0A_BYTES" \
    --l0b-bytes "$L0B_BYTES" \
    --l0c-bytes "$L0C_BYTES" \
    --l1-bytes "$L1_BYTES" \
    --rank-limit 8 \
    --validate-only

python3 - "$TEMPLATE_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as source:
    # Template coverage includes the exact RuntimeKb controls. Some template
    # families are the official choice for a workload but do not rank in that
    # workload's searched Top-K; excluding controls made suffix coverage
    # depend on proxy ranking instead of the callback contract.
    rows = list(csv.DictReader(source))

expected_suffixes = {
    0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201
}
expected_families = {
    "BASE",
    "SINGLE_CORE_SPLIT_K",
    "DETERMINISTIC_SPLIT_K",
    "AL1_FULL_LOAD",
    "BL1_FULL_LOAD",
    "BL1_FULL_LOAD_FIXPIPE",
    "BL1_FULL_LOAD_VEC_NZ2ND",
}
observed_suffixes = {int(row["callback_kernel_suffix"]) for row in rows}
observed_families = {row["callback_kernel_family"] for row in rows}
assert observed_suffixes == expected_suffixes, (
    observed_suffixes, expected_suffixes - observed_suffixes
)
assert observed_families == expected_families
assert all(
    row["search_template"] == row["callback_kernel_family"]
    for row in rows
)
print(
    "template_contract_validation passed "
    f"candidates={len(rows)} suffixes={len(observed_suffixes)} "
    f"families={len(observed_families)}"
)
PY

./build/official_matmul_runner \
    --candidates config/workloads_validation.csv \
    --validate-input
