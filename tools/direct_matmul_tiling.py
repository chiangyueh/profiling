#!/usr/bin/env python3
"""Materialize solver rows as exact CANN 8.1 MatMulV3 tiling buffers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npu_cost_model import (  # noqa: E402
    MemorySpace,
    Resource,
    ascend_910b3,
    cann81_matmul_effective_l1_bytes,
    source_kernel_suffix,
    validate_cann_tiling,
)
from npu_cost_model.matmul_layout import source_layout_conversion  # noqa: E402


KNOWLEDGE_COLUMNS = {
    "usedCoreNum": "used_core_num",
    "singleCoreM": "single_core_m",
    "singleCoreN": "single_core_n",
    "singleCoreK": "single_core_k",
    "baseM": "base_m",
    "baseN": "base_n",
    "baseK": "base_k",
    "depthA1": "depth_a1",
    "depthB1": "depth_b1",
    "stepM": "step_m",
    "stepN": "step_n",
    "iterateOrder": "iterate_order",
    "stepKa": "step_ka",
    "stepKb": "step_kb",
    "dbL0A": "db_l0a",
    "dbL0B": "db_l0b",
    "dbL0C": "db_l0c",
    "l2MTileCnt": "bank_l2_m_tile_count",
    "l2NTileCnt": "bank_l2_n_tile_count",
    "l2MTileBlock": "bank_l2_m_tile_block",
    "l2NTileBlock": "bank_l2_n_tile_block",
    "l2IterateOrder": "bank_l2_iterate_order",
    "tilingEnable": "bank_tiling_enable",
}

ABI_BYTES = 272
UB_BYTES = 196_352
RPC_WORKSPACE_BYTES = 20 * 1024 * 1024
SUPPORTED_KERNELS = {
    "fp16": {0, 1, 20, 21, 30, 31, 201, 10201},
    "bf16": {0, 1, 20, 21, 30, 31, 201, 10201},
    "fp32": {1, 21, 31, 101, 201, 10201, 20201},
}


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def knowledge_from_row(row: dict[str, str]) -> dict[str, int]:
    missing = [column for column in KNOWLEDGE_COLUMNS.values() if row.get(column, "") == ""]
    if missing:
        raise ValueError(f"candidate is missing tiling columns: {','.join(missing)}")
    return {
        name: int(row[column]) for name, column in KNOWLEDGE_COLUMNS.items()
    }


def l2_cache_flag(
    m: int,
    n: int,
    output_bytes: int,
    l2_bytes: int,
    knowledge: dict[str, int],
) -> int:
    """Mirror CANN 8.1 mat_mul_v3_l2_cache.cpp."""

    a_enabled = False
    b_enabled = False
    c_enabled = m * n * output_bytes <= l2_bytes
    bias_enabled = True
    enabled = knowledge["tilingEnable"]
    split = enabled % 10

    if split == 0:
        if knowledge["l2MTileCnt"] > 1 or knowledge["l2NTileCnt"] > 1:
            b_enabled = knowledge["l2MTileCnt"] > 1
            a_enabled = knowledge["l2NTileCnt"] > 1
        else:
            if knowledge["singleCoreM"] < m:
                b_enabled = True
            if knowledge["singleCoreN"] < n:
                a_enabled = True
            tasks = (
                ceil_div(m, knowledge["singleCoreM"])
                * ceil_div(n, knowledge["singleCoreN"])
            )
            if ceil_div(tasks, knowledge["usedCoreNum"]) > 1:
                b_enabled = True
    elif split == 2:
        c_enabled = True
        b_enabled = knowledge["singleCoreM"] < m
        a_enabled = knowledge["singleCoreN"] < n
        m_full = knowledge["singleCoreM"] <= knowledge["baseM"] * knowledge["stepM"]
        n_full = knowledge["singleCoreN"] <= knowledge["baseN"] * knowledge["stepN"]
        k_full = knowledge["singleCoreK"] <= int(knowledge.get("K", 0))
        if not m_full and not n_full:
            b_enabled = True
            a_enabled = not k_full
    elif split == 3:
        c_enabled = True
        b_enabled = knowledge["singleCoreM"] < m
        tasks = (
            ceil_div(m, knowledge["singleCoreM"])
            * ceil_div(knowledge["K"], knowledge["singleCoreK"])
        )
        one_round = ceil_div(tasks, knowledge["usedCoreNum"]) <= 1
        n_full = knowledge["singleCoreN"] == n
        if not one_round and not n_full:
            a_enabled = True
            b_enabled = not one_round

    if a_enabled and b_enabled and c_enabled and bias_enabled:
        return 1
    flag = 0
    if not a_enabled:
        flag |= 1 << 1
    if not b_enabled:
        flag |= 1 << 2
    if not bias_enabled:
        flag |= 1 << 3
    if not c_enabled:
        flag |= 1 << 4
    return flag


def _nd2nz_overflow(
    n_aligned: int,
    n_value: int,
    base_n: int,
    base_d: int,
    dtype_bytes: int,
) -> bool:
    n_aligned_loop = ceil_div(n_aligned, base_n)
    n_value_loop = ceil_div(n_value, base_n)
    if n_aligned_loop == n_value_loop:
        return False
    # The source expression is uint64_t; preserve its wrap semantics exactly.
    mask = (1 << 64) - 1
    completed = (n_value // base_n - 1) & mask
    remaining = (n_aligned - ((completed * base_n) & mask)) & mask
    return (
        remaining * base_d
        > UB_BYTES // 2 // dtype_bytes
    )


def nd2nz_base(
    dtype_bytes: int,
    n_value: int,
    d_value: int,
    used_cores: int,
) -> tuple[int, int]:
    """Mirror CANN 8.1 MatmulV3BaseTiling::CalcNd2NzTiling."""

    vector_cores = max(2 * used_cores, 1)
    base_threshold = 2048 // dtype_bytes
    c0 = 32 // dtype_bytes
    n_aligned = align_up(n_value, 16)
    d_aligned = align_up(d_value, c0)
    if d_value <= base_threshold:
        base_d = max(d_aligned, 1)
        initial_n = UB_BYTES // 2 // dtype_bytes // base_d
        rounds = max(ceil_div(ceil_div(n_aligned, vector_cores), initial_n), 1)
        base_n = max(ceil_div(ceil_div(n_aligned, vector_cores), rounds), 16)
        while base_n > 16 and _nd2nz_overflow(
            n_aligned, n_value, base_n, base_d, dtype_bytes
        ):
            base_n -= 1
        return base_n, base_d

    last_tail = 0
    best_n = 16
    best_d = 4096 // dtype_bytes
    for base_bytes in (6144, 4096, 2048):
        base_d = max(min(d_aligned, base_bytes // dtype_bytes), 1)
        d_loop = ceil_div(d_aligned, base_d)
        d_tail = d_aligned % base_d
        if 0 < d_tail < 512 // dtype_bytes:
            if base_d * dtype_bytes == 6144:
                continue
            d_loop -= 1
            base_d = max(align_up(ceil_div(d_aligned, d_loop), c0), 1)
        base_n = max(UB_BYTES // 2 // dtype_bytes // base_d, 16)
        if base_n * base_d * dtype_bytes * 2 > UB_BYTES:
            continue
        if _nd2nz_overflow(n_aligned, n_value, base_n, base_d, dtype_bytes):
            continue
        n_loop = ceil_div(n_aligned, base_n)
        tail = n_loop * d_loop % vector_cores
        while base_n > 16:
            if _nd2nz_overflow(n_aligned, n_value, base_n, base_d, dtype_bytes):
                base_n -= 1
                n_loop = ceil_div(n_aligned, base_n)
                tail = n_loop * d_loop % vector_cores
                continue
            if tail == 0:
                return base_n, base_d
            if tail > last_tail:
                last_tail = tail
                best_d = base_d
                best_n = base_n
            base_n -= 1
            n_loop = ceil_div(n_aligned, base_n)
            tail = n_loop * d_loop % vector_cores
    return best_n, best_d


def workspace_bytes(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    knowledge: dict[str, int],
    suffix: int,
) -> int:
    width = 4 if dtype == "fp32" else 2
    split = knowledge["tilingEnable"] % 10
    fix = (knowledge["tilingEnable"] // 1000) % 10
    size = RPC_WORKSPACE_BYTES
    if split == 2:
        size = m * align_up(n, 256 // width) * 4 + RPC_WORKSPACE_BYTES
    elif split == 3:
        size = (
            knowledge["usedCoreNum"]
            * knowledge["singleCoreN"]
            * knowledge["singleCoreM"]
            * 2 * 4
            + RPC_WORKSPACE_BYTES
        )
    if fix == 1:
        size += (
            align_up(n, 512 // width) * knowledge["baseM"]
            * knowledge["usedCoreNum"] * 2 * width
        )
    elif fix == 2:
        size += (
            align_up(n, 16) * knowledge["baseM"]
            * knowledge["usedCoreNum"] * 2 * width
        )

    graph = {
        0: "base", 1: "base", 20: "single_core_split_k",
        21: "single_core_split_k", 30: "deterministic_split_k",
        31: "deterministic_split_k", 101: "al1_full_load",
        200: "bl1_full_load", 201: "bl1_full_load",
        10200: "bl1_full_load_fixpipe", 10201: "bl1_full_load_fixpipe",
        20201: "bl1_full_load_vec_nz2nd",
    }[suffix]
    conversion = source_layout_conversion(
        m, n, k, dtype, trans_a, trans_b, graph_name=graph
    )
    c0 = 32 // width
    k_c0 = align_up(k, c0)
    k_n = align_up(k, 16)
    if conversion.a:
        size += (
            align_up(m, c0) * k_n * width if trans_a
            else align_up(m, 16) * k_c0 * width
        )
    if conversion.b:
        size += (
            align_up(n, 16) * k_c0 * width if trans_b
            else align_up(n, c0) * k_n * width
        )
    return size


@dataclass(frozen=True)
class MaterializedTiling:
    blob: bytes
    sha256: str
    fnv1a64: str
    suffix: int
    workspace_bytes: int
    l2_cache_flag: int
    nd2nz_a: int
    nd2nz_b: int


def materialize(row: dict[str, str], *, l2_bytes: int, aic_cores: int) -> MaterializedTiling:
    m, n, k = (int(row[name]) for name in ("m", "n", "k"))
    dtype = row["dtype"].lower()
    trans_a = truthy(row.get("trans_a"))
    trans_b = truthy(row.get("trans_b"))
    knowledge = knowledge_from_row(row)
    knowledge["K"] = k

    hardware = ascend_910b3()
    core_counts = dict(hardware.core_counts)
    core_counts[Resource.CUBE] = aic_cores
    capacities = dict(hardware.capacities)
    capacities[MemorySpace.L2] = l2_bytes
    capacities[MemorySpace.L1] = cann81_matmul_effective_l1_bytes(
        capacities[MemorySpace.L1]
    )
    from dataclasses import replace
    hardware = replace(hardware, core_counts=core_counts, capacities=capacities)
    violations = validate_cann_tiling(
        m, n, k, dtype, trans_a, trans_b, knowledge, hardware
    )
    if violations:
        raise ValueError("validator rejected candidate: " + ",".join(violations))

    suffix = source_kernel_suffix(
        m, n, k, dtype, trans_a, trans_b, knowledge
    )
    declared_suffix = int(row.get("model_kernel_suffix") or suffix)
    if declared_suffix != suffix:
        raise ValueError(
            f"kernel suffix mismatch model={declared_suffix} materialized={suffix}"
        )
    if suffix not in SUPPORTED_KERNELS.get(dtype, set()):
        raise ValueError(f"direct kernel was not built for {dtype}/suffix={suffix}")

    graph_name = row.get("execution_mode") or "base"
    if graph_name == "base_iterate_all":
        graph_name = "base"
    conversion = source_layout_conversion(
        m, n, k, dtype, trans_a, trans_b, graph_name=graph_name
    )
    width = 4 if dtype == "fp32" else 2
    base_an = base_ad = base_bn = base_bd = 0
    if conversion.a:
        n_value, d_value = (k, m) if trans_a else (m, k)
        base_an, base_ad = nd2nz_base(
            width, n_value, d_value, knowledge["usedCoreNum"]
        )
    if conversion.b:
        n_value, d_value = (n, k) if trans_b else (k, n)
        base_bn, base_bd = nd2nz_base(
            width, n_value, d_value, knowledge["usedCoreNum"]
        )

    words = [0] * (ABI_BYTES // 4)
    cube = {
        0: knowledge["usedCoreNum"], 1: m, 2: n, 3: k, 4: k,
        5: knowledge["singleCoreM"], 6: knowledge["singleCoreN"],
        7: knowledge["singleCoreK"], 8: knowledge["baseM"],
        9: knowledge["baseN"], 10: knowledge["baseK"],
        11: knowledge["depthA1"], 12: knowledge["depthB1"],
        13: knowledge["stepM"], 14: knowledge["stepN"],
        17: knowledge["iterateOrder"], 26: knowledge["stepKa"],
        27: knowledge["stepKb"], 30: knowledge["dbL0A"],
        31: knowledge["dbL0B"], 32: knowledge["dbL0C"],
    }
    for index, value in cube.items():
        words[index] = value
    words[50:55] = [
        knowledge["l2MTileCnt"], knowledge["l2NTileCnt"],
        knowledge["l2MTileBlock"], knowledge["l2NTileBlock"],
        knowledge["l2IterateOrder"],
    ]
    words[56:61] = [
        int(trans_a), int(trans_b), int(conversion.a), int(conversion.b), 0,
    ]
    flag = l2_cache_flag(m, n, width, l2_bytes, knowledge)
    words[62] = flag
    words[64:68] = [base_an, base_ad, base_bn, base_bd]
    blob = struct.pack("<68I", *words)
    digest = hashlib.sha256(blob).hexdigest()
    fnv = 0xCBF29CE484222325
    for value in blob:
        fnv = ((fnv ^ value) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return MaterializedTiling(
        blob=blob,
        sha256=digest,
        fnv1a64=f"{fnv:016x}",
        suffix=suffix,
        workspace_bytes=workspace_bytes(
            m, n, k, dtype, trans_a, trans_b, knowledge, suffix
        ),
        l2_cache_flag=flag,
        nd2nz_a=int(conversion.a),
        nd2nz_b=int(conversion.b),
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)


def write_manifest(
    candidates: Path,
    output_dir: Path,
    manifest: Path,
    *,
    l2_bytes: int,
    aic_cores: int,
    include_reserves: bool = False,
) -> int:
    with candidates.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fields = [
        "workload_id", "rank", "candidate_role", "m", "n", "k",
        "dtype", "trans_a", "trans_b", "used_core_num", "kernel_suffix",
        "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
        "model_schedule_sha256", "is_reserve", "l2_cache_flag",
        "nd2nz_a", "nd2nz_b", "required_successful_tilings",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("candidate_role") != "searched":
            continue
        if not include_reserves and truthy(row.get("is_reserve")):
            continue
        tiling = materialize(row, l2_bytes=l2_bytes, aic_cores=aic_cores)
        filename = (
            f"{safe_name(row['workload_id'])}__r{int(row['rank']):04d}__"
            f"{row['model_schedule_sha256'][:16]}.bin"
        )
        path = output_dir / filename
        if not path.is_file() or path.read_bytes() != tiling.blob:
            path.write_bytes(tiling.blob)
        manifest_rows.append({
            "workload_id": row["workload_id"],
            "rank": row["rank"],
            "candidate_role": "searched",
            "m": row["m"], "n": row["n"], "k": row["k"],
            "dtype": row["dtype"],
            "trans_a": row["trans_a"], "trans_b": row["trans_b"],
            "used_core_num": row["used_core_num"],
            "kernel_suffix": str(tiling.suffix),
            "workspace_bytes": str(tiling.workspace_bytes),
            "tiling_path": str(path.resolve()),
            "tiling_sha256": tiling.sha256,
            "tiling_fnv1a64": tiling.fnv1a64,
            "model_schedule_sha256": row["model_schedule_sha256"],
            "is_reserve": row.get("is_reserve", "0"),
            "l2_cache_flag": str(tiling.l2_cache_flag),
            "nd2nz_a": str(tiling.nd2nz_a),
            "nd2nz_b": str(tiling.nd2nz_b),
            "required_successful_tilings": row["required_successful_tilings"],
        })
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return len(manifest_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--include-reserves", action="store_true")
    args = parser.parse_args()
    count = write_manifest(
        args.candidates, args.output_dir, args.manifest,
        l2_bytes=args.l2_bytes, aic_cores=args.aic_cores,
        include_reserves=args.include_reserves,
    )
    print(json.dumps({
        "status": "passed", "candidates": count,
        "manifest": str(args.manifest), "tiling_bytes": ABI_BYTES,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
