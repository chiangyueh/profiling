#!/usr/bin/env python3
"""Deterministic, unique MatMul shapes for paired NPU ranking validation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


CATALOG_SIZE = 240
SELECTED_SIZE = 200
GENERAL_CATALOG_SIZE = 180
DEEP_REDUCTION_CATALOG_SIZE = 60
MAX_INPUT_BYTES = 160 * 1024 * 1024
MAX_OUTPUT_BYTES = 96 * 1024 * 1024


@dataclass(frozen=True)
class Mode:
    dtype: str
    trans_a: int
    trans_b: int


MODES = tuple(
    Mode(dtype, trans_a, trans_b)
    for dtype in ("fp16", "bf16", "fp32")
    for trans_a in (0, 1)
    for trans_b in (0, 1)
)

# These are not single expected-answer shapes.  Each row marks a distinct
# legal workload region whose CANN 8.1 dispatch has a different resident-set,
# output, reduction, or ND2NZ route.  The solver still receives every graph
# and chooses from parameters; the labels are coverage metadata only.
FAMILY_STRESS_CASES = (
    # AL1 full load: FP32 NT, skinny M, large resident K.
    (4, 48, 4096, Mode("fp32", 0, 1), "cann81_al1_full_load"),
    (8, 96, 6144, Mode("fp32", 0, 1), "cann81_al1_full_load"),
    (12, 160, 7168, Mode("fp32", 0, 1), "cann81_al1_full_load"),
    (16, 304, 7168, Mode("fp32", 0, 1), "cann81_al1_full_load"),
    # BL1 full load: tall M and a complete small B resident set.
    (8192, 64, 32, Mode("fp16", 0, 0), "cann81_bl1_full_load"),
    (16384, 96, 64, Mode("bf16", 0, 0), "cann81_bl1_full_load"),
    (32768, 128, 128, Mode("fp16", 0, 0), "cann81_bl1_full_load"),
    (65536, 192, 256, Mode("fp16", 0, 0), "cann81_bl1_full_load"),
    # BL1 + FixPipe: unaligned narrow output with FP16/BF16 conversion off.
    (10240, 7, 16, Mode("fp16", 0, 0), "cann81_bl1_fixpipe"),
    (12288, 17, 32, Mode("bf16", 0, 0), "cann81_bl1_fixpipe"),
    (16384, 31, 64, Mode("fp16", 0, 0), "cann81_bl1_fixpipe"),
    (20480, 47, 128, Mode("fp16", 0, 0), "cann81_bl1_fixpipe"),
    # BL1 + vector NZ2ND: FP32 route selected from conversion state, not a
    # transpose-name special case.
    (10240, 9, 8, Mode("fp32", 0, 0), "cann81_bl1_vec_nz2nd"),
    (12288, 17, 16, Mode("fp32", 0, 0), "cann81_bl1_vec_nz2nd"),
    (16384, 31, 24, Mode("fp32", 0, 0), "cann81_bl1_vec_nz2nd"),
    (20480, 47, 32, Mode("fp32", 0, 0), "cann81_bl1_vec_nz2nd"),
    # Explicit mixed-ND2NZ routes cover suffixes 0/20/30/200/10200.
    (257, 1009, 4097, Mode("fp16", 0, 0), "cann81_mixed_nd2nz"),
    (128, 17, 16384, Mode("fp16", 0, 0), "cann81_splitk_mixed_nd2nz"),
    (28672, 64, 17, Mode("fp16", 0, 0), "cann81_bl1_mixed_nd2nz"),
    (24576, 17, 7, Mode("fp32", 0, 0), "cann81_fixpipe_mixed_nd2nz"),
)

PHYSICAL_AIC_CORES = 20
M_VALUES = (
    96, 112, 128, 144, 160, 176, 192, 224, 256, 320, 384, 448,
    512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
    3584, 4096,
)
N_VALUES = (
    80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512, 640,
    768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096,
    5120, 6144, 7168,
)
K_VALUES = (
    128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
    4096, 6144, 7168, 8192, 12288, 16384,
)


def element_bytes(dtype: str) -> int:
    return 4 if dtype == "fp32" else 2


def within_budget(m: int, n: int, k: int, dtype: str) -> bool:
    width = element_bytes(dtype)
    return (
        (m * k + k * n) * width <= MAX_INPUT_BYTES
        and m * n * width <= MAX_OUTPUT_BYTES
    )


def _row(
    index: int,
    m: int,
    n: int,
    k: int,
    mode: Mode,
    family: str,
    search_family: str = "hardware_ideal_region",
) -> dict[str, str]:
    return {
        "workload_id": f"matmul_rank_{index:03d}",
        "m": str(m),
        "n": str(n),
        "k": str(k),
        "dtype": mode.dtype,
        "trans_a": str(mode.trans_a),
        "trans_b": str(mode.trans_b),
        # Every workload exposes the complete physical AIC pool.  The solver
        # chooses usedCoreNum from the schedule, rather than changing the
        # hardware capacity from one shape to the next.
        "max_cores": str(PHYSICAL_AIC_CORES),
        "search_family": search_family,
        "coverage": (
            f"{family};{mode.dtype};t{mode.trans_a}{mode.trans_b};"
            f"full_aic_{PHYSICAL_AIC_CORES};m{m};n{n};k{k}"
        ),
    }


def build_catalog() -> list[dict[str, str]]:
    anchors = (
        # Colleague notation was M x K x N; rows store conventional M,N,K.
        (2048, 1536, 7168),
        (2048, 7168, 2048),
        (4096, 512, 7168),
    )
    rows: list[dict[str, str]] = []
    seen: set[tuple[int, int, int]] = set()
    for m, n, k in anchors:
        seen.add((m, n, k))
        rows.append(_row(
            len(rows), m, n, k, Mode("fp16", 0, 0),
            "colleague_anchor",
        ))

    for m, n, k, mode, family in FAMILY_STRESS_CASES:
        if not within_budget(m, n, k, mode.dtype):
            raise RuntimeError(f"family stress shape exceeds budget: {m}x{n}x{k}")
        signature = (m, n, k)
        if signature in seen:
            raise RuntimeError(f"duplicate family stress shape: {signature}")
        seen.add(signature)
        rows.append(_row(len(rows), m, n, k, mode, family))

    # Coprime strides traverse only reviewed boundary values.  There is no
    # random generation and no duplicate M,N,K triple hidden behind a dtype.
    for sequence in range(1_000_000):
        m = M_VALUES[(sequence * 7 + sequence // 13) % len(M_VALUES)]
        n = N_VALUES[(sequence * 11 + sequence // 17) % len(N_VALUES)]
        k = K_VALUES[(sequence * 13 + sequence // 19) % len(K_VALUES)]
        mode = MODES[(sequence * 5 + sequence // 23) % len(MODES)]
        signature = (m, n, k)
        if signature in seen or not within_budget(m, n, k, mode.dtype):
            continue
        seen.add(signature)
        rows.append(
            _row(len(rows), m, n, k, mode, "base_hardware_lattice")
        )
        if len(rows) == GENERAL_CATALOG_SIZE:
            break
    if len(rows) != GENERAL_CATALOG_SIZE:
        raise RuntimeError("could not fill the general validation catalog")

    split_modes = (Mode("fp16", 0, 0), Mode("bf16", 0, 0))
    split_m = (128, 192, 256, 320, 384, 448, 512, 640, 768)
    split_n = (128, 160, 192, 256, 320, 384, 448, 512, 640, 768)
    split_k = (16384, 24576, 32768, 40960, 49152)
    split_count = 0
    for sequence in range(1_000_000):
        m = split_m[(sequence * 5 + sequence // 11) % len(split_m)]
        n = split_n[(sequence * 7 + sequence // 13) % len(split_n)]
        k = split_k[(sequence * 3 + sequence // 17) % len(split_k)]
        mode = split_modes[(sequence + sequence // 19) % len(split_modes)]
        signature = (m, n, k)
        if signature in seen or not within_budget(m, n, k, mode.dtype):
            continue
        seen.add(signature)
        rows.append(_row(
            len(rows), m, n, k, mode,
            "deep_reduction_lattice",
        ))
        split_count += 1
        if split_count == DEEP_REDUCTION_CATALOG_SIZE:
            break
    if len(rows) != CATALOG_SIZE:
        raise RuntimeError(f"generated {len(rows)} workloads, expected {CATALOG_SIZE}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_catalog()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"MATMUL_MODEL_VALIDATION_CATALOG shapes={len(rows)} selected_target={SELECTED_SIZE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
