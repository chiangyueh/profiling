#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


BLOCK_COUNTS = {
    "l2": 80,
    "concurrency": 60,
    "buffer": 60,
    "splitk": 50,
}
MAX_INPUT_BYTES = 240 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class Mode:
    dtype: str
    trans_a: int
    trans_b: int


MODES = (
    Mode("fp16", 0, 0),
    Mode("fp16", 0, 1),
    Mode("fp16", 1, 0),
    Mode("fp16", 1, 1),
    Mode("bf16", 0, 0),
    Mode("bf16", 0, 1),
    Mode("bf16", 1, 0),
    Mode("bf16", 1, 1),
    Mode("fp32", 0, 0),
    Mode("fp32", 0, 1),
    Mode("fp32", 1, 0),
    Mode("fp32", 1, 1),
)


def element_bytes(dtype: str) -> int:
    return 4 if dtype == "fp32" else 2


def within_numeric_budget(m: int, n: int, k: int, dtype: str) -> bool:
    width = element_bytes(dtype)
    return (
        (m * k + k * n) * width <= MAX_INPUT_BYTES
        and m * n * width <= MAX_OUTPUT_BYTES
    )


def coverage_rows(
    block: str,
    count: int,
    m_values: tuple[int, ...],
    n_values: tuple[int, ...],
    k_values: tuple[int, ...],
    modes: tuple[Mode, ...] = MODES,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[int, int, int, str, int, int]] = set()
    # The coprime strides walk the explicit lattice without random sampling.
    # Every retained point is still one of the reviewed boundary values above.
    for sequence in range(200_000):
        m = m_values[(sequence * 5 + sequence // 11) % len(m_values)]
        n = n_values[(sequence * 7 + sequence // 13) % len(n_values)]
        k = k_values[(sequence * 11 + sequence // 17) % len(k_values)]
        mode = modes[(sequence * 5 + sequence // 7) % len(modes)]
        signature = (m, n, k, mode.dtype, mode.trans_a, mode.trans_b)
        if signature in seen or not within_numeric_budget(m, n, k, mode.dtype):
            continue
        seen.add(signature)
        index = len(rows)
        rows.append(
            {
                "workload_id": f"{block}_{index:03d}",
                "m": str(m),
                "n": str(n),
                "k": str(k),
                "dtype": mode.dtype,
                "trans_a": str(mode.trans_a),
                "trans_b": str(mode.trans_b),
                "max_cores": "20",
                "experiment_block": block,
                "coverage": (
                    f"{block};{mode.dtype};t{mode.trans_a}{mode.trans_b};"
                    f"m{m};n{n};k{k}"
                ),
            }
        )
        if len(rows) == count:
            return rows
    raise RuntimeError(f"could not construct {count} unique {block} workloads")


def build_catalog() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        coverage_rows(
            "l2",
            BLOCK_COUNTS["l2"],
            (768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 4112),
            (640, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096),
            (1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288),
        )
    )
    rows.extend(
        coverage_rows(
            "concurrency",
            BLOCK_COUNTS["concurrency"],
            (320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2064),
            (320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2064),
            (256, 384, 512, 768, 1024, 1536, 2048, 3072),
        )
    )
    rows.extend(
        coverage_rows(
            "buffer",
            BLOCK_COUNTS["buffer"],
            (256, 384, 512, 768, 1024, 1280, 1536, 2048),
            (256, 384, 512, 768, 1024, 1280, 1536, 2048),
            (
                127, 128, 129, 255, 256, 257, 383, 384, 385,
                511, 512, 513, 1023, 1024, 1025, 2048, 4096,
            ),
        )
    )
    # Split-K is deliberately NN and aligned.  The varied M/N/K values expose
    # sync/reduction scaling without mixing in transpose conversion paths.
    split_modes = (Mode("fp16", 0, 0), Mode("bf16", 0, 0))
    rows.extend(
        coverage_rows(
            "splitk",
            BLOCK_COUNTS["splitk"],
            (128, 256, 384, 512, 640, 768, 896, 1024),
            (128, 256, 384, 512, 640, 768, 896, 1024),
            (16384, 24576, 32768, 40960, 49152),
            split_modes,
        )
    )
    expected = sum(BLOCK_COUNTS.values())
    if len(rows) != expected:
        raise RuntimeError(f"catalog has {len(rows)} rows, expected {expected}")
    signatures = {
        (
            row["m"], row["n"], row["k"], row["dtype"],
            row["trans_a"], row["trans_b"],
        )
        for row in rows
    }
    if len(signatures) != len(rows):
        raise RuntimeError("catalog contains duplicate semantic shapes")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_catalog()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "workload_id", "m", "n", "k", "dtype", "trans_a", "trans_b",
        "max_cores", "experiment_block", "coverage",
    )
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "MATMUL_CONTROLLED_WORKLOADS "
        + " ".join(f"{name}={count}" for name, count in BLOCK_COUNTS.items())
        + f" total={len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
