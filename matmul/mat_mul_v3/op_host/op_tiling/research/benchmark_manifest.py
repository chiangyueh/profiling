#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    workload_id: str
    m: int
    n: int
    k: int
    dtype: str
    trans_a: int
    trans_b: int
    max_cores: int
    benchmark_group: str
    benchmark_intent: str


def _case(
    index: int,
    group: str,
    intent: str,
    shape: tuple[int, int, int],
    dtype: str,
    layout: tuple[int, int],
    cores: int,
) -> Case:
    return Case(
        workload_id=f"bench_v2_{index:03d}_{group}",
        m=shape[0],
        n=shape[1],
        k=shape[2],
        dtype=dtype,
        trans_a=layout[0],
        trans_b=layout[1],
        max_cores=cores,
        benchmark_group=group,
        benchmark_intent=intent,
    )


def build_cases(cores: int) -> list[Case]:
    layouts = ((0, 0), (1, 0), (0, 1), (1, 1))
    cases: list[Case] = []

    def add_group(
        group: str,
        intent: str,
        shapes: list[tuple[int, int, int]],
        dtypes: tuple[str, ...],
        layout_offset: int = 0,
    ) -> None:
        for offset, shape in enumerate(shapes):
            cases.append(
                _case(
                    len(cases) + 1,
                    group,
                    intent,
                    shape,
                    dtypes[offset % len(dtypes)],
                    layouts[(offset + layout_offset) % len(layouts)],
                    cores,
                )
            )

    add_group(
        "balanced",
        "balanced dense, macro/cube aligned and nearby tails",
        [
            (1024, 1280, 1536),
            (1280, 1536, 2048),
            (1536, 1792, 2560),
            (1792, 2048, 3072),
            (2048, 2304, 3584),
            (2304, 2560, 4096),
            (2560, 3072, 4608),
            (3072, 3584, 5120),
            (3584, 4096, 5632),
            (4096, 4608, 6144),
            (4608, 5120, 6656),
            (5120, 5632, 7168),
            (1009, 1279, 1537),
            (1277, 1531, 2053),
            (1535, 1793, 2557),
            (1793, 2111, 3073),
            (2113, 2431, 3587),
            (2431, 2801, 4097),
            (2801, 3199, 4609),
            (3199, 3583, 5119),
            (3583, 4093, 5633),
            (4093, 4607, 6143),
            (4607, 5119, 6653),
            (5119, 5631, 7169),
        ],
        ("fp16", "fp32", "bf16", "fp16"),
    )
    add_group(
        "aspect",
        "tall and wide output grids across one, two and many waves",
        [
            (4096, 256, 2048),
            (6144, 384, 3072),
            (8192, 512, 4096),
            (12288, 640, 1536),
            (16384, 768, 1024),
            (24576, 896, 768),
            (32768, 256, 448),
            (49152, 160, 256),
            (3583, 257, 2053),
            (7169, 383, 3073),
            (12287, 511, 4097),
            (24575, 769, 1025),
            (256, 4096, 2048),
            (384, 6144, 3072),
            (512, 8192, 4096),
            (640, 12288, 1536),
            (768, 16384, 1024),
            (896, 24576, 768),
            (256, 32768, 448),
            (160, 49152, 256),
            (257, 3583, 2053),
            (383, 7169, 3073),
            (511, 12287, 4097),
            (769, 24575, 1025),
        ],
        ("fp16", "bf16", "fp32", "fp16"),
        1,
    )
    add_group(
        "underfilled",
        "skinny-M, skinny-N and low-output core underfill",
        [
            (24, 320, 5120),
            (31, 1536, 8192),
            (47, 3584, 12288),
            (73, 4608, 16384),
            (97, 5376, 15360),
            (127, 6144, 16384),
            (191, 7168, 12289),
            (255, 8192, 24576),
            (320, 24, 5120),
            (1536, 31, 8192),
            (3584, 47, 12288),
            (4608, 73, 16384),
            (5376, 97, 15360),
            (6144, 127, 16384),
            (7168, 191, 12289),
            (8192, 255, 24576),
            (32, 32, 128),
            (64, 96, 4096),
            (96, 64, 8192),
            (128, 128, 16384),
            (192, 128, 24576),
            (256, 192, 32768),
            (320, 256, 49152),
            (384, 288, 15360),
        ],
        ("fp16", "fp32", "bf16", "fp16"),
        2,
    )
    add_group(
        "reduction",
        "K-pass and split-K crossover with fixed and irregular outputs",
        [
            (2048, 2560, 128),
            (2304, 2816, 320),
            (2560, 3072, 448),
            (2816, 3328, 768),
            (3072, 3584, 1024),
            (3328, 3840, 2048),
            (3584, 4096, 4096),
            (4096, 4608, 8192),
            (1536, 2048, 12288),
            (1280, 1792, 16384),
            (1024, 1536, 24576),
            (768, 1280, 32768),
            (640, 896, 49152),
            (2113, 2591, 129),
            (2303, 2801, 319),
            (2557, 3073, 449),
            (2819, 3329, 769),
            (3071, 3583, 1025),
            (3329, 3839, 2049),
            (3583, 4093, 4097),
            (2049, 2431, 8193),
            (1023, 1537, 12289),
            (831, 609, 20481),
            (511, 383, 32769),
        ],
        ("fp16", "bf16", "fp32", "fp16"),
        3,
    )
    add_group(
        "alignment",
        "independent M/N/K tail penalties around cube and macro boundaries",
        [
            (1024, 1536, 2048),
            (1025, 1536, 2048),
            (1024, 1537, 2048),
            (1024, 1536, 2049),
            (1279, 1792, 3072),
            (1280, 1791, 3072),
            (1280, 1792, 3073),
            (1281, 1793, 3071),
            (2048, 2304, 4096),
            (2047, 2304, 4096),
            (2048, 2305, 4096),
            (2048, 2304, 4095),
            (2559, 3072, 5120),
            (2560, 3071, 5120),
            (2560, 3072, 5119),
            (2561, 3073, 5121),
            (4096, 4608, 6144),
            (4095, 4608, 6144),
            (4096, 4607, 6144),
            (4096, 4608, 6145),
            (4093, 4609, 6139),
            (3587, 4091, 5633),
            (3073, 3583, 5117),
            (2431, 2803, 4099),
        ],
        ("fp16", "bf16", "fp32", "fp16"),
    )
    add_group(
        "l1_residency",
        "A/B L1 residency and full-load template applicability boundaries",
        [
            (16, 256, 4096),
            (16, 304, 5120),
            (32, 384, 4096),
            (48, 512, 8192),
            (64, 640, 12288),
            (96, 768, 16384),
            (128, 896, 24576),
            (192, 1024, 32768),
            (256, 16, 4096),
            (304, 16, 5120),
            (384, 32, 4096),
            (512, 48, 8192),
            (640, 64, 12288),
            (768, 96, 16384),
            (896, 128, 24576),
            (1024, 192, 32768),
            (32768, 128, 128),
            (40960, 160, 128),
            (49152, 192, 128),
            (57344, 224, 192),
            (128, 32768, 128),
            (160, 40960, 128),
            (192, 49152, 128),
            (224, 57344, 192),
        ],
        ("fp32", "fp16", "bf16", "fp16"),
        1,
    )
    add_group(
        "l2_wave",
        "L2 working-set, output-wave and traversal-order transitions",
        [
            (1792, 2816, 3584),
            (2304, 3328, 4096),
            (2816, 3840, 4608),
            (3328, 4352, 5120),
            (3840, 4864, 5632),
            (4352, 5376, 6144),
            (4864, 5888, 6656),
            (5376, 6400, 7168),
            (6144, 1280, 6400),
            (7168, 1536, 7680),
            (8192, 1792, 8192),
            (9216, 2048, 6144),
            (1280, 6144, 6400),
            (1536, 7168, 7680),
            (1792, 8192, 8192),
            (2048, 9216, 6144),
            (1793, 2801, 3587),
            (2303, 3329, 4097),
            (2817, 3839, 4609),
            (3327, 4351, 5119),
            (3839, 4865, 5633),
            (4353, 5375, 6143),
            (4863, 5887, 6657),
            (5377, 6399, 7169),
        ],
        ("fp16", "fp16", "bf16", "fp16"),
        2,
    )
    cross_shapes = (
        (768, 1280, 1536),
        (1023, 1537, 3073),
    )
    cross_index = 0
    for dtype in ("fp16", "bf16", "fp32"):
        for layout in layouts:
            for shape in cross_shapes:
                cases.append(
                    _case(
                        len(cases) + 1,
                        "dtype_layout",
                        "complete dtype by NN/NT/TN/TT cross product",
                        shape,
                        dtype,
                        layout,
                        cores,
                    )
                )
                cross_index += 1

    if len(cases) != 192:
        raise AssertionError(f"expected 192 workloads, got {len(cases)}")
    identities: dict[tuple[object, ...], list[str]] = {}
    for case in cases:
        identity = (
            case.m,
            case.n,
            case.k,
            case.dtype,
            case.trans_a,
            case.trans_b,
        )
        identities.setdefault(identity, []).append(case.workload_id)
    duplicates = [
        workload_ids for workload_ids in identities.values() if len(workload_ids) > 1
    ]
    if duplicates:
        raise AssertionError(f"benchmark contains duplicate identities: {duplicates}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()

    cases = build_cases(args.aic_cores)
    if args.mode == "smoke":
        by_group: dict[str, list[Case]] = {}
        for case in cases:
            by_group.setdefault(case.benchmark_group, []).append(case)
        cases = [case for group_cases in by_group.values() for case in group_cases[:2]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(asdict(cases[0])))
        writer.writeheader()
        writer.writerows(asdict(case) for case in cases)

    groups = Counter(case.benchmark_group for case in cases)
    dtypes = Counter(case.dtype for case in cases)
    layouts = Counter(f"{case.trans_a}{case.trans_b}" for case in cases)
    print(
        "BENCHMARK_MANIFEST "
        + json.dumps(
            {
                "version": "hardware_coverage_v2",
                "workloads": len(cases),
                "groups": dict(sorted(groups.items())),
                "dtypes": dict(sorted(dtypes.items())),
                "layouts": dict(sorted(layouts.items())),
                "custom_candidates_stage1": 64,
                "custom_candidates_stage2": 32,
                "maximum_custom_measurements": len(cases) * 96,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    for case in cases:
        print(
            "BENCHMARK_WORKLOAD "
            + json.dumps(
                asdict(case),
                separators=(",", ":"),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
