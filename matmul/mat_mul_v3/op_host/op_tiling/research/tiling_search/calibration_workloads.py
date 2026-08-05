from __future__ import annotations

from dataclasses import dataclass

from .contracts import align_up
from .domain import Hardware, INPUT_BYTES, Template, Workload


@dataclass(frozen=True)
class CalibrationWorkload:
    workload: Workload
    template_quotas: dict[Template, int]
    design_axis: str
    resident_ratio: float


def _aligned_resident_dimension(
    capacity: int,
    ratio: float,
    fixed_extent: int,
    element_bytes: int,
    alignment: int,
) -> int:
    fixed = align_up(fixed_extent, 16)
    raw = int(capacity * ratio) // max(1, fixed * element_bytes)
    return max(alignment, raw // alignment * alignment)


def _resident_ratio(
    first: int,
    second: int,
    element_bytes: int,
    capacity: int,
    second_alignment: int,
) -> float:
    resident = (
        align_up(first, 16)
        * align_up(second, second_alignment)
        * element_bytes
    )
    return resident / max(1, capacity)


def _workload(
    workload_id: str,
    m: int,
    n: int,
    k: int,
    dtype: str,
    layout: tuple[bool, bool],
    hardware: Hardware,
) -> Workload:
    return Workload(
        workload_id=workload_id,
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        trans_a=layout[0],
        trans_b=layout[1],
        max_cores=hardware.aic_cores,
    )


def generate_template_calibration_workloads(
    hardware: Hardware,
) -> list[CalibrationWorkload]:
    """Create hardware-derived workloads for every supported template.

    These are experiment designs, not deployment shape families. Full-load
    dimensions are derived from fractions of the detected L1 capacity. Split-K
    dimensions are derived from output parallelism relative to the AIC count.
    """

    specs: list[CalibrationWorkload] = []
    l1 = hardware.effective_l1_bytes
    layouts = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    )
    ratios = (0.38, 0.50, 0.61, 0.70, 0.78, 0.84)
    dtypes = ("fp16", "bf16", "fp32", "fp32", "fp16", "bf16")

    for index, (ratio, dtype, layout) in enumerate(
        zip(ratios, dtypes, layouts), 1
    ):
        element_bytes = INPUT_BYTES[dtype]
        k = 1024 if element_bytes == 2 else 512
        m = _aligned_resident_dimension(
            l1, ratio, k, element_bytes, 16
        )
        n = hardware.aic_cores * 256 + index * 128
        actual = _resident_ratio(
            m, k, element_bytes, l1, 16
        )
        specs.append(
            CalibrationWorkload(
                workload=_workload(
                    f"template_v2_al1_{index:02d}",
                    m,
                    n,
                    k,
                    dtype,
                    layout,
                    hardware,
                ),
                template_quotas={Template.AL1_FULL_LOAD: 8},
                design_axis="a_l1_resident_ratio",
                resident_ratio=actual,
            )
        )

    bl1_ratios = (
        0.30,
        0.34,
        0.39,
        0.44,
        0.49,
        0.54,
        0.59,
        0.64,
        0.67,
        0.52,
        0.55,
        0.55,
    )
    bl1_layouts = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
        (False, False),
        (False, True),
        (True, False),
        (True, True),
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )
    for index, (ratio, layout) in enumerate(
        zip(bl1_ratios, bl1_layouts), 1
    ):
        dtype = "fp16" if index % 2 else "bf16"
        element_bytes = INPUT_BYTES[dtype]
        k = 128 + (index - 1) * 16
        n = _aligned_resident_dimension(
            l1, ratio, k, element_bytes, 16
        )
        m = hardware.aic_cores * 1024 + index * 256
        actual = _resident_ratio(
            k, n, element_bytes, l1, 16
        )
        specs.append(
            CalibrationWorkload(
                workload=_workload(
                    f"template_v2_bl1_fix_{index:02d}",
                    m,
                    n,
                    k,
                    dtype,
                    layout,
                    hardware,
                ),
                template_quotas={
                    Template.BL1_FULL_LOAD: 5,
                    Template.BL1_FULL_LOAD_FIXPIPE: 1,
                },
                design_axis="b_l1_resident_ratio_fixpipe",
                resident_ratio=actual,
            )
        )

    fp32_layouts = (
        (False, False),
        (False, True),
        (False, False),
        (False, True),
        (False, False),
        (False, True),
        (False, False),
        (False, True),
        (False, False),
        (False, True),
        (False, False),
        (False, True),
    )
    for index, (ratio, layout) in enumerate(
        zip(bl1_ratios, fp32_layouts), 1
    ):
        dtype = "fp32"
        element_bytes = INPUT_BYTES[dtype]
        k = 64 + (index - 1) * 16
        n = _aligned_resident_dimension(
            l1, ratio, k, element_bytes, 16
        )
        m = hardware.aic_cores * 768 + index * 256
        actual = _resident_ratio(
            k, n, element_bytes, l1, 8
        )
        specs.append(
            CalibrationWorkload(
                workload=_workload(
                    f"template_v2_bl1_vec_{index:02d}",
                    m,
                    n,
                    k,
                    dtype,
                    layout,
                    hardware,
                ),
                template_quotas={
                    Template.BL1_FULL_LOAD: 5,
                    Template.BL1_FULL_LOAD_VEC_NZ2ND: 1,
                },
                design_axis="b_l1_resident_ratio_vec_nz2nd",
                resident_ratio=actual,
            )
        )

    cube = 128
    reduction_unit = hardware.aic_cores * 1024
    split_shapes = (
        (
            2 * cube - cube // 2,
            cube,
            reduction_unit * 4 // 5,
            "fp16",
            (False, False),
        ),
        (
            2 * cube,
            2 * cube - cube // 2,
            reduction_unit * 6 // 5,
            "fp16",
            (False, True),
        ),
        (
            3 * cube,
            2 * cube,
            reduction_unit * 8 // 5,
            "bf16",
            (False, False),
        ),
        (
            4 * cube,
            3 * cube,
            reduction_unit * 12 // 5,
            "fp16",
            (True, False),
        ),
        (
            3 * cube - cube // 2,
            3 * cube - 7 * cube // 8,
            reduction_unit + 1,
            "fp16",
            (True, True),
        ),
        (
            3 * cube,
            3 * cube - 3 * cube // 4,
            reduction_unit * 3 // 4,
            "fp32",
            (False, False),
        ),
    )
    for index, (m, n, k, dtype, layout) in enumerate(
        split_shapes, 1
    ):
        output_tiles = (
            (m + 127) // 128
            * ((n + 127) // 128)
        )
        specs.append(
            CalibrationWorkload(
                workload=_workload(
                    f"template_v2_split_{index:02d}",
                    m,
                    n,
                    k,
                    dtype,
                    layout,
                    hardware,
                ),
                template_quotas={
                    Template.SINGLE_CORE_SPLIT_K: 5,
                    Template.DETERMINISTIC_SPLIT_K: 2,
                },
                design_axis="output_parallelism_per_aic",
                resident_ratio=(
                    output_tiles / max(1, hardware.aic_cores)
                ),
            )
        )
    return specs


def encode_template_quotas(quotas: dict[Template, int]) -> str:
    return "|".join(
        f"{template.value}:{quota}"
        for template, quota in sorted(
            quotas.items(), key=lambda item: item[0].value
        )
    )


def decode_template_quotas(value: str) -> dict[Template, int]:
    quotas: dict[Template, int] = {}
    for item in value.split("|"):
        if not item:
            continue
        name, separator, raw_quota = item.partition(":")
        if not separator:
            raise ValueError(f"invalid template quota: {item}")
        template = Template(name)
        quota = int(raw_quota)
        if quota <= 0:
            raise ValueError(f"template quota must be positive: {item}")
        quotas[template] = quota
    return quotas
