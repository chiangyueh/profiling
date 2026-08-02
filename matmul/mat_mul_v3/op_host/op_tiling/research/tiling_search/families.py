from __future__ import annotations

from dataclasses import dataclass

from .contracts import ceil_div
from .domain import INPUT_BYTES, Workload


@dataclass(frozen=True)
class WorkloadStrata:
    geometry: str
    reduction: str
    parallelism: str
    alignment: str
    layout: str

    @property
    def composite(self) -> str:
        return "/".join(
            (
                self.geometry,
                self.reduction,
                self.parallelism,
                self.alignment,
            )
        )


def classify_workload(
    workload: Workload,
    aic_cores: int,
) -> WorkloadStrata:
    """Describe workload geometry for reporting without gating the search."""
    input_bytes = INPUT_BYTES[workload.dtype]
    macro_m = 128
    macro_n = 128
    macro_k = 256 // input_bytes
    m_tiles = ceil_div(workload.m, macro_m)
    n_tiles = ceil_div(workload.n, macro_n)

    if m_tiles == 1 and n_tiles > 1:
        geometry = "skinny_m"
    elif n_tiles == 1 and m_tiles > 1:
        geometry = "skinny_n"
    elif m_tiles >= 4 * n_tiles:
        geometry = "tall"
    elif n_tiles >= 4 * m_tiles:
        geometry = "wide"
    else:
        geometry = "balanced"

    k_passes = ceil_div(workload.k, macro_k)
    if k_passes <= 8:
        reduction = "shallow_k"
    elif k_passes >= 128:
        reduction = "deep_k"
    else:
        reduction = "regular_k"

    output_tiles = m_tiles * n_tiles
    cores = max(1, min(workload.max_cores, aic_cores))
    if output_tiles < cores:
        parallelism = "underfilled"
    elif output_tiles <= 2 * cores:
        parallelism = "one_to_two_waves"
    else:
        parallelism = "multi_wave"

    n_alignment = 32 // input_bytes
    if (
        workload.m % macro_m == 0
        and workload.n % macro_n == 0
        and workload.k % (4 * macro_k) == 0
    ):
        alignment = "macro_aligned"
    elif (
        workload.m % 16 == 0
        and workload.n % n_alignment == 0
        and workload.k % macro_k == 0
    ):
        alignment = "cube_aligned"
    else:
        alignment = "tail"

    layout = (
        ("T" if workload.trans_a else "N")
        + ("T" if workload.trans_b else "N")
    )
    return WorkloadStrata(
        geometry=geometry,
        reduction=reduction,
        parallelism=parallelism,
        alignment=alignment,
        layout=layout,
    )
