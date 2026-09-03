"""CANN 8.1 MatMulV3 input-layout conversion equations.

This module describes which ND input must first be materialized as NZ.  It is
shared by the operator IR and the CANN ABI adapter so workspace traffic,
kernel suffix selection, and validation cannot silently disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import dtype_bytes


_ON_THE_WAY_BYTES = frozenset((32, 64, 96, 128, 160, 192, 224, 256, 384))
_ND2NZ_ON_THE_FLY_LIMIT = 65535
_VNCHW_OUTER_THRESHOLD = 8192
_VNCHW_MAX_BYTES = 512
_DETERMINISTIC_ON_THE_WAY_MIN_BYTES = 384


@dataclass(frozen=True)
class LayoutConversion:
    a: bool
    b: bool

    @property
    def disable_mix_nd2nz(self) -> int:
        """Low tiling-key digit: 1 disables the mixed ND2NZ kernel."""

        return int(not (self.a or self.b))


def _needs_nd2nz(outer: int, inner: int, dtype: str, layout: str) -> bool:
    if layout != "ND":
        return False
    width = dtype_bytes(dtype)
    aligned_256b = (inner * width) % 256 == 0
    on_the_way = inner * width in _ON_THE_WAY_BYTES
    # CANN 8.1's non-HF32 path can perform ordinary FP32 conversion on the
    # way while its inner dimension fits the 16-bit instruction field.
    ordinary = (
        (not aligned_256b or inner > _ND2NZ_ON_THE_FLY_LIMIT)
        and not on_the_way
        and not (dtype == "fp32" and inner < _ND2NZ_ON_THE_FLY_LIMIT)
    )
    vnchw = (
        outer > _VNCHW_OUTER_THRESHOLD
        and inner > 1
        and (
            inner * width <= 192
            or (inner * width <= 384 and inner % 2 == 0)
            or (inner * width <= _VNCHW_MAX_BYTES and inner % 4 == 0)
        )
        and not aligned_256b
        and not on_the_way
        and inner != 32 // width
    )
    return ordinary or vnchw


def source_layout_conversion(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    *,
    a_layout: str = "ND",
    b_layout: str = "ND",
    graph_name: str = "base",
) -> LayoutConversion:
    """Return the exact C220 A/B conversion flags used to form the key.

    The first part mirrors ``GetMoreArgs``.  The final adjustments mirror the
    only CANN 8.1 graph implementations that change those flags before
    ``DoTilingKey``.
    """

    inner_a, outer_a = ((m, k) if trans_a else (k, m))
    inner_b, outer_b = ((k, n) if trans_b else (n, k))
    convert_a = _needs_nd2nz(outer_a, inner_a, dtype, a_layout)
    convert_b = _needs_nd2nz(outer_b, inner_b, dtype, b_layout)

    # CANN explicitly preconverts B for this large-M associative-cache
    # conflict, even though the shape can otherwise use on-the-way ND2NZ.
    if (
        not trans_b
        and n % 16384 == 0
        and m > 4096
        and k >= 6656
        and b_layout == "ND"
        and dtype in ("fp16", "bf16")
    ):
        convert_b = True

    if graph_name == "deterministic_split_k":
        width = dtype_bytes(dtype)
        minimum = _DETERMINISTIC_ON_THE_WAY_MIN_BYTES // width
        if minimum <= inner_a <= _ND2NZ_ON_THE_FLY_LIMIT:
            convert_a = False
        if minimum <= inner_b <= _ND2NZ_ON_THE_FLY_LIMIT:
            convert_b = False
    elif graph_name in (
        "bl1_full_load_fixpipe",
        "bl1_full_load_vec_nz2nd",
    ):
        # The BL1 FixPipe route consumes B in its resident form.  For
        # FP16/BF16 NeedSolveFixBound also forces A to the on-the-way path.
        convert_b = False
        if dtype in ("fp16", "bf16"):
            convert_a = False

    return LayoutConversion(convert_a, convert_b)
