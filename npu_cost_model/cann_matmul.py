"""CANN 8.1 MatMulV3 ABI adapter for the generic hardware simulator.

Only this boundary understands CANN field names and tiling-key digits.  It
does not contain measured latency, workload-history gates, or fitted shape
thresholds.  Once translated, the evaluator sees only an IR graph and a
numeric :class:`TilingPlan`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

from .hardware import Hardware
from .ir import MemorySpace, Resource, dtype_bytes
from .matmul_layout import source_layout_conversion
from .operators import matmul
from .schedule import TilingPlan
from .simulator import SimulationResult, align_up, ceil_div, simulate


CANN81_MATMUL_FAMILIES = (
    "base",
    "single_core_split_k",
    "deterministic_split_k",
    "al1_full_load",
    "bl1_full_load",
    "bl1_full_load_fixpipe",
    "bl1_full_load_vec_nz2nd",
)
CANN81_MATMUL_KERNEL_SUFFIXES = (
    0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201,
)

_ND2NZ_ON_THE_WAY_BYTES = frozenset(
    (32, 64, 96, 128, 160, 192, 224, 256, 384)
)


def cann81_matmul_effective_l1_bytes(reported_bytes: int) -> int:
    """Return the C220 single-op L1 capacity used by MatMulV3.

    Ascend910B3 reports 512 KiB minus the 256-byte RPC reservation.  The
    CANN 8.1 MatMulV3 ``CalL1Tiling`` path restores that reservation for a
    standalone operator.  Rounding to the allocator's KiB boundary also
    leaves an already-normalized capacity unchanged.
    """

    return align_up(int(reported_bytes), 1024)


@lru_cache(maxsize=256)
def _matmul_operator(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    a_layout: str,
    b_layout: str,
    output_dtype: str | None,
    has_bias: bool,
):
    """Reuse the immutable IR graph across ABI lowering and validation."""

    return matmul(
        m, n, k, dtype,
        trans_a=trans_a,
        trans_b=trans_b,
        a_layout=a_layout,
        b_layout=b_layout,
        output_dtype=output_dtype,
        has_bias=has_bias,
    )


class SplitCoreMode(IntEnum):
    BASE = 0
    SINGLE_CORE_SPLIT_K = 2
    DETERMINISTIC_SPLIT_K = 3


class FullLoadMode(IntEnum):
    BASE = 0
    AL1_FULL_LOAD = 1
    BL1_FULL_LOAD = 2


class FixOptMode(IntEnum):
    BASE = 0
    BASE_ENABLE_ALIGNOUT = 1
    VEC_NZ2ND_UNALIGNOUT = 2


@dataclass(frozen=True)
class MatmulExecutionGraph:
    split: SplitCoreMode
    full_load: FullLoadMode
    fix: FixOptMode
    mix_nd2nz: int
    name: str


def _integer(values: Mapping[str, int], name: str) -> int:
    value = int(values[name])
    if value <= 0:
        raise ValueError(f"MatMulV3 {name} must be positive")
    return value


def decode_execution_graph(
    knowledge: Mapping[str, int],
    *,
    aligned: bool | None = None,
) -> MatmulExecutionGraph:
    """Decode every C220 MatMulV3 dataflow digit used by CANN 8.1."""

    enabled = int(knowledge["tilingEnable"])
    try:
        split = SplitCoreMode(enabled % 10)
        full_load = FullLoadMode((enabled // 10) % 10)
        fix = FixOptMode((enabled // 1000) % 10)
    except ValueError as exception:
        raise ValueError("undefined MatMulV3 tilingEnable digit") from exception
    if enabled // 10000:
        raise ValueError("undefined MatMulV3 tilingEnable high digits")
    mix_nd2nz = int(knowledge.get("mixNd2Nz", int(bool(aligned))))
    if mix_nd2nz not in (0, 1):
        raise ValueError("MatMulV3 mixNd2Nz must be 0 or 1")
    if split == SplitCoreMode.DETERMINISTIC_SPLIT_K:
        name = "deterministic_split_k"
    elif split == SplitCoreMode.SINGLE_CORE_SPLIT_K:
        name = "single_core_split_k"
    elif full_load == FullLoadMode.AL1_FULL_LOAD:
        name = "al1_full_load"
    elif full_load == FullLoadMode.BL1_FULL_LOAD:
        if fix == FixOptMode.VEC_NZ2ND_UNALIGNOUT:
            name = "bl1_full_load_vec_nz2nd"
        elif fix == FixOptMode.BASE_ENABLE_ALIGNOUT:
            name = "bl1_full_load_fixpipe"
        else:
            name = "bl1_full_load"
    else:
        name = "base"
    return MatmulExecutionGraph(
        split, full_load, fix, mix_nd2nz, name
    )


def execution_mode_name(knowledge: Mapping[str, int]) -> str:
    return decode_execution_graph(knowledge).name


def kernel_suffix(knowledge: Mapping[str, int], *, aligned: bool) -> int:
    """Construct the low five decimal tiling-key digits used on DAV C220."""

    graph = decode_execution_graph(knowledge, aligned=aligned)
    return (
        int(graph.fix) * 10000
        + int(graph.full_load) * 100
        + int(graph.split) * 10
        + graph.mix_nd2nz
    )


def source_kernel_suffix(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    knowledge: Mapping[str, int],
    *,
    a_layout: str = "ND",
    b_layout: str = "ND",
) -> int:
    """Construct the suffix from source-derived input conversion flags."""

    graph = decode_execution_graph(knowledge)
    mix = source_layout_conversion(
        m, n, k, dtype, trans_a, trans_b,
        a_layout=a_layout,
        b_layout=b_layout,
        graph_name=graph.name,
    ).disable_mix_nd2nz
    return (
        int(graph.fix) * 10000
        + int(graph.full_load) * 100
        + int(graph.split) * 10
        + mix
    )


def _l2_schedule_reasons(
    m: int,
    n: int,
    knowledge: Mapping[str, int],
) -> list[str]:
    reasons: list[str] = []
    single_m = int(knowledge["singleCoreM"])
    single_n = int(knowledge["singleCoreN"])
    m_total = ceil_div(m, single_m)
    n_total = ceil_div(n, single_n)
    m_block = int(knowledge["l2MTileBlock"])
    n_block = int(knowledge["l2NTileBlock"])
    m_count = int(knowledge["l2MTileCnt"])
    n_count = int(knowledge["l2NTileCnt"])
    if not m_block or not n_block:
        if not (
            m_block == n_block == 0 and m_count == n_count == 1
        ):
            reasons.append("L2_DISABLED_FIELDS_INCONSISTENT")
        return reasons
    if m_count != ceil_div(m_total, m_block):
        reasons.append("L2_M_TILE_COUNT_INCONSISTENT")
    if n_count != ceil_div(n_total, n_block):
        reasons.append("L2_N_TILE_COUNT_INCONSISTENT")
    m_tail = m_total - (m_count - 1) * m_block
    n_tail = n_total - (n_count - 1) * n_block
    if not 1 <= m_tail <= m_block:
        reasons.append("L2_M_TAIL_OUTSIDE_BLOCK")
    if not 1 <= n_tail <= n_block:
        reasons.append("L2_N_TAIL_OUTSIDE_BLOCK")
    return reasons


def _source_bl1_full_load_available(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    conversion_b: bool,
    l1_bytes: int,
    *,
    base_n: int,
    has_bias: bool,
) -> bool:
    """Mirror ``DoBL1FullLoadTiling``'s source-visible entry contract."""

    in_bytes = dtype_bytes(dtype)
    c0 = 32 // in_bytes
    inner_a = m if trans_a else k
    outer_a = k if trans_a else m
    # Keep the source's two different units exactly: its first lookup uses
    # N directly, while the transposed-B guard looks up K in bytes.
    b_on_the_way = (
        n in _ND2NZ_ON_THE_WAY_BYTES
        and (
            not trans_b
            or k * in_bytes in _ND2NZ_ON_THE_WAY_BYTES
        )
    )
    aligned_bl1 = b_on_the_way and not conversion_b
    a_vnchw = (
        dtype == "fp32"
        and outer_a >= 72_368
        and 1 < inner_a <= c0
    )
    bias_bytes = base_n * 4 if has_bias else 0
    resident_b_fits = (
        l1_bytes // 2 - bias_bytes > k * n * in_bytes
    )
    return (
        m > 16 * max(k, n)
        and k <= 256
        and (aligned_bl1 or (a_vnchw and resident_b_fits))
    )


def validate_cann_tiling(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    knowledge: Mapping[str, int],
    hardware: Hardware,
    *,
    a_layout: str = "ND",
    b_layout: str = "ND",
    output_dtype: str | None = None,
    has_bias: bool = False,
    aoe_injection: bool = False,
) -> tuple[str, ...]:
    """Return structural CANN/kernel violations without running a callback.

    Shape-specific profitability ranges from the source tiler are
    intentionally absent: they decide whether a legal graph is worthwhile,
    not whether its kernel can execute correctly.
    """

    reasons: list[str] = []
    required_positive = (
        "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
        "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
        "stepN", "stepKa", "stepKb", "dbL0A", "dbL0B", "dbL0C",
        "l2MTileCnt", "l2NTileCnt",
    )
    missing = tuple(name for name in required_positive if name not in knowledge)
    if missing:
        return tuple(f"MISSING_{name}" for name in missing)
    if any(int(knowledge[name]) <= 0 for name in required_positive):
        reasons.append("NON_POSITIVE_TILING_FIELD")
    if int(knowledge.get("l2MTileBlock", -1)) < 0 or int(
        knowledge.get("l2NTileBlock", -1)
    ) < 0:
        reasons.append("NEGATIVE_L2_TILE_BLOCK")
    try:
        graph = decode_execution_graph(
            knowledge,
            # The graph family is encoded by tilingEnable.  When a RuntimeKb
            # row omits the derived low key digit, source_layout_conversion
            # below remains the authority rather than this placeholder.
            aligned=True,
        )
    except ValueError:
        return tuple((*reasons, "UNDEFINED_EXECUTION_GRAPH"))

    available_graphs = {
        algorithm.name
        for algorithm in _matmul_operator(
            m, n, k, dtype,
            trans_a, trans_b, a_layout, b_layout, output_dtype, has_bias,
        ).algorithms
    }
    if graph.name not in available_graphs:
        reasons.append("EXECUTION_GRAPH_UNAVAILABLE_FOR_WORKLOAD")

    split = graph.split
    full = graph.full_load
    fix = graph.fix
    del aoe_injection
    legal_combinations = {
        (SplitCoreMode.BASE, FullLoadMode.BASE, FixOptMode.BASE),
        (SplitCoreMode.BASE, FullLoadMode.AL1_FULL_LOAD, FixOptMode.BASE),
        (SplitCoreMode.BASE, FullLoadMode.BL1_FULL_LOAD, FixOptMode.BASE),
        (SplitCoreMode.BASE, FullLoadMode.BL1_FULL_LOAD,
         FixOptMode.BASE_ENABLE_ALIGNOUT),
        (SplitCoreMode.BASE, FullLoadMode.BL1_FULL_LOAD,
         FixOptMode.VEC_NZ2ND_UNALIGNOUT),
        (SplitCoreMode.SINGLE_CORE_SPLIT_K, FullLoadMode.BASE,
         FixOptMode.BASE),
        (SplitCoreMode.DETERMINISTIC_SPLIT_K, FullLoadMode.BASE,
         FixOptMode.BASE),
    }
    if (split, full, fix) not in legal_combinations:
        reasons.append("KERNEL_GRAPH_COMBINATION_NOT_DISPATCHED")
    if graph.name in (
        "al1_full_load", "bl1_full_load_vec_nz2nd",
    ) and "mixNd2Nz" in knowledge and graph.mix_nd2nz != 1:
        reasons.append("EXECUTION_GRAPH_REQUIRES_NO_ND2NZ")
    expected_mix = source_layout_conversion(
        m, n, k, dtype, trans_a, trans_b,
        a_layout=a_layout,
        b_layout=b_layout,
        graph_name=graph.name,
    ).disable_mix_nd2nz
    if "mixNd2Nz" in knowledge and graph.mix_nd2nz != expected_mix:
        reasons.append("MIX_ND2NZ_SUFFIX_MISMATCH")

    used = int(knowledge["usedCoreNum"])
    if used > hardware.core_count(Resource.CUBE):
        reasons.append("USED_CORES_EXCEED_HARDWARE")
    if int(knowledge.get("iterateOrder", -1)) not in (0, 1):
        reasons.append("INVALID_CUBE_ITERATE_ORDER")
    if int(knowledge.get("l2IterateOrder", -1)) not in (0, 1, 2):
        reasons.append("INVALID_L2_ITERATE_ORDER")
    if any(int(knowledge[name]) not in (1, 2) for name in (
        "dbL0A", "dbL0B", "dbL0C"
    )):
        reasons.append("INVALID_L0_BUFFER_COUNT")

    in_bytes = dtype_bytes(dtype)
    k0 = 8 if dtype == "fp32" and not trans_a and trans_b else 16
    base_m = int(knowledge["baseM"])
    base_n = int(knowledge["baseN"])
    base_k = int(knowledge["baseK"])
    if base_m % 16 or base_n % 16 or base_k % k0:
        reasons.append("CUBE_BASE_ALIGNMENT_VIOLATION")
    l0a = base_m * base_k * in_bytes * int(knowledge["dbL0A"])
    l0b = base_n * base_k * in_bytes * int(knowledge["dbL0B"])
    l0c = base_m * base_n * 4 * int(knowledge["dbL0C"])
    for name, size, space in (
        ("L0A", l0a, MemorySpace.L0A),
        ("L0B", l0b, MemorySpace.L0B),
        ("L0C", l0c, MemorySpace.L0C),
    ):
        capacity = hardware.capacities.get(space, 0)
        if capacity and size > capacity:
            reasons.append(f"{name}_CAPACITY_EXCEEDED")

    step_m = int(knowledge["stepM"])
    step_n = int(knowledge["stepN"])
    step_ka = int(knowledge["stepKa"])
    step_kb = int(knowledge["stepKb"])
    depth_a = int(knowledge["depthA1"])
    depth_b = int(knowledge["depthB1"])
    one_a = step_m * step_ka
    one_b = step_n * step_kb
    if depth_a % one_a or depth_a // one_a not in (1, 2):
        reasons.append("AL1_DEPTH_NOT_ONE_OR_TWO_PACKETS")
    if depth_b % one_b or depth_b // one_b not in (1, 2):
        reasons.append("BL1_DEPTH_NOT_ONE_OR_TWO_PACKETS")
    if step_ka % step_kb and step_kb % step_ka:
        reasons.append("STEP_KA_KB_INCOMMENSURATE")

    single_m = int(knowledge["singleCoreM"])
    single_n = int(knowledge["singleCoreN"])
    single_k = int(knowledge["singleCoreK"])
    a1 = base_m * base_k * depth_a * in_bytes
    b1 = base_n * base_k * depth_b * in_bytes
    if full == FullLoadMode.AL1_FULL_LOAD:
        a1 = align_up(m, 16) * align_up(k, 32 // in_bytes) * in_bytes
    if full == FullLoadMode.BL1_FULL_LOAD:
        b1 = align_up(n, 16 if trans_b else 32 // in_bytes) * align_up(k, 16) * in_bytes
    l1_capacity = hardware.capacities.get(MemorySpace.L1, 0)
    if l1_capacity and a1 + b1 > l1_capacity:
        reasons.append("L1_RESIDENT_SET_CAPACITY_EXCEEDED")

    output_tasks = ceil_div(m, single_m) * ceil_div(n, single_n)
    k_chunks = ceil_div(k, single_k)
    if split == SplitCoreMode.BASE:
        if single_k < k:
            reasons.append("NON_SPLIT_GRAPH_DOES_NOT_COVER_K")
        if full == FullLoadMode.BASE and (
            single_m > base_m
            or single_n > base_n
            or base_m > align_up(single_m, 16)
            or base_n > align_up(single_n, 16)
            or base_k > align_up(single_k, k0)
            or step_m != 1
            or step_n != 1
        ):
            reasons.append("BASE_TASK_AND_INNER_GEOMETRY_INCONSISTENT")
        if used > output_tasks:
            reasons.append("USED_CORES_EXCEED_OUTPUT_TASKS")
    elif split == SplitCoreMode.SINGLE_CORE_SPLIT_K:
        if single_k != step_ka * base_k or step_ka != step_kb:
            reasons.append("SERIAL_SPLIT_K_CHUNK_INCONSISTENT")
        if k_chunks < 2:
            reasons.append("SERIAL_SPLIT_K_REQUIRES_MULTIPLE_K_CHUNKS")
        if single_m < step_m * base_m or single_n < step_n * base_n:
            reasons.append("SERIAL_SPLIT_K_TASK_SMALLER_THAN_ITERATE_REGION")
        if used > output_tasks:
            reasons.append("USED_CORES_EXCEED_OUTPUT_TASKS")
    else:
        # Deterministic Split-K is a parallel partial-C producer graph plus
        # an AIV workspace reduction.  CANN's source tiler reaches it only
        # with multiple K partitions and derives usedCoreNum from those
        # partitions.  One AIC is not the serial form of this graph.
        if used < 2:
            reasons.append(
                "DETERMINISTIC_SPLIT_K_REQUIRES_MULTIPLE_CUBE_CORES"
            )
        if single_k % base_k:
            reasons.append("PARALLEL_SPLIT_K_CHUNK_NOT_BASE_ALIGNED")
        if k_chunks < 2:
            reasons.append("PARALLEL_SPLIT_K_REQUIRES_MULTIPLE_K_CHUNKS")
        if used > output_tasks * k_chunks:
            reasons.append("USED_CORES_EXCEED_SPLIT_TASKS")
        expected_base_k = 256 // in_bytes
        mk33 = (
            step_m, step_n, step_ka, step_kb,
            depth_a, depth_b, single_m,
        ) == (3, 1, 3, 3, 9, 6, 384)
        nk33 = (
            step_m, step_n, step_ka, step_kb,
            depth_a, depth_b, single_n,
        ) == (1, 3, 3, 3, 6, 9, 384)
        if (
            (base_m, base_n, base_k) != (128, 128, expected_base_k)
            or not (mk33 or nk33)
            or single_k != 3 * base_k
            or used > k_chunks
        ):
            reasons.append("DETERMINISTIC_SPLIT_K_3X3_CONTRACT")
        # Every deterministic partial is allocated with a
        # singleCoreM*singleCoreN stride.  The C220 reducer and MatMul API
        # round the final N block to 256 bytes in FP32 workspace.  If that
        # rounded tail is wider than the allocated N stride, one ping-pong
        # slot overwrites the next (the 512x160x49152 BF16 device failure
        # starts exactly at the first affected row).  This is a workspace
        # address bound, not a profitability threshold.
        n_tail = n - (ceil_div(n, single_n) - 1) * single_n
        if align_up(n_tail, 256 // dtype_bytes("fp32")) > single_n:
            reasons.append("DETERMINISTIC_SPLIT_K_N_TAIL_WORKSPACE_OVERFLOW")

    if full == FullLoadMode.AL1_FULL_LOAD:
        if not (
            dtype == "fp32"
            and not trans_a
            and trans_b
            and m <= 16 < n <= 16 * hardware.core_count(Resource.CUBE)
            and k >= 4096
            and k % (512 // in_bytes) == 0
        ):
            reasons.append("AL1_FULL_LOAD_WORKLOAD_CONTRACT")
        if not (
            base_m == 16
            and base_n == 16
            and base_k == 256
            and m <= single_m <= 16
            and single_n == 16
            and single_k >= k
            and step_m == step_n == step_kb == 1
            and step_ka == ceil_div(k, base_k)
            and depth_a == step_ka
            and depth_b == 2
        ):
            reasons.append("AL1_FULL_LOAD_GEOMETRY_CONTRACT")

    if full == FullLoadMode.BL1_FULL_LOAD:
        c0 = 32 // in_bytes
        if k > 256:
            reasons.append("BL1_FULL_LOAD_K_EXCEEDS_256")
        conversion = source_layout_conversion(
            m, n, k, dtype, trans_a, trans_b,
            a_layout=a_layout,
            b_layout=b_layout,
            graph_name=graph.name,
        )
        if fix == FixOptMode.BASE and not _source_bl1_full_load_available(
            m, n, k, dtype, trans_a, trans_b,
            conversion.b, l1_capacity,
            base_n=base_n, has_bias=has_bias,
        ):
            reasons.append("BL1_FULL_LOAD_WORKLOAD_CONTRACT")
        if not (
            single_k >= k
            and step_m == 1
            and step_n == ceil_div(n, base_n)
            and step_ka == step_kb == ceil_div(k, base_k)
            and depth_b == step_n * step_kb
        ):
            reasons.append("BL1_FULL_LOAD_GEOMETRY_CONTRACT")
        if fix == FixOptMode.BASE and not (
            single_m == 2 * base_m
            and single_n >= n
            and depth_a == 2 * step_ka
        ):
            reasons.append("BL1_FULL_LOAD_PACKET_CONTRACT")
        if fix != FixOptMode.BASE:
            align_elements = 256 // dtype_bytes(output_dtype or dtype)
            fix_bound = (
                n < 256
                and k <= 256
                and n % align_elements != 0
                and align_elements % n != 0
                and m >= hardware.core_count(Resource.CUBE) * 128
                and not (n < c0 and k < c0)
            )
            if dtype in ("fp16", "bf16"):
                fix_bound = fix_bound and not trans_a and (
                    k % (256 // in_bytes) == 0 or k < 256 // in_bytes
                )
            if not fix_bound:
                reasons.append("BL1_FIXPIPE_WORKLOAD_CONTRACT")
            if single_m != base_m or single_n != base_n:
                reasons.append("BL1_FIXPIPE_TASK_CONTRACT")
            if base_m % 128 or base_n != align_up(n, 16):
                reasons.append("BL1_FIXPIPE_BASE_GEOMETRY_CONTRACT")
            if fix == FixOptMode.VEC_NZ2ND_UNALIGNOUT and not (
                dtype == "fp32"
                and not conversion.a
                and k % 8 == 0
                and n <= 192
            ):
                reasons.append("BL1_VEC_NZ2ND_WORKLOAD_CONTRACT")

    if split in (
        SplitCoreMode.SINGLE_CORE_SPLIT_K,
        SplitCoreMode.DETERMINISTIC_SPLIT_K,
    ) and (
        int(knowledge["l2MTileBlock"]) <= 0
        or int(knowledge["l2NTileBlock"]) <= 0
    ):
        reasons.append("SPLIT_K_REQUIRES_L2_TILE_BLOCKS")
    if fix != FixOptMode.BASE and (
        int(knowledge["l2MTileBlock"]) != 0
        or int(knowledge["l2NTileBlock"]) != 0
    ):
        reasons.append("BL1_FIXPIPE_REQUIRES_DISABLED_L2_TILING")

    reasons.extend(_l2_schedule_reasons(m, n, knowledge))
    return tuple(dict.fromkeys(reasons))


_GRAPH_TILING_ENABLE = {
    "base": 0,
    "al1_full_load": 10,
    "bl1_full_load": 20,
    "bl1_full_load_fixpipe": 1020,
    "bl1_full_load_vec_nz2nd": 2020,
    "single_core_split_k": 2,
    "deterministic_split_k": 3,
}


def _l2_fields(
    m: int,
    n: int,
    single_m: int,
    single_n: int,
    plan: TilingPlan,
    *,
    disabled: bool = False,
) -> dict[str, int]:
    if disabled:
        return {
            "l2MTileCnt": 1, "l2NTileCnt": 1,
            "l2MTileBlock": 0, "l2NTileBlock": 0,
            "l2IterateOrder": 0,
        }
    m_total = ceil_div(m, single_m)
    n_total = ceil_div(n, single_n)
    m_block = max(1, min(m_total, ceil_div(plan.caches["m"], single_m)))
    n_block = max(1, min(n_total, ceil_div(plan.caches["n"], single_n)))
    traversal = plan.traversal or ("m", "n")
    return {
        "l2MTileCnt": ceil_div(m_total, m_block),
        "l2NTileCnt": ceil_div(n_total, n_block),
        "l2MTileBlock": m_block,
        "l2NTileBlock": n_block,
        "l2IterateOrder": 2 if traversal[-1] == "m" else 1,
    }


def _packet_depths(
    base_m: int,
    base_n: int,
    base_k: int,
    step_m: int,
    step_n: int,
    step_ka: int,
    step_kb: int,
    input_bytes: int,
    l1_capacity: int,
    prefer_double: bool,
) -> tuple[int, int] | None:
    """Choose one/two L1 packets using only declared capacity equations."""

    packet_a = base_m * base_k * step_m * step_ka * input_bytes
    packet_b = base_n * base_k * step_n * step_kb * input_bytes
    for a_packets, b_packets in (
        ((2, 2), (2, 1), (1, 2), (1, 1))
        if prefer_double else
        ((1, 1),)
    ):
        if packet_a * a_packets + packet_b * b_packets <= l1_capacity:
            return (
                step_m * step_ka * a_packets,
                step_n * step_kb * b_packets,
            )
    return None


def _update_l1_step_k(
    step_k: int,
    base_k: int,
    k: int,
    input_bytes: int,
) -> int:
    """Apply the C220 DMA packet alignment rule to one K step."""

    if step_k <= 0 or step_k * base_k >= k:
        return step_k
    packet_bytes = step_k * base_k * input_bytes
    quantum = 512 if packet_bytes > 512 else 256 if packet_bytes > 256 else 0
    base_bytes = base_k * input_bytes
    if (
        quantum
        and packet_bytes % quantum
        and quantum % base_bytes == 0
    ):
        while step_k > 1 and step_k * base_bytes % quantum:
            step_k -= 1
    return step_k


def _base_l1_packet_depths(
    base_m: int,
    base_n: int,
    base_k: int,
    k: int,
    input_bytes: int,
    l1_capacity: int,
    has_bias: bool,
) -> tuple[int, int, int, int] | None:
    """Reproduce CANN 8.1 ``CalL1Tiling`` from physical capacities.

    The returned values are ``depthA1, depthB1, stepKa, stepKb``.  They are
    derived from L1 size and transfer alignment only; no callback, latency
    table, measured result, or shape-specific constant participates.
    """

    l1_capacity = int(l1_capacity)
    half = 2
    double_buffer = 2
    bias_bytes = 256 * 4 if has_bias else 0
    depth_a = l1_capacity // half // base_m // base_k // input_bytes
    depth_b = l1_capacity // half // base_n // base_k // input_bytes
    if depth_a <= 0 or depth_b <= 0:
        return None
    size_a = depth_a * base_m * base_k * input_bytes
    size_b = depth_b * base_n * base_k * input_bytes
    if size_a + size_b > l1_capacity - bias_bytes:
        if base_m <= base_n:
            depth_a //= half
        else:
            depth_b //= half
    step_ka = _update_l1_step_k(
        depth_a // double_buffer, base_k, k, input_bytes
    )
    step_kb = _update_l1_step_k(
        depth_b // double_buffer, base_k, k, input_bytes
    )
    if step_ka <= 0 or step_kb <= 0:
        return None
    if step_ka >= step_kb:
        step_ka = step_ka // step_kb * step_kb
    else:
        step_kb = step_kb // step_ka * step_ka
    depth_a = step_ka * double_buffer
    depth_b = step_kb * double_buffer
    if (
        depth_a * base_m * base_k * input_bytes
        + depth_b * base_n * base_k * input_bytes
        + bias_bytes
        > l1_capacity
    ):
        return None
    return depth_a, depth_b, step_ka, step_kb


def lower_plan_to_cann(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    plan: TilingPlan,
    hardware: Hardware,
    *,
    a_layout: str = "ND",
    b_layout: str = "ND",
    output_dtype: str | None = None,
    has_bias: bool = False,
) -> dict[str, int]:
    """Lower a generic hardware schedule to one complete CANN 8.1 ABI row.

    No callback, RuntimeKb record, measured latency or shape whitelist is
    consulted.  Family-specific constants below are execution-ABI geometry
    (for example the C220 3x3 deterministic reduction packet), while the
    shape-dependent sizes and core count come from ``plan``.
    """

    operator = _matmul_operator(
        m, n, k, dtype,
        trans_a, trans_b, a_layout, b_layout, output_dtype, has_bias,
    )
    if not 0 <= plan.algorithm < len(operator.algorithms):
        raise ValueError("plan algorithm is outside the MatMul execution graph set")
    graph_name = operator.algorithms[plan.algorithm].name
    if graph_name not in _GRAPH_TILING_ENABLE:
        raise ValueError(f"no CANN lowering for execution graph {graph_name}")

    input_bytes = dtype_bytes(dtype)
    k0 = 8 if dtype == "fp32" and not trans_a and trans_b else 16
    l1_capacity = hardware.capacities.get(MemorySpace.L1, 0)
    buffers = plan.buffer_counts
    base_m = align_up(min(max(16, plan.tiles["m"]), max(16, m)), 16)
    base_n = align_up(min(max(16, plan.tiles["n"]), max(16, n)), 16)
    base_k = align_up(min(max(k0, plan.tiles["k"]), max(k0, k)), k0)
    # A plan already passed generic local-memory validation, but padding a
    # tail at the ABI boundary can cross a capacity edge.  Shrink only by a
    # hardware alignment quantum until all three Cube memories fit.
    while base_m * base_k * input_bytes * buffers.get(MemorySpace.L0A, 1) > hardware.capacities[MemorySpace.L0A]:
        base_m -= 16
        if base_m < 16:
            raise ValueError("cannot lower plan within L0A capacity")
    while base_n * base_k * input_bytes * buffers.get(MemorySpace.L0B, 1) > hardware.capacities[MemorySpace.L0B]:
        base_n -= 16
        if base_n < 16:
            raise ValueError("cannot lower plan within L0B capacity")
    while base_m * base_n * 4 * buffers.get(MemorySpace.L0C, 1) > hardware.capacities[MemorySpace.L0C]:
        if base_m >= base_n and base_m > 16:
            base_m -= 16
        elif base_n > 16:
            base_n -= 16
        else:
            raise ValueError("cannot lower plan within L0C capacity")

    single_m = max(min(m, plan.tasks["m"]), min(m, base_m))
    single_n = max(min(n, plan.tasks["n"]), min(n, base_n))
    split = SplitCoreMode(_GRAPH_TILING_ENABLE[graph_name] % 10)
    full = FullLoadMode((_GRAPH_TILING_ENABLE[graph_name] // 10) % 10)
    fix = FixOptMode((_GRAPH_TILING_ENABLE[graph_name] // 1000) % 10)
    prefer_double = buffers.get(MemorySpace.L1, 1) == 2

    step_m = max(1, ceil_div(single_m, base_m))
    step_n = max(1, ceil_div(single_n, base_n))
    step_ka = step_kb = 1
    single_k = k
    depth_a = depth_b = 1

    if graph_name == "deterministic_split_k":
        # Exact C220 deterministic workspace-reduction packet geometry.
        base_m = base_n = 128
        base_k = 256 // input_bytes
        if (plan.traversal or ("m", "n"))[-1] == "m":
            step_m, step_n = 1, 3
            depth_a, depth_b = 6, 9
            single_m, single_n = m, 384
        else:
            step_m, step_n = 3, 1
            depth_a, depth_b = 9, 6
            single_m, single_n = 384, n
        step_ka = step_kb = 3
        single_k = 3 * base_k
    elif full == FullLoadMode.AL1_FULL_LOAD and split == SplitCoreMode.BASE:
        if not (
            dtype == "fp32"
            and not trans_a
            and trans_b
            and m <= 16 < n <= 16 * hardware.core_count(Resource.CUBE)
            and k >= 4096
            and k % (512 // input_bytes) == 0
        ):
            raise ValueError("AL1 full-load is outside the CANN 8.1 source contract")
        base_m = base_n = 16
        base_k = 256
        single_m = m
        single_n = 16
        step_m = step_n = 1
        step_ka = ceil_div(k, base_k)
        step_kb = 1
        depth_a = step_ka
        depth_b = 2
        single_k = k
    elif full == FullLoadMode.BL1_FULL_LOAD:
        if k > 256:
            raise ValueError("BL1 full-load requires K <= 256")
        if fix == FixOptMode.BASE:
            conversion = source_layout_conversion(
                m, n, k, dtype, trans_a, trans_b,
                a_layout=a_layout,
                b_layout=b_layout,
                graph_name=graph_name,
            )
            if not _source_bl1_full_load_available(
                m, n, k, dtype, trans_a, trans_b,
                conversion.b, l1_capacity,
                base_n=base_n, has_bias=has_bias,
            ):
                raise ValueError("BL1 full-load is outside the CANN 8.1 source contract")
            base_n = align_up(min(n, base_n), 16 if trans_b else 32 // input_bytes)
            base_k = align_up(min(k, base_k), k0)
            step_m = 1
            step_n = ceil_div(n, base_n)
            step_ka = step_kb = ceil_div(k, base_k)
            depth_a = 2 * step_ka
            depth_b = step_n * step_kb
            while (
                base_k * (depth_a * base_m + depth_b * base_n) * input_bytes
                > l1_capacity
            ):
                base_m //= 2
                base_m = base_m // 16 * 16
                if base_m < 16:
                    raise ValueError("BL1 full-load resident set does not fit L1")
            single_m = 2 * base_m
            single_n = n
            single_k = k
        else:
            out_bytes = dtype_bytes(output_dtype or dtype)
            align_elements = 256 // out_bytes
            c0 = 32 // input_bytes
            fix_bound = (
                n < 256
                and n % align_elements != 0
                and align_elements % n != 0
                and m >= hardware.core_count(Resource.CUBE) * 128
                and not (n < c0 and k < c0)
            )
            if dtype in ("fp16", "bf16"):
                fix_bound = fix_bound and not trans_a and (
                    k % (256 // input_bytes) == 0 or k < 256 // input_bytes
                )
            if not fix_bound:
                raise ValueError("BL1 FixPipe is outside the CANN 8.1 source contract")
            conversion = source_layout_conversion(
                m, n, k, dtype, trans_a, trans_b,
                a_layout=a_layout,
                b_layout=b_layout,
                graph_name=graph_name,
            )
            if graph_name == "bl1_full_load_vec_nz2nd" and not (
                dtype == "fp32" and not conversion.a
                and k % 8 == 0 and n <= 192
            ):
                raise ValueError("BL1 vector NZ2ND is outside the CANN 8.1 source contract")
            base_n = align_up(n, 16)
            ub_capacity = hardware.capacities.get(MemorySpace.UB, 0)
            # The source reserves 1/256 of UB per element row before the two
            # AIV sub-blocks/ping-pong offsets are applied.
            base_m_max = ub_capacity // 256 // out_bytes
            base_m = min(
                hardware.capacities[MemorySpace.L0C] // (base_n * 4),
                base_m_max,
            )
            base_m = base_m // 128 * 128
            if base_m < 128:
                raise ValueError("BL1 FixPipe has no legal baseM")
            base_ka = hardware.capacities[MemorySpace.L0A] // 2 // input_bytes // base_m
            base_kb = hardware.capacities[MemorySpace.L0B] // 2 // input_bytes // base_n
            base_k = min(base_ka, base_kb) // 16 * 16
            if base_k < 16:
                raise ValueError("BL1 FixPipe has no legal baseK")
            depth_b = ceil_div(k, base_k)
            step_kb = depth_b
            remaining = l1_capacity // input_bytes - depth_b * base_n * base_k
            depth_a = max(0, remaining // (base_m * base_k))
            depth_a = min(depth_a, depth_b)
            if depth_a >= 8:
                step_ka, depth_a = 4, 8
            else:
                step_ka = depth_a
            if depth_a <= 0:
                raise ValueError("BL1 FixPipe has no legal AL1 packet")
            step_m = step_n = 1
            single_m = base_m
            single_n = base_n
            single_k = k
    elif split == SplitCoreMode.SINGLE_CORE_SPLIT_K:
        step_m = max(1, min(3, ceil_div(single_m, base_m)))
        step_n = max(1, min(3, ceil_div(single_n, base_n)))
        target_parts = max(2, plan.reductions.get("k", 2))
        target_k = align_up(ceil_div(k, target_parts), base_k)
        max_step = max(1, (k - 1) // base_k)
        step_ka = step_kb = max(1, min(max_step, ceil_div(target_k, base_k)))
        single_k = step_ka * base_k
        while step_ka > 0:
            depths = _packet_depths(
                base_m, base_n, base_k, step_m, step_n,
                step_ka, step_kb, input_bytes, l1_capacity,
                prefer_double,
            )
            if depths is not None:
                depth_a, depth_b = depths
                single_k = step_ka * base_k
                break
            step_ka -= 1
            step_kb = step_ka
        if step_ka <= 0 or single_k >= k:
            raise ValueError("serial Split-K has no legal L1-resident K packet")
    else:
        # BASE uses one scheduled base region per core task.  This exact
        # relation is what prevents the incomplete-output failure reproduced
        # by the colleague's hand-authored 512x512x512 tiling.
        single_m = min(m, base_m)
        single_n = min(n, base_n)
        step_m = step_n = 1
        depths = _base_l1_packet_depths(
            base_m, base_n, base_k, k, input_bytes, l1_capacity,
            has_bias,
        )
        if depths is None:
            raise ValueError("direct graph L1 packet does not fit")
        depth_a, depth_b, step_ka, step_kb = depths

    output_tasks = ceil_div(m, single_m) * ceil_div(n, single_n)
    k_chunks = ceil_div(k, single_k)
    executable_tasks = output_tasks * k_chunks if split == SplitCoreMode.DETERMINISTIC_SPLIT_K else output_tasks
    used_cores = min(
        plan.used_cores,
        executable_tasks,
        hardware.core_count(Resource.CUBE),
    )
    if graph_name == "deterministic_split_k":
        used_cores = min(k_chunks, hardware.core_count(Resource.CUBE))

    disabled_l2 = full == FullLoadMode.BL1_FULL_LOAD and fix != FixOptMode.BASE
    aligned_mix = source_layout_conversion(
        m, n, k, dtype, trans_a, trans_b,
        a_layout=a_layout,
        b_layout=b_layout,
        graph_name=graph_name,
    ).disable_mix_nd2nz
    knowledge = {
        "usedCoreNum": max(1, used_cores),
        "singleCoreM": single_m,
        "singleCoreN": single_n,
        "singleCoreK": single_k,
        "baseM": base_m,
        "baseN": base_n,
        "baseK": base_k,
        "depthA1": depth_a,
        "depthB1": depth_b,
        "stepM": step_m,
        "stepN": step_n,
        "iterateOrder": 1 if (plan.traversal or ("m", "n"))[-1] == "m" else 0,
        "stepKa": step_ka,
        "stepKb": step_kb,
        "dbL0A": buffers.get(MemorySpace.L0A, 1),
        "dbL0B": buffers.get(MemorySpace.L0B, 1),
        "dbL0C": buffers.get(MemorySpace.L0C, 1),
        "tilingEnable": _GRAPH_TILING_ENABLE[graph_name],
        "mixNd2Nz": aligned_mix,
        "fp32Addmm": 0,
        **_l2_fields(
            m, n, single_m, single_n, plan, disabled=disabled_l2,
        ),
    }
    violations = validate_cann_tiling(
        m, n, k, dtype, trans_a, trans_b, knowledge, hardware,
        a_layout=a_layout,
        b_layout=b_layout,
        output_dtype=output_dtype,
        has_bias=has_bias,
    )
    if violations:
        raise ValueError(
            f"lowered {graph_name} plan violates CANN contract: "
            + ",".join(violations)
        )
    return knowledge


def plan_from_cann(
    m: int,
    n: int,
    k: int,
    knowledge: Mapping[str, int],
    *,
    dtype: str = "fp16",
    trans_a: bool = False,
    trans_b: bool = False,
    a_layout: str = "ND",
    b_layout: str = "ND",
    output_dtype: str | None = None,
    has_bias: bool = False,
) -> TilingPlan:
    """Translate every source-visible CANN 8.1 C220 MatMulV3 graph."""

    base_m = _integer(knowledge, "baseM")
    base_n = _integer(knowledge, "baseN")
    base_k = _integer(knowledge, "baseK")
    single_m = _integer(knowledge, "singleCoreM")
    single_n = _integer(knowledge, "singleCoreN")
    single_k = _integer(knowledge, "singleCoreK")
    graph = decode_execution_graph(
        knowledge,
        aligned=(m % 16 == 0 and n % 16 == 0 and k % base_k == 0),
    )
    operator = _matmul_operator(
        m, n, k, dtype,
        trans_a, trans_b, a_layout, b_layout, output_dtype, has_bias,
    )
    names = {algorithm.name: index for index, algorithm in enumerate(operator.algorithms)}
    if graph.name not in names:
        raise ValueError(
            f"MatMulV3 execution graph {graph.name} is unavailable for this dtype/layout"
        )

    reduction_parts = (
        1
        if graph.split == SplitCoreMode.BASE
        else ceil_div(k, single_k)
    )
    l2_m = int(knowledge["l2MTileBlock"]) * single_m
    l2_n = int(knowledge["l2NTileBlock"]) * single_n
    l2_order = int(knowledge["l2IterateOrder"])
    traversal = ("n", "m") if l2_order == 2 else ("m", "n")
    a_packets = _integer(knowledge, "depthA1") // (
        _integer(knowledge, "stepM") * _integer(knowledge, "stepKa")
    )
    b_packets = _integer(knowledge, "depthB1") // (
        _integer(knowledge, "stepN") * _integer(knowledge, "stepKb")
    )
    l1_buffers = 2 if min(a_packets, b_packets) >= 2 else 1
    step_m = _integer(knowledge, "stepM")
    step_n = _integer(knowledge, "stepN")
    step_ka = _integer(knowledge, "stepKa")
    step_kb = _integer(knowledge, "stepKb")
    invocation_tiles = ()
    if graph.split == SplitCoreMode.SINGLE_CORE_SPLIT_K:
        invocation_tiles = (
            ("m", _integer(knowledge, "stepM") * base_m),
            ("n", _integer(knowledge, "stepN") * base_n),
            ("k", single_k),
        )
    transfer_tiles = (
        (
            "A", MemorySpace.GM, MemorySpace.L1, "m",
            min(single_m, step_m * base_m),
        ),
        (
            "A", MemorySpace.GM, MemorySpace.L1, "k",
            min(single_k, step_ka * base_k),
        ),
        (
            "B", MemorySpace.GM, MemorySpace.L1, "n",
            min(single_n, step_n * base_n),
        ),
        (
            "B", MemorySpace.GM, MemorySpace.L1, "k",
            min(single_k, step_kb * base_k),
        ),
    )
    return TilingPlan(
        algorithm=names[graph.name],
        axis_tiles=(("m", base_m), ("n", base_n), ("k", base_k)),
        task_tiles=(("m", single_m), ("n", single_n), ("k", single_k)),
        invocation_tiles=invocation_tiles,
        transfer_tiles=transfer_tiles,
        cache_tiles=(
            ("m", max(min(single_m, m), min(max(single_m, l2_m), m))),
            ("n", max(min(single_n, n), min(max(single_n, l2_n), n))),
            ("k", k if graph.split == SplitCoreMode.BASE else single_k),
        ),
        used_cores=_integer(knowledge, "usedCoreNum"),
        reduction_parts=(("k", reduction_parts),),
        buffers=(
            (MemorySpace.L1, l1_buffers),
            (MemorySpace.L0A, _integer(knowledge, "dbL0A")),
            (MemorySpace.L0B, _integer(knowledge, "dbL0B")),
            (MemorySpace.L0C, _integer(knowledge, "dbL0C")),
        ),
        traversal=traversal,
    )


def base_plan_from_cann(
    m: int,
    n: int,
    k: int,
    knowledge: Mapping[str, int],
    **kwargs,
) -> TilingPlan:
    plan = plan_from_cann(m, n, k, knowledge, **kwargs)
    if decode_execution_graph(knowledge).split != SplitCoreMode.BASE:
        raise ValueError("base_plan_from_cann requires a non-Split-K graph")
    return plan


def simulate_cann_base(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    knowledge: Mapping[str, int],
    hardware: Hardware,
) -> SimulationResult:
    plan = base_plan_from_cann(
        m, n, k, knowledge,
        dtype=dtype, trans_a=trans_a, trans_b=trans_b,
    )
    return simulate(
        _matmul_operator(
            m, n, k, dtype, trans_a, trans_b, "ND", "ND", None, False
        ),
        plan,
        hardware,
    )
