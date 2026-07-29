#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path


warnings.filterwarnings("ignore", category=Warning, module="requests")


KNOWLEDGE_FIELDS = (
    "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
    "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
    "stepN", "iterateOrder", "stepKa", "stepKb", "dbL0A", "dbL0B",
    "dbL0C", "l2MTileCnt", "l2NTileCnt", "l2MTileBlock",
    "l2NTileBlock", "l2IterateOrder", "tilingEnable",
)

BANK_COLUMNS = {
    "tilingEnable": "bank_tiling_enable",
    "l2MTileCnt": "bank_l2_m_tile_count",
    "l2NTileCnt": "bank_l2_n_tile_count",
    "l2MTileBlock": "bank_l2_m_tile_block",
    "l2NTileBlock": "bank_l2_n_tile_block",
    "l2IterateOrder": "bank_l2_iterate_order",
}

EXTRA_COLUMNS = [
    *BANK_COLUMNS.values(),
    "search_template", "search_guidance", "search_model_score",
    "search_bottleneck", "search_rationale", "search_transition_gain",
    "search_resume_policy", "search_stop_reason",
    "search_model_cycles", "search_model_raw_ratio_vs_bank_seed",
    "search_model_ratio_vs_bank_seed", "search_model_calibration",
    "search_model_confidence", "search_model_breakdown",
    "search_hbm_bytes", "search_l2_bytes", "search_seed_key",
    "callback_tiling_sha256", "callback_tiling_bytes",
    "callback_tiling_key", "callback_block_dim", "callback_workspace_bytes",
    "callback_l2_cache_flag", "callback_base_an", "callback_base_ad",
    "callback_base_bn", "callback_base_bd",
    "callback_kernel_suffix", "callback_kernel_variant",
    "callback_kernel_family",
    "callback_derived_diff_vs_default",
    "callback_derived_diff_vs_bank_seed",
    "search_history_match",
]

DTYPE_NAME = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}

INPUT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}
OUTPUT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}

# These are the complete low-decimal suffixes dispatched by the installed
# CANN 8.1 MatMulV3 AscendC kernel on DAV C220. The suffix is authoritative:
# modes added by later CANN releases are deliberately not generated.
CANN81_KERNEL_VARIANTS = {
    0: ("BASE_UNALIGNED", "BASE"),
    1: ("BASE_ALIGNED", "BASE"),
    20: ("SINGLE_CORE_SPLIT_K_UNALIGNED", "SINGLE_CORE_SPLIT_K"),
    21: ("SINGLE_CORE_SPLIT_K_ALIGNED", "SINGLE_CORE_SPLIT_K"),
    30: ("DETERMINISTIC_SPLIT_K_UNALIGNED", "DETERMINISTIC_SPLIT_K"),
    31: ("DETERMINISTIC_SPLIT_K_ALIGNED", "DETERMINISTIC_SPLIT_K"),
    101: ("AL1_FULL_LOAD_ALIGNED", "AL1_FULL_LOAD"),
    200: ("BL1_FULL_LOAD_UNALIGNED", "BL1_FULL_LOAD"),
    201: ("BL1_FULL_LOAD_ALIGNED", "BL1_FULL_LOAD"),
    10200: ("BL1_FULL_LOAD_FIXPIPE_UNALIGNED", "BL1_FULL_LOAD_FIXPIPE"),
    10201: ("BL1_FULL_LOAD_FIXPIPE_ALIGNED", "BL1_FULL_LOAD_FIXPIPE"),
    20201: ("BL1_FULL_LOAD_VEC_NZ2ND", "BL1_FULL_LOAD_VEC_NZ2ND"),
}

# MatmulV3TilingData is TCubeTiling (50 words), L2CacheTilePara (5),
# seven run-info words, l2CacheFlag, then the ND->NZ vector tiling. These
# offsets are part of the public MatMulV3 tiling-data definition. Keep unknown
# trailing words in the hash even when a CANN release extends the structure.
DERIVED_WORDS = {
    62: "l2CacheFlag",
    63: "baseAN",
    64: "baseAD",
    65: "baseBN",
    66: "baseBD",
}


class SearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Workload:
    workload_id: str
    m: int
    n: int
    k: int
    dtype: str
    trans_a: bool
    trans_b: bool
    max_cores: int


@dataclass(frozen=True)
class Hardware:
    aic_cores: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    l1_bytes: int
    l2_bytes: int
    l2_bytes_per_cycle_per_core: float
    hbm_bytes_per_cycle_per_core: float


@dataclass(frozen=True)
class CallbackTiling:
    key: int
    block_dim: int
    workspaces: tuple[int, ...]
    blob: bytes
    sha256: str
    words: tuple[int, ...]
    cube: tuple[int, ...]
    l2: tuple[int, ...]
    knowledge: dict[str, int]
    derived: dict[str, int]


@dataclass
class Seed:
    default: CallbackTiling
    bank: CallbackTiling

    @property
    def key(self) -> int:
        return self.default.key

    @property
    def block_dim(self) -> int:
        return self.default.block_dim

    @property
    def workspace(self) -> int:
        return sum(self.default.workspaces)

    @property
    def knowledge(self) -> dict[str, int]:
        return self.default.knowledge


@dataclass(frozen=True)
class ModelEstimate:
    cycles: float
    hbm_bytes: float
    l2_bytes: float
    cube_cycles: float
    mte2_cycles: float
    mte1_cycles: float
    fixpipe_cycles: float
    nd2nz_cycles: float
    issue_cycles: float
    balance: float
    confidence: str

    def breakdown(self) -> str:
        values = {
            "cube": self.cube_cycles,
            "mte2": self.mte2_cycles,
            "mte1": self.mte1_cycles,
            "fixpipe": self.fixpipe_cycles,
            "nd2nz": self.nd2nz_cycles,
            "issue": self.issue_cycles,
            "balance": self.balance,
        }
        return json.dumps(values, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class BottleneckProfile:
    dominant: str
    output_tasks: int
    active_cores: int
    core_utilization: float
    round_efficiency: float
    l2_efficiency: float
    l2_working_set_ratio: float
    input_to_cube: float
    fixpipe_to_cube: float
    confidence: str

    def summary(self) -> str:
        return (
            f"dominant={self.dominant} "
            f"core_util={self.core_utilization:.3f} "
            f"round_eff={self.round_efficiency:.3f} "
            f"l2_eff={self.l2_efficiency:.3f} "
            f"l2_ws={self.l2_working_set_ratio:.3f} "
            f"input/cube={self.input_to_cube:.3f} "
            f"fix/cube={self.fixpipe_to_cube:.3f}"
        )


@dataclass
class CandidateProposal:
    knowledge: dict[str, int]
    guidance: str
    rationale: str
    transition_gain: float
    resume_policy: str = "require_existing"


@dataclass(frozen=True)
class MeasurementEvidence:
    ratio_vs_official: float
    ratio_vs_bank: float | None
    record_id: str


@dataclass
class State:
    row: dict[str, str]
    knowledge: dict[str, int]
    model_score: float
    normalized_score: float
    hbm_bytes: float
    l2_bytes: float
    template: str
    guidance: str = "analytical_beam"
    callback: CallbackTiling | None = None
    estimate: ModelEstimate | None = None


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def cube_k0(dtype: str) -> int:
    return 8 if dtype == "fp32" else 16


def base_k_alignment(workload: Workload) -> int:
    # TCubeTiling permits FP32 K0 alignment only for ND x transposed-ND.
    # All other layouts use 16-element baseK alignment.
    if workload.dtype == "fp32" and not workload.trans_a and workload.trans_b:
        return 8
    return 16


def effective_l1_bytes(hardware: Hardware) -> int:
    # Ascend910B3 reports 512 KiB minus one 256-byte bookkeeping block through
    # platform_info, while the official MatMulV3 tiler legally emits a full
    # 512 KiB A1+B1 plan. Match that allocator-visible KiB granularity.
    return align_up(hardware.l1_bytes, 1024)


def callback_derived_diff(
    left: CallbackTiling,
    right: CallbackTiling,
) -> str:
    names: list[str] = []
    for index, name in DERIVED_WORDS.items():
        if index >= len(left.words) or index >= len(right.words):
            continue
        if left.words[index] != right.words[index]:
            names.append(
                f"{name}:{left.words[index]}->{right.words[index]}"
            )
    if len(left.words) != len(right.words):
        names.append(f"word_count:{len(left.words)}->{len(right.words)}")
    return "|".join(names)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        raise SearchError(f"{path}: missing CSV header")
    return fields, rows


def load_workloads(path: Path) -> list[Workload]:
    _, rows = read_csv(path)
    workloads: list[Workload] = []
    for row in rows:
        dtype = row["dtype"].lower()
        workloads.append(
            Workload(
                workload_id=row.get("workload_id") or row["id"],
                m=int(row["m"]),
                n=int(row["n"]),
                k=int(row["k"]),
                dtype=dtype,
                trans_a=truthy(row.get("trans_a")),
                trans_b=truthy(row.get("trans_b")),
                max_cores=int(row.get("max_cores") or 24),
            )
        )
    return workloads


def decode_tiling_enable(tiling_key: int) -> int:
    # CANN 8.1 kernel key suffix is FIX,SPECIAL,FULL,SPLIT,MIX in decimal
    # positions 10^4..10^0. RuntimeKb omits MIX and encodes
    # split + 10*full + 1000*fix.
    suffix = tiling_key % 100000
    split = (suffix // 10) % 10
    full = (suffix // 100) % 10
    fix = (suffix // 10000) % 10
    return split + 10 * full + 1000 * fix


def kernel_suffix(tiling_key: int) -> int:
    return tiling_key % 100000


def kernel_variant(tiling_key: int) -> str:
    suffix = kernel_suffix(tiling_key)
    try:
        return CANN81_KERNEL_VARIANTS[suffix][0]
    except KeyError as exception:
        raise SearchError(
            f"CANN 8.1 MatMulV3 kernel has no dispatch for suffix {suffix}"
        ) from exception


def kernel_family(tiling_key: int) -> str:
    suffix = kernel_suffix(tiling_key)
    try:
        return CANN81_KERNEL_VARIANTS[suffix][1]
    except KeyError as exception:
        raise SearchError(
            f"CANN 8.1 MatMulV3 kernel has no dispatch for suffix {suffix}"
        ) from exception


def fix_mode(knowledge: dict[str, int]) -> int:
    return (knowledge["tilingEnable"] // 1000) % 10


def special_mode(knowledge: dict[str, int]) -> int:
    return (knowledge["tilingEnable"] // 10000) % 10


def split_mode(knowledge: dict[str, int]) -> int:
    return knowledge["tilingEnable"] % 10


def full_load_mode(knowledge: dict[str, int]) -> int:
    return (knowledge["tilingEnable"] // 10) % 10


def template_name(knowledge: dict[str, int]) -> str:
    split = split_mode(knowledge)
    full = full_load_mode(knowledge)
    fix = fix_mode(knowledge)
    if split == 3:
        return "DETERMINISTIC_SPLIT_K"
    if split == 2:
        return "SINGLE_CORE_SPLIT_K"
    if full == 1:
        return "AL1_FULL_LOAD"
    if full == 2:
        if fix == 1:
            return "BL1_FULL_LOAD_FIXPIPE"
        if fix == 2:
            return "BL1_FULL_LOAD_VEC_NZ2ND"
        return "BL1_FULL_LOAD"
    return "BASE"


def tensor_desc(name: str, shape: list[int], dtype: str) -> dict:
    return {
        "name": name,
        "shape": shape,
        "ori_shape": shape,
        "format": "ND",
        "ori_format": "ND",
        "dtype": DTYPE_NAME[dtype],
    }


def callback_knowledge(
    result: dict,
    cube: tuple[int, ...],
    l2: tuple[int, ...],
) -> dict[str, int]:
    return {
        "usedCoreNum": cube[0],
        "singleCoreM": cube[5],
        "singleCoreN": cube[6],
        "singleCoreK": cube[7],
        "baseM": cube[8],
        "baseN": cube[9],
        "baseK": cube[10],
        "depthA1": cube[11],
        "depthB1": cube[12],
        "stepM": cube[13],
        "stepN": cube[14],
        "iterateOrder": cube[17],
        "stepKa": cube[26],
        "stepKb": cube[27],
        "dbL0A": cube[30],
        "dbL0B": cube[31],
        "dbL0C": cube[32],
        "l2MTileCnt": l2[0],
        "l2NTileCnt": l2[1],
        "l2MTileBlock": l2[2],
        "l2NTileBlock": l2[3],
        "l2IterateOrder": l2[4],
        "tilingEnable": decode_tiling_enable(int(result["tiling_key"])),
    }


def parse_callback_result(result: dict) -> CallbackTiling:
    blob = bytes(result["tiling_data"])
    if len(blob) < 220 or len(blob) % 4:
        raise SearchError(
            f"unexpected MatMulV3 tiling_data size={len(blob)}"
        )
    cube = struct.unpack_from("<50i", blob, 0)
    l2 = struct.unpack_from("<5I", blob, 200)
    words = struct.unpack(f"<{len(blob) // 4}I", blob)
    derived = {
        name: int(words[index])
        for index, name in DERIVED_WORDS.items()
        if index < len(words)
    }
    workspaces = tuple(int(value) for value in (result.get("workspaces") or [0]))
    key = int(result["tiling_key"])
    kernel_variant(key)
    return CallbackTiling(
        key=key,
        block_dim=int(result["block_dim"]),
        workspaces=workspaces,
        blob=blob,
        sha256=hashlib.sha256(blob).hexdigest(),
        words=tuple(int(value) for value in words),
        cube=tuple(int(value) for value in cube),
        l2=tuple(int(value) for value in l2),
        knowledge=callback_knowledge(result, cube, l2),
        derived=derived,
    )


def invoke_official_callback(
    workload: Workload,
    injected: dict[str, int] | None = None,
) -> CallbackTiling:
    from tbe.common.utils import op_tiling

    shape_a = [workload.k, workload.m] if workload.trans_a else [workload.m, workload.k]
    shape_b = [workload.n, workload.k] if workload.trans_b else [workload.k, workload.n]
    attrs = [
        {"name": "transpose_x1", "dtype": "bool", "value": workload.trans_a},
        {"name": "transpose_x2", "dtype": "bool", "value": workload.trans_b},
        {"name": "offset_x", "dtype": "int", "value": 0},
        {"name": "enable_hf32", "dtype": "bool", "value": False},
    ]
    op_tiling._RT_BANK_CACHE = injected or {}
    result = op_tiling.do_op_tiling(
        "MatMulV3",
        {},
        [
            tensor_desc("x1", shape_a, workload.dtype),
            tensor_desc("x2", shape_b, workload.dtype),
            None,
            None,
        ],
        [tensor_desc("y", [workload.m, workload.n], workload.dtype)],
        attrs=attrs,
    )
    return parse_callback_result(result)


def parse_seed(workload: Workload) -> Seed:
    default = invoke_official_callback(workload)
    bank = invoke_official_callback(workload, default.knowledge)
    return Seed(default=default, bank=bank)


def raw_knowledge(row: dict[str, str]) -> dict[str, int]:
    return {
        "usedCoreNum": int(row["used_core_num"]),
        "singleCoreM": int(row["single_core_m"]),
        "singleCoreN": int(row["single_core_n"]),
        "singleCoreK": int(row["single_core_k"]),
        "baseM": int(row["base_m"]),
        "baseN": int(row["base_n"]),
        "baseK": int(row["base_k"]),
        "depthA1": int(row["depth_a1"]),
        "depthB1": int(row["depth_b1"]),
        "stepM": int(row["step_m"]),
        "stepN": int(row["step_n"]),
        "iterateOrder": int(row["iterate_order"]),
        "stepKa": int(row["step_ka"]),
        "stepKb": int(row["step_kb"]),
        "dbL0A": int(row["db_l0a"]),
        "dbL0B": int(row["db_l0b"]),
        "dbL0C": int(row["db_l0c"]),
        "l2MTileCnt": 1,
        "l2NTileCnt": 1,
        "l2MTileBlock": 1,
        "l2NTileBlock": 1,
        "l2IterateOrder": 0,
        "tilingEnable": 0,
    }


def l2_base_schedule_legal(
    workload: Workload,
    knowledge: dict[str, int],
) -> bool:
    m_total = ceil_div(workload.m, knowledge["singleCoreM"])
    n_total = ceil_div(workload.n, knowledge["singleCoreN"])
    m_block = knowledge["l2MTileBlock"]
    n_block = knowledge["l2NTileBlock"]
    if m_block == 0 or n_block == 0:
        return (
            m_block == 0
            and n_block == 0
            and knowledge["l2MTileCnt"] == 1
            and knowledge["l2NTileCnt"] == 1
        )
    if (
        knowledge["l2MTileCnt"] != ceil_div(m_total, m_block)
        or knowledge["l2NTileCnt"] != ceil_div(n_total, n_block)
    ):
        return False
    m_tail = m_total - (knowledge["l2MTileCnt"] - 1) * m_block
    n_tail = n_total - (knowledge["l2NTileCnt"] - 1) * n_block
    return 1 <= m_tail <= m_block and 1 <= n_tail <= n_block


def tiling_enable_legal(knowledge: dict[str, int]) -> bool:
    split = split_mode(knowledge)
    full = full_load_mode(knowledge)
    fix = fix_mode(knowledge)
    special = special_mode(knowledge)
    if special != 0 or split not in (0, 2, 3):
        return False
    if full not in (0, 1, 2) or fix not in (0, 1, 2):
        return False
    if split != 0:
        return full == 0 and fix == 0
    if full == 0:
        return fix == 0
    if full == 1:
        return fix == 0
    return True


def hard_legal(
    workload: Workload,
    knowledge: dict[str, int],
    hardware: Hardware,
) -> bool:
    required_positive = (
        "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
        "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
        "stepN", "stepKa", "stepKb", "dbL0A", "dbL0B", "dbL0C",
        "l2MTileCnt", "l2NTileCnt",
    )
    if any(knowledge[name] <= 0 for name in required_positive):
        return False
    if knowledge["l2MTileBlock"] < 0 or knowledge["l2NTileBlock"] < 0:
        return False
    if not tiling_enable_legal(knowledge):
        return False
    if knowledge["usedCoreNum"] > min(workload.max_cores, hardware.aic_cores):
        return False
    if (
        knowledge["iterateOrder"] not in (0, 1)
        or knowledge["l2IterateOrder"] not in (0, 1, 2)
    ):
        return False
    if any(knowledge[name] not in (1, 2) for name in ("dbL0A", "dbL0B", "dbL0C")):
        return False

    k0 = base_k_alignment(workload)
    if (
        knowledge["baseM"] % 16
        or knowledge["baseN"] % 16
        or knowledge["baseK"] % k0
    ):
        return False

    in_bytes = INPUT_BYTES[workload.dtype]
    l0a = (
        knowledge["baseM"] * knowledge["baseK"] * in_bytes
        * knowledge["dbL0A"]
    )
    l0b = (
        knowledge["baseN"] * knowledge["baseK"] * in_bytes
        * knowledge["dbL0B"]
    )
    l0c = knowledge["baseM"] * knowledge["baseN"] * 4 * knowledge["dbL0C"]
    if (
        l0a > hardware.l0a_bytes
        or l0b > hardware.l0b_bytes
        or l0c > hardware.l0c_bytes
    ):
        return False

    one_a = knowledge["stepM"] * knowledge["stepKa"]
    one_b = knowledge["stepN"] * knowledge["stepKb"]
    if knowledge["depthA1"] % one_a or knowledge["depthB1"] % one_b:
        return False
    if knowledge["depthA1"] // one_a not in (1, 2):
        return False
    if knowledge["depthB1"] // one_b not in (1, 2):
        return False
    a1 = (
        knowledge["baseM"] * knowledge["baseK"]
        * knowledge["depthA1"] * in_bytes
    )
    b1 = (
        knowledge["baseN"] * knowledge["baseK"]
        * knowledge["depthB1"] * in_bytes
    )
    if a1 + b1 > effective_l1_bytes(hardware):
        return False

    if (
        knowledge["stepKa"] % knowledge["stepKb"]
        and knowledge["stepKb"] % knowledge["stepKa"]
    ):
        return False
    split = split_mode(knowledge)
    full = full_load_mode(knowledge)
    fix = fix_mode(knowledge)

    if split == 0 and full == 0:
        if (
            knowledge["singleCoreK"] != workload.k
            or knowledge["singleCoreM"] > knowledge["baseM"]
            or knowledge["singleCoreN"] > knowledge["baseN"]
            or knowledge["baseM"] > align_up(knowledge["singleCoreM"], 16)
            or knowledge["baseN"] > align_up(knowledge["singleCoreN"], 16)
            or knowledge["baseK"] > align_up(
                knowledge["singleCoreK"], k0
            )
            or not l2_base_schedule_legal(workload, knowledge)
        ):
            return False
    elif split == 2:
        if (
            knowledge["singleCoreK"]
            != knowledge["stepKa"] * knowledge["baseK"]
            or knowledge["stepKa"] != knowledge["stepKb"]
            or knowledge["singleCoreK"] >= workload.k
            or ceil_div(workload.k, knowledge["singleCoreK"]) < 2
            or knowledge["singleCoreM"]
            < knowledge["stepM"] * knowledge["baseM"]
        ):
            return False
        if (
            knowledge["l2IterateOrder"] == 1
            and knowledge["singleCoreN"]
            < knowledge["stepN"] * knowledge["baseN"]
        ):
            return False
    elif split == 3:
        expected_base_k = 256 // in_bytes
        if (
            knowledge["baseM"], knowledge["baseN"], knowledge["baseK"]
        ) != (128, 128, expected_base_k):
            return False
        mk33 = (
            knowledge["stepM"], knowledge["stepN"],
            knowledge["stepKa"], knowledge["stepKb"],
            knowledge["depthA1"], knowledge["depthB1"],
            knowledge["singleCoreM"],
        ) == (3, 1, 3, 3, 9, 6, 384)
        nk33 = (
            knowledge["stepM"], knowledge["stepN"],
            knowledge["stepKa"], knowledge["stepKb"],
            knowledge["depthA1"], knowledge["depthB1"],
            knowledge["singleCoreN"],
        ) == (1, 3, 3, 3, 6, 9, 384)
        k_chunks = ceil_div(workload.k, knowledge["singleCoreK"])
        if (
            not (mk33 or nk33)
            or knowledge["singleCoreK"] != 3 * knowledge["baseK"]
            or k_chunks < 2
            or knowledge["usedCoreNum"] > k_chunks
        ):
            return False
    elif split == 0 and full == 1:
        if (
            workload.dtype != "fp32"
            or workload.trans_a
            or not workload.trans_b
            or workload.m > 16
            or workload.n <= 16
            or workload.n > 16 * hardware.aic_cores
            or workload.k < 4096
            or workload.k % (512 // in_bytes)
            or knowledge["baseM"] != 16
            or knowledge["singleCoreM"] < workload.m
            or knowledge["singleCoreM"] > 16
            or knowledge["singleCoreN"] < knowledge["baseN"]
            or knowledge["singleCoreK"] < workload.k
            or knowledge["stepM"] != 1
            or knowledge["stepN"] != 1
            or knowledge["stepKa"] != ceil_div(
                workload.k, knowledge["baseK"]
            )
            or knowledge["stepKb"] != 1
            or knowledge["depthA1"] != knowledge["stepKa"]
            or knowledge["depthB1"] not in (1, 2)
            or not l2_base_schedule_legal(workload, knowledge)
        ):
            return False
        al1_bytes = (
            align_up(workload.m, 16)
            * align_up(workload.k, 32 // in_bytes)
            * in_bytes
        )
        if al1_bytes + b1 > effective_l1_bytes(hardware):
            return False
    elif split == 0 and full == 2:
        if (
            workload.m <= 16 * max(workload.k, workload.n)
            or workload.k > 256
            or knowledge["singleCoreK"] < workload.k
            or knowledge["singleCoreM"] < knowledge["baseM"]
            or knowledge["singleCoreN"] < workload.n
            or knowledge["stepM"] != 1
            or knowledge["stepN"] != ceil_div(
                workload.n, knowledge["baseN"]
            )
            or knowledge["stepKa"] != ceil_div(
                workload.k, knowledge["baseK"]
            )
            or knowledge["stepKb"] != knowledge["stepKa"]
            or knowledge["depthB1"]
            != knowledge["stepN"] * knowledge["stepKb"]
            or not l2_base_schedule_legal(workload, knowledge)
        ):
            return False
        c0 = 32 // in_bytes
        resident_b = (
            align_up(workload.k, 16)
            * align_up(knowledge["singleCoreN"], c0)
            * in_bytes
        )
        if b1 < resident_b:
            return False
        fixpipe_bound = bl1_fixpipe_bound(workload)
        if fix == 1 and not fixpipe_bound:
            return False
        if fix == 2 and (
            workload.dtype != "fp32"
            or workload.trans_a
            or workload.n > 192
            or workload.k % c0
        ):
            return False
        if fix in (1, 2) and (
            knowledge["l2MTileBlock"] != 0
            or knowledge["l2NTileBlock"] != 0
        ):
            return False
    else:
        return False

    if split in (2, 3):
        if (
            knowledge["l2MTileBlock"] <= 0
            or knowledge["l2NTileBlock"] <= 0
        ):
            return False
    if split == 3:
        k_chunks = ceil_div(workload.k, knowledge["singleCoreK"])
        if (
            knowledge["usedCoreNum"] > k_chunks
            or knowledge["usedCoreNum"] <= 0
        ):
            return False
    return True


def axis_values(total: int, target: int) -> list[int]:
    values = {
        1,
        total,
        max(1, min(total, target)),
        max(1, min(total, target - 1)),
        max(1, min(total, target + 1)),
        max(1, min(total, ceil_div(target, 2))),
        max(1, min(total, target * 2)),
    }
    return sorted(values)


def l2_schedules(
    workload: Workload,
    knowledge: dict[str, int],
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    m_total = ceil_div(workload.m, knowledge["singleCoreM"])
    n_total = ceil_div(workload.n, knowledge["singleCoreN"])
    seed_k = seed.knowledge
    target_m_elements = seed_k["singleCoreM"] * seed_k["l2MTileBlock"]
    target_n_elements = seed_k["singleCoreN"] * seed_k["l2NTileBlock"]
    target_m = ceil_div(target_m_elements, knowledge["singleCoreM"])
    target_n = ceil_div(target_n_elements, knowledge["singleCoreN"])

    schedules: list[tuple[float, dict[str, int]]] = []
    for m_block in axis_values(m_total, target_m):
        for n_block in axis_values(n_total, target_n):
            for order in (seed_k["l2IterateOrder"], 1 - seed_k["l2IterateOrder"]):
                candidate = dict(knowledge)
                candidate["l2MTileBlock"] = m_block
                candidate["l2NTileBlock"] = n_block
                candidate["l2MTileCnt"] = ceil_div(m_total, m_block)
                candidate["l2NTileCnt"] = ceil_div(n_total, n_block)
                candidate["l2IterateOrder"] = order

                m_extent = min(workload.m, m_block * knowledge["singleCoreM"])
                n_extent = min(workload.n, n_block * knowledge["singleCoreN"])
                in_bytes = INPUT_BYTES[workload.dtype]
                out_bytes = OUTPUT_BYTES[workload.dtype]
                working_set = (
                    m_extent * workload.k * in_bytes
                    + workload.k * n_extent * in_bytes
                    + m_extent * n_extent * out_bytes
                )
                # MatMulV3 scales its empirically safe 100 MiB working-set
                # threshold linearly from the 192 MiB Ascend910B L2.
                l2_working_set_limit = (
                    hardware.l2_bytes
                    / float(192 * 1024 * 1024)
                    * 100 * 1024 * 1024
                )
                capacity_overflow = max(
                    0.0, working_set / max(1.0, l2_working_set_limit) - 1.0
                )
                m_tail = m_total - (candidate["l2MTileCnt"] - 1) * m_block
                n_tail = n_total - (candidate["l2NTileCnt"] - 1) * n_block
                # Match MatmulV3BaseTiling::InitL2SplitParams: the outer
                # axis permits five-way conflict on a 20-AIC product; the
                # unfavorable inner layout permits four-way conflict.
                if knowledge["baseN"] >= knowledge["baseM"]:
                    m_conflict = 4 if workload.trans_a else 5
                    n_conflict = 5
                else:
                    m_conflict = 5
                    n_conflict = 4 if not workload.trans_b else 5
                tail_penalty = 0.0
                if (
                    m_total > m_block
                    and m_tail * m_conflict < hardware.aic_cores
                ):
                    tail_penalty += (
                        hardware.aic_cores - m_tail * m_conflict
                    ) / hardware.aic_cores
                if (
                    n_total > n_block
                    and n_tail * n_conflict < hardware.aic_cores
                ):
                    tail_penalty += (
                        hardware.aic_cores - n_tail * n_conflict
                    ) / hardware.aic_cores
                distance = (
                    abs(m_block - target_m) / max(1, target_m)
                    + abs(n_block - target_n) / max(1, target_n)
                )
                schedules.append(
                    (5.0 * capacity_overflow + 2.0 * tail_penalty + 0.01 * distance, candidate)
                )
    schedules.sort(key=lambda item: item[0])
    unique: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for _, candidate in schedules:
        signature = tuple(candidate[field] for field in KNOWLEDGE_FIELDS)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(candidate)
        if len(unique) == 3:
            break
    return unique


def deduplicate_knowledge(
    candidates: list[dict[str, int]],
) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        signature = knowledge_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(candidate)
    return result


def partition_pairs(
    workload: Workload,
    hardware: Hardware,
    max_rounds: int = 8,
) -> list[tuple[int, int]]:
    core_limit = max(
        1, min(workload.max_cores, hardware.aic_cores)
    )
    max_m_parts = max(1, min(8 * core_limit, ceil_div(workload.m, 16)))
    max_n_parts = max(1, min(8 * core_limit, ceil_div(workload.n, 16)))
    pairs: set[tuple[int, int]] = {(1, 1)}
    for rounds in range(1, max_rounds + 1):
        target = rounds * core_limit
        ideal_m = math.sqrt(
            target * max(1.0, workload.m) / max(1.0, workload.n)
        )
        probes = {
            1,
            min(max_m_parts, target),
            max(1, min(max_m_parts, int(ideal_m))),
            max(1, min(max_m_parts, int(math.ceil(ideal_m)))),
        }
        for divisor in range(1, int(math.sqrt(target)) + 1):
            if target % divisor == 0:
                probes.add(min(max_m_parts, divisor))
                probes.add(min(max_m_parts, target // divisor))
        for m_parts in probes:
            for n_parts in {
                max(1, min(max_n_parts, target // max(1, m_parts))),
                max(1, min(max_n_parts, ceil_div(target, max(1, m_parts)))),
            }:
                pairs.add((m_parts, n_parts))
    return sorted(
        pairs,
        key=lambda pair: (
            abs(pair[0] * pair[1] - core_limit),
            abs(
                pair[0] / max(1.0, pair[1])
                - workload.m / max(1.0, workload.n)
            ),
            pair,
        ),
    )


def base_candidate_space(
    workload: Workload,
    seed: Seed,
    raw_rows: list[dict[str, str]],
    hardware: Hardware,
    beam_width: int,
) -> list[dict[str, int]]:
    anchors: list[dict[str, int]] = []
    if template_name(seed.bank.knowledge) == "BASE":
        anchors.append(dict(seed.bank.knowledge))
    raw_limit = max(beam_width * 2, 32)
    for row in raw_rows:
        if len(anchors) >= raw_limit:
            break
        if (
            not truthy(row.get("valid"))
            or row.get("candidate_role") != "searched"
            or row.get("execution_mode") != "base_iterate_all"
        ):
            continue
        anchors.append(raw_knowledge(row))

    candidates: list[dict[str, int]] = []
    for anchor in deduplicate_knowledge(anchors):
        single_shapes = {
            (anchor["baseM"], anchor["baseN"]),
        }
        if (
            anchor["singleCoreM"] <= anchor["baseM"]
            and anchor["singleCoreN"] <= anchor["baseN"]
        ):
            single_shapes.add(
                (anchor["singleCoreM"], anchor["singleCoreN"])
            )
        for single_m, single_n in single_shapes:
            candidate = dict(anchor)
            candidate.update(
                {
                    "singleCoreM": single_m,
                    "singleCoreN": single_n,
                    "singleCoreK": workload.k,
                    "usedCoreNum": min(
                        hardware.aic_cores,
                        workload.max_cores,
                        ceil_div(workload.m, single_m)
                        * ceil_div(workload.n, single_n),
                    ),
                    "tilingEnable": 0,
                }
            )
            for scheduled in l2_schedules(
                workload, candidate, seed, hardware
            ):
                for db_a, db_b, db_c in {
                    (
                        scheduled["dbL0A"],
                        scheduled["dbL0B"],
                        scheduled["dbL0C"],
                    ),
                    (1, 1, 1),
                    (2, 2, 1),
                    (2, 2, 2),
                }:
                    variant = dict(scheduled)
                    variant.update(
                        dbL0A=db_a,
                        dbL0B=db_b,
                        dbL0C=db_c,
                    )
                    if hard_legal(workload, variant, hardware):
                        candidates.append(variant)
    return deduplicate_knowledge(candidates)


def skinny_n_one_block_per_core_geometry(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> bool:
    """Check the output-grid geometry shared by the skinny-N experiments."""
    core_limit = min(workload.max_cores, hardware.aic_cores)
    balanced_m = align_up(ceil_div(workload.m, core_limit), 16)
    return (
        workload.dtype == "fp16"
        and not workload.trans_a
        and not workload.trans_b
        and template_name(seed.bank.knowledge) == "BASE"
        and 16 < workload.n <= 32
        and workload.k % 64 == 0
        and 128 < balanced_m <= 256
        and ceil_div(workload.m, balanced_m) == core_limit
    )


def skinny_n_large_k_applicable(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> bool:
    """Select the K=16384 BASE family supported by five NPU shapes."""
    return (
        workload.k == 16384
        and skinny_n_one_block_per_core_geometry(
            workload, seed, hardware
        )
    )


def skinny_n_boundary_k16384_applicable(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> bool:
    """Select the N=33..47 boundary anchored by the N=33 NPU run.

    This is intentionally narrower than the broad BASE transition frontier.
    It covers the adjacent one-N-block geometry where the official seed keeps
    a smaller M tile and extra L2 rounds, while the measured anchor assigns
    exactly one M block per AIC.
    """
    core_limit = min(workload.max_cores, hardware.aic_cores)
    balanced_m = align_up(ceil_div(workload.m, core_limit), 16)
    return (
        workload.dtype == "fp16"
        and not workload.trans_a
        and not workload.trans_b
        and template_name(seed.bank.knowledge) == "BASE"
        and workload.k == 16384
        and 32 < workload.n < 48
        and 128 < balanced_m <= 256
        and ceil_div(workload.m, balanced_m) == core_limit
    )


def skinny_n_transition48_k16384_applicable(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> bool:
    """Select the exact N=48 crossover for a baseN=64 confirmation run."""
    core_limit = min(workload.max_cores, hardware.aic_cores)
    balanced_m = align_up(ceil_div(workload.m, core_limit), 16)
    return (
        workload.dtype == "fp16"
        and not workload.trans_a
        and not workload.trans_b
        and template_name(seed.bank.knowledge) == "BASE"
        and workload.k == 16384
        and workload.n == 48
        and 128 < balanced_m <= 256
        and ceil_div(workload.m, balanced_m) == core_limit
    )


def skinny_n_boundary64_k16384_applicable(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> bool:
    """Select the adjacent baseN=64 one-N-block validation range."""
    core_limit = min(workload.max_cores, hardware.aic_cores)
    balanced_m = align_up(ceil_div(workload.m, core_limit), 16)
    return (
        workload.dtype == "fp16"
        and not workload.trans_a
        and not workload.trans_b
        and template_name(seed.bank.knowledge) == "BASE"
        and workload.k == 16384
        and 48 < workload.n <= 64
        and 128 < balanced_m <= 256
        and ceil_div(workload.m, balanced_m) == core_limit
    )


def skinny_n_low_k_causal_applicable(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> bool:
    """Return whether lower-K skinny-N should receive new causal probes.

    The preregistered lower-K probes were measured in net_log2. They did not
    beat both the official operator and the bank seed, so the low-K
    extrapolation is closed rather than expanded.
    """
    return False


def skinny_n_low_k_falsified_applicable(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> bool:
    """Select the lower-K skinny-N region rejected by the NPU campaign."""
    return (
        8192 <= workload.k < 16384
        and skinny_n_one_block_per_core_geometry(
            workload, seed, hardware
        )
    )


def official_l1_for_base(
    workload: Workload,
    knowledge: dict[str, int],
    hardware: Hardware,
) -> tuple[int, int, int, int]:
    """Reproduce MatMulV3 CalL1Tiling for a proposed BASE shape."""
    total_l1 = effective_l1_bytes(hardware)
    in_bytes = INPUT_BYTES[workload.dtype]
    base_m = knowledge["baseM"]
    base_n = knowledge["baseN"]
    base_k = knowledge["baseK"]
    depth_a = total_l1 // 2 // base_m // base_k // in_bytes
    depth_b = total_l1 // 2 // base_n // base_k // in_bytes
    size_a = depth_a * base_m * base_k * in_bytes
    size_b = depth_b * base_n * base_k * in_bytes
    if size_a + size_b > total_l1:
        if base_m <= base_n:
            depth_a //= 2
        else:
            depth_b //= 2
    step_ka = max(1, depth_a // 2)
    step_kb = max(1, depth_b // 2)
    if step_ka >= step_kb:
        step_ka = max(step_kb, step_ka // step_kb * step_kb)
    else:
        step_kb = max(step_ka, step_kb // step_ka * step_ka)
    return 2 * step_ka, 2 * step_kb, step_ka, step_kb


def skinny_n_large_k_candidate_space(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    """Four ablations around the measured 20-core skinny-N winner.

    The official no-transpose seed commonly chooses baseM at or below 128,
    while the callback also accepts larger capacity-safe BASE tiles. For one
    N block, use the smallest aligned M block that creates exactly one output
    block per AIC, then isolate the effects of L1 allocation, L0C double
    buffering, and L2 traversal. This is a constrained local extension of the
    official BASE schedule, not a generic raw-MultiCoreMatmulTiling sweep.
    """
    if not (
        skinny_n_large_k_applicable(workload, seed, hardware)
        or skinny_n_low_k_causal_applicable(
            workload, seed, hardware
        )
    ):
        return []

    core_limit = min(workload.max_cores, hardware.aic_cores)
    base_m = align_up(ceil_div(workload.m, core_limit), 16)
    base_n = align_up(workload.n, 16)
    base_k = 64
    m_blocks = ceil_div(workload.m, base_m)
    n_blocks = ceil_div(workload.n, base_n)
    m_tile_block = min(4, m_blocks)
    n_tile_block = min(max(1, core_limit // 4), n_blocks)

    common = dict(seed.bank.knowledge)
    common.update(
        {
            "usedCoreNum": min(core_limit, m_blocks * n_blocks),
            "singleCoreM": base_m,
            "singleCoreN": base_n,
            "singleCoreK": workload.k,
            "baseM": base_m,
            "baseN": base_n,
            "baseK": base_k,
            "stepM": 1,
            "stepN": 1,
            "iterateOrder": 0,
            "dbL0A": 2,
            "dbL0B": 2,
            "l2MTileCnt": ceil_div(m_blocks, m_tile_block),
            "l2NTileCnt": ceil_div(n_blocks, n_tile_block),
            "l2MTileBlock": m_tile_block,
            "l2NTileBlock": n_tile_block,
            "tilingEnable": 0,
        }
    )

    official_depth_a, official_depth_b, official_step_a, official_step_b = (
        official_l1_for_base(workload, common, hardware)
    )
    balance_only = dict(common)
    balance_only.update(
        {
            "depthA1": official_depth_a,
            "depthB1": official_depth_b,
            "stepKa": official_step_a,
            "stepKb": official_step_b,
            "dbL0C": 1,
            "l2IterateOrder": 0,
        }
    )

    # N occupies one 32-column block. Keep a double-buffered four-step B
    # window, then select the largest supported A depth that fits the exact
    # platform L1 capacity. baseM=256 cannot hold depthA1=16 on 910B3.
    rebalance_depth_a = 16
    rebalanced_b_bytes = base_n * base_k * 8 * INPUT_BYTES[workload.dtype]
    rebalanced_a_bytes = (
        base_m * base_k * rebalance_depth_a
        * INPUT_BYTES[workload.dtype]
    )
    if rebalanced_a_bytes + rebalanced_b_bytes > effective_l1_bytes(hardware):
        rebalance_depth_a = 8
    l1_rebalanced = dict(common)
    l1_rebalanced.update(
        {
            "depthA1": rebalance_depth_a,
            "depthB1": 8,
            "stepKa": rebalance_depth_a // 2,
            "stepKb": 4,
            "dbL0C": 1,
            "l2IterateOrder": 0,
        }
    )
    l0c_double_buffered = dict(l1_rebalanced)
    l0c_double_buffered["dbL0C"] = 2
    l2_reordered = dict(l0c_double_buffered)
    l2_reordered["l2IterateOrder"] = 1

    return [
        candidate
        for candidate in (
            balance_only,
            l1_rebalanced,
            l0c_double_buffered,
            l2_reordered,
        )
        if hard_legal(workload, candidate, hardware)
    ]


def learned_skinny_n_k16384_schedule(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    """Apply the schedule rule supported by the K=16384 holdouts.

    For baseM below 256, L1 rebalance alone was the stable winner. At
    baseM=256 the A-side L1 depth is capacity-limited to eight and the
    measured L0C/L2 overlap transition is required.
    """
    if not skinny_n_large_k_applicable(
        workload, seed, hardware
    ):
        return []
    candidates = skinny_n_large_k_candidate_space(
        workload, seed, hardware
    )
    if len(candidates) != 4:
        return []
    return [candidates[-1] if candidates[1]["baseM"] == 256 else candidates[1]]


def learned_skinny_n_boundary_k16384_schedule(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    """Apply the measured one-block-per-AIC boundary schedule.

    Evidence: skinny_n_boundary_n33 measured 1.17x faster than official and
    1.18x faster than the bank seed. N=40 and N=47 independently preserve the
    baseN=48 schedule.
    """
    if not skinny_n_boundary_k16384_applicable(
        workload, seed, hardware
    ):
        return []
    return one_block_per_aic_boundary_schedule(
        workload, seed, hardware, base_n=48, depth_b=40
    )


def learned_skinny_n_transition48_k16384_schedule(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    """Test N=48 with the baseN=64 packet used successfully at N=49..64.

    The previous baseN=48 latency came from a different profiling run than its
    latest controls, so its apparent regression is not a valid paired result.
    This one-candidate crossover test is the adjacent successful schedule, not
    a new parameter sweep.
    """
    if not skinny_n_transition48_k16384_applicable(
        workload, seed, hardware
    ):
        return []
    return one_block_per_aic_boundary_schedule(
        workload, seed, hardware, base_n=64, depth_b=32
    )


def one_block_per_aic_boundary_schedule(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    *,
    base_n: int,
    depth_b: int,
) -> list[dict[str, int]]:
    """Build one exact BASE schedule for an aligned skinny-N band."""
    core_limit = min(workload.max_cores, hardware.aic_cores)
    base_m = align_up(ceil_div(workload.m, core_limit), 16)
    base_k = 64
    m_blocks = ceil_div(workload.m, base_m)
    n_blocks = ceil_div(workload.n, base_n)
    knowledge = dict(seed.bank.knowledge)
    knowledge.update(
        {
            "usedCoreNum": min(core_limit, m_blocks * n_blocks),
            "singleCoreM": base_m,
            "singleCoreN": base_n,
            "singleCoreK": workload.k,
            "baseM": base_m,
            "baseN": base_n,
            "baseK": base_k,
            "depthA1": 8,
            "depthB1": depth_b,
            "stepM": 1,
            "stepN": 1,
            "iterateOrder": 0,
            "stepKa": 4,
            "stepKb": depth_b // 2,
            "dbL0A": 2,
            "dbL0B": 2,
            "dbL0C": 1,
            "l2MTileCnt": 1,
            "l2NTileCnt": 1,
            "l2MTileBlock": m_blocks,
            "l2NTileBlock": n_blocks,
            "l2IterateOrder": 0,
            "tilingEnable": 0,
        }
    )
    return [knowledge] if hard_legal(workload, knowledge, hardware) else []


def learned_skinny_n_boundary64_k16384_schedule(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    """Test whether the measured boundary rule survives baseN 48 -> 64.

    N=49/56/64 are preregistered before NPU execution. CANN 8.1 emits a
    baseN=64 bank seed for all three shapes, and accepts this exact 23-field
    one-block-per-AIC schedule with an L1 8x32 packet split.
    """
    if not skinny_n_boundary64_k16384_applicable(
        workload, seed, hardware
    ):
        return []
    return one_block_per_aic_boundary_schedule(
        workload, seed, hardware, base_n=64, depth_b=32
    )


def attention_score_k128_l2_candidate_space(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    """Restore source-guided L2 grouping for the shallow-K transB case.

    The 1024x1024x128 attention-score workload is dominated by short-K input
    movement and output-grid scheduling. Earlier NPU evidence showed that the
    useful action was not the core-grid transition alone; the candidate must
    also split the M blocks into L2 rounds of four rows. Keep this as a narrow
    attention-score experiment, not a broad BASE rule.
    """
    if not (
        workload.workload_id == "attention_score_1024"
        and workload.dtype == "fp16"
        and not workload.trans_a
        and workload.trans_b
        and workload.m == 1024
        and workload.n == 1024
        and workload.k == 128
        and template_name(seed.bank.knowledge) == "BASE"
    ):
        return []
    candidates: list[dict[str, int]] = []
    for base_m, base_n, m_tile_block, n_tile_block in (
        (128, 208, 4, 5),
        (112, 256, 4, 4),
    ):
        m_blocks = ceil_div(workload.m, base_m)
        n_blocks = ceil_div(workload.n, base_n)
        knowledge = dict(seed.bank.knowledge)
        knowledge.update(
            {
                "usedCoreNum": min(
                    workload.max_cores,
                    hardware.aic_cores,
                    m_blocks * n_blocks,
                ),
                "singleCoreM": base_m,
                "singleCoreN": base_n,
                "singleCoreK": workload.k,
                "baseM": base_m,
                "baseN": base_n,
                "baseK": 64,
                "stepM": 1,
                "stepN": 1,
                "iterateOrder": 0,
                "dbL0A": 2,
                "dbL0B": 2,
                "dbL0C": 1,
                "l2MTileCnt": ceil_div(m_blocks, m_tile_block),
                "l2NTileCnt": ceil_div(n_blocks, n_tile_block),
                "l2MTileBlock": m_tile_block,
                "l2NTileBlock": n_tile_block,
                "l2IterateOrder": 0,
                "tilingEnable": 0,
            }
        )
        depth_a, depth_b, step_a, step_b = official_l1_for_base(
            workload, knowledge, hardware
        )
        knowledge.update(
            {
                "depthA1": depth_a,
                "depthB1": depth_b,
                "stepKa": step_a,
                "stepKb": step_b,
            }
        )
        if hard_legal(workload, knowledge, hardware):
            candidates.append(knowledge)
    return deduplicate_knowledge(candidates)


def skinny_n_low_k_causal_space(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    """Isolate L2 grouping, then L1 depth, on the lower-K holdouts."""
    if not skinny_n_low_k_causal_applicable(
        workload, seed, hardware
    ):
        return []
    candidates = skinny_n_large_k_candidate_space(
        workload, seed, hardware
    )
    return candidates[:2]


def focused_candidate_guidance(knowledge: dict[str, int]) -> str:
    if knowledge["depthB1"] >= 32:
        return "skinny_n_ablation_core_balance"
    if knowledge["dbL0C"] == 1:
        return "skinny_n_ablation_l1_rebalance"
    if knowledge["l2IterateOrder"] == 0:
        return "skinny_n_ablation_l0c_double_buffer"
    return "skinny_n_ablation_l2_order"


def official_seed_order_fields(
    workload: Workload,
    seed: Seed,
) -> tuple[str, ...]:
    family = template_name(seed.bank.knowledge)
    if family == "BASE":
        return ("iterateOrder", "l2IterateOrder")
    if (
        family == "DETERMINISTIC_SPLIT_K"
        and deterministic_split_k_supported_range(workload)
        and workload.m % 128 == 0
        and workload.n % 128 == 0
        and workload.k % 128 == 0
    ):
        return ("iterateOrder",)
    return ()


def official_seed_order_candidate_space(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    """Source-guided traversal ablations around the exact RuntimeKb seed.

    BASE exposes both loop orders. The public MatMulV3 source fixes the
    single-core split-K outer loop and ties full-load traversal to operand
    residency, so those templates are not mutated. Deterministic split-K gets
    only the measured iterateOrder ablation, and only when M/N/K have no
    128-element tile tail.
    """

    candidates: list[dict[str, int]] = []
    for field in official_seed_order_fields(workload, seed):
        candidate = dict(seed.bank.knowledge)
        candidate[field] = 1 - candidate[field]
        if hard_legal(workload, candidate, hardware):
            candidates.append(candidate)
    return deduplicate_knowledge(candidates)


def official_local_guidance(
    knowledge: dict[str, int],
    seed: Seed,
) -> str:
    seed_knowledge = seed.bank.knowledge
    if (
        knowledge["baseM"],
        knowledge["baseN"],
        knowledge["baseK"],
    ) != (
        seed_knowledge["baseM"],
        seed_knowledge["baseN"],
        seed_knowledge["baseK"],
    ):
        return focused_candidate_guidance(knowledge)
    if knowledge["iterateOrder"] != seed_knowledge["iterateOrder"]:
        return "official_seed_iterate_order_ablation"
    if knowledge["l2IterateOrder"] != seed_knowledge["l2IterateOrder"]:
        return "official_seed_l2_order_ablation"
    return "official_seed_local_control"


def al1_full_load_applicable(
    workload: Workload,
    hardware: Hardware,
) -> bool:
    if (
        workload.dtype != "fp32"
        or workload.trans_a
        or not workload.trans_b
        or workload.m > 16
        or workload.n <= 16
        or workload.n > 16 * hardware.aic_cores
        or workload.k < 4096
        or workload.k % 128
    ):
        return False
    al1_bytes = align_up(workload.m, 16) * workload.k * 4
    return (
        al1_bytes + 16 * 256 * 4 * 2
        <= effective_l1_bytes(hardware)
    )


def al1_candidate_space(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    if not al1_full_load_applicable(workload, hardware):
        return []
    candidates: list[dict[str, int]] = []
    seed_k = (
        seed.bank.knowledge
        if template_name(seed.bank.knowledge) == "AL1_FULL_LOAD"
        else None
    )
    if seed_k is not None:
        candidates.append(dict(seed_k))
    base_n_values = {16, 32, 64}
    base_k_values = {64, 128, 256}
    if seed_k is not None:
        base_n_values.add(seed_k["baseN"])
        base_k_values.add(seed_k["baseK"])
    max_parts = min(
        hardware.aic_cores,
        workload.max_cores,
        ceil_div(workload.n, 16),
    )
    for base_n in sorted(base_n_values):
        for base_k in sorted(base_k_values):
            for n_parts in range(1, max_parts + 1):
                single_n = max(
                    base_n,
                    align_up(ceil_div(workload.n, n_parts), 16),
                )
                n_chunks = ceil_div(workload.n, single_n)
                for depth_b in (1, 2):
                    for db_c in (1, 2):
                        step_ka = ceil_div(workload.k, base_k)
                        candidate = {
                            "usedCoreNum": min(
                                hardware.aic_cores,
                                workload.max_cores,
                                n_chunks,
                            ),
                            "singleCoreM": workload.m,
                            "singleCoreN": single_n,
                            "singleCoreK": workload.k,
                            "baseM": 16,
                            "baseN": base_n,
                            "baseK": base_k,
                            "depthA1": step_ka,
                            "depthB1": depth_b,
                            "stepM": 1,
                            "stepN": 1,
                            "iterateOrder": (
                                seed_k["iterateOrder"] if seed_k else 0
                            ),
                            "stepKa": step_ka,
                            "stepKb": 1,
                            "dbL0A": 2,
                            "dbL0B": 2,
                            "dbL0C": db_c,
                            "l2MTileCnt": 1,
                            "l2NTileCnt": 1,
                            "l2MTileBlock": 1,
                            "l2NTileBlock": n_chunks,
                            "l2IterateOrder": 1,
                            "tilingEnable": 10,
                        }
                        if hard_legal(workload, candidate, hardware):
                            candidates.append(candidate)
    return deduplicate_knowledge(candidates)


def bl1_fixpipe_bound(workload: Workload) -> bool:
    output_bytes = OUTPUT_BYTES[workload.dtype]
    align_elements = 32 // output_bytes
    return (
        workload.n < 256
        and workload.n % align_elements != 0
        and align_elements % workload.n != 0
    )


def bl1_full_load_applicable(
    workload: Workload,
    hardware: Hardware,
) -> bool:
    if (
        workload.m <= 16 * max(workload.k, workload.n)
        or workload.k > 256
    ):
        return False
    in_bytes = INPUT_BYTES[workload.dtype]
    c0 = 32 // in_bytes
    resident_b = (
        align_up(workload.k, 16)
        * align_up(workload.n, c0)
        * in_bytes
    )
    return resident_b < effective_l1_bytes(hardware)


def bl1_fix_modes(workload: Workload) -> list[int]:
    if not bl1_fixpipe_bound(workload):
        return [0]
    modes = [1]
    c0 = 32 // INPUT_BYTES[workload.dtype]
    if (
        workload.dtype == "fp32"
        and not workload.trans_a
        and workload.n <= 192
        and workload.k % c0 == 0
    ):
        modes.append(2)
    return modes


def bl1_candidate_space(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    if not bl1_full_load_applicable(workload, hardware):
        return []
    candidates: list[dict[str, int]] = []
    seed_k = (
        seed.bank.knowledge
        if template_name(seed.bank.knowledge).startswith("BL1_FULL_LOAD")
        else None
    )
    if seed_k is not None:
        candidates.append(dict(seed_k))

    in_bytes = INPUT_BYTES[workload.dtype]
    c0 = 32 // in_bytes
    n_alignment = 16 if workload.trans_b else c0
    base_m_values = {16, 32, 64, 128, 256}
    base_n_values = {
        align_up(workload.n, n_alignment),
        *(
            value
            for value in (16, 32, 64, 128, 192, 256)
            if value >= workload.n
        ),
    }
    base_k_values = {
        align_up(workload.k, base_k_alignment(workload)),
        32,
        64,
        128,
        256,
    }
    if seed_k is not None:
        base_m_values.add(seed_k["baseM"])
        base_n_values.add(seed_k["baseN"])
        base_k_values.add(seed_k["baseK"])

    core_limit = min(hardware.aic_cores, workload.max_cores)
    m_parts_values = {
        1, 2, 4, 5, 10, core_limit,
        2 * core_limit, 4 * core_limit, 8 * core_limit,
    }
    for base_m in sorted(base_m_values):
        for base_n in sorted(base_n_values):
            for base_k in sorted(base_k_values):
                if base_k % base_k_alignment(workload):
                    continue
                step_n = ceil_div(workload.n, base_n)
                step_k = ceil_div(workload.k, base_k)
                single_n = max(workload.n, base_n)
                for m_parts in sorted(m_parts_values):
                    partition_m = align_up(
                        ceil_div(workload.m, max(1, m_parts)), 16
                    )
                    for single_m in {
                        base_m,
                        2 * base_m,
                        4 * base_m,
                        8 * base_m,
                        max(base_m, partition_m),
                    }:
                        m_chunks = ceil_div(workload.m, single_m)
                        n_chunks = ceil_div(workload.n, single_n)
                        for a1_buffers in (1, 2):
                            for fix in bl1_fix_modes(workload):
                                for db_c in (1, 2):
                                    if fix:
                                        l2_m_block = 0
                                        l2_n_block = 0
                                        l2_order = 0
                                    else:
                                        l2_m_block = m_chunks
                                        l2_n_block = n_chunks
                                        l2_order = 1
                                    candidate = {
                                        "usedCoreNum": min(
                                            core_limit,
                                            m_chunks * n_chunks,
                                        ),
                                        "singleCoreM": single_m,
                                        "singleCoreN": single_n,
                                        "singleCoreK": workload.k,
                                        "baseM": base_m,
                                        "baseN": base_n,
                                        "baseK": base_k,
                                        "depthA1": (
                                            a1_buffers * step_k
                                        ),
                                        "depthB1": step_n * step_k,
                                        "stepM": 1,
                                        "stepN": step_n,
                                        "iterateOrder": (
                                            seed_k["iterateOrder"]
                                            if seed_k else 0
                                        ),
                                        "stepKa": step_k,
                                        "stepKb": step_k,
                                        "dbL0A": 2,
                                        "dbL0B": 2,
                                        "dbL0C": db_c,
                                        "l2MTileCnt": 1,
                                        "l2NTileCnt": 1,
                                        "l2MTileBlock": l2_m_block,
                                        "l2NTileBlock": l2_n_block,
                                        "l2IterateOrder": l2_order,
                                        "tilingEnable": 20 + fix * 1000,
                                    }
                                    if hard_legal(
                                        workload, candidate, hardware
                                    ):
                                        candidates.append(candidate)
    return deduplicate_knowledge(candidates)


def single_core_split_k_applicable(
    workload: Workload,
    seed: Seed,
) -> bool:
    return (
        template_name(seed.bank.knowledge) == "SINGLE_CORE_SPLIT_K"
        or (
            workload.k >= 32768
            and workload.m >= 128
            and workload.n >= 128
        )
    )


def single_core_split_k_candidate_space(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    if not single_core_split_k_applicable(workload, seed):
        return []
    candidates: list[dict[str, int]] = []
    seed_k = (
        seed.bank.knowledge
        if template_name(seed.bank.knowledge) == "SINGLE_CORE_SPLIT_K"
        else None
    )
    if seed_k is not None:
        candidates.append(dict(seed_k))

    base_k = 256 // INPUT_BYTES[workload.dtype]
    algorithms = (
        # name, stepM, stepN, stepK, depthA, depthB, iterate, L2 order
        ("MK33", 3, 1, 3, 9, 6, 1, 0),
        ("NK33", 1, 3, 3, 6, 9, 0, 1),
        ("MK24", 2, 1, 4, 8, 8, 1, 0),
        ("MK14", 1, 1, 4, 8, 8, 1, 0),
    )
    align_m = max(16, 64 // INPUT_BYTES[workload.dtype])
    align_n = max(16, 256 // INPUT_BYTES[workload.dtype])
    pairs = partition_pairs(workload, hardware, max_rounds=4)
    for _, step_m, step_n, step_k, depth_a, depth_b, order, l2_order in algorithms:
        inner_m = step_m * 128
        inner_n = step_n * 128
        single_k = step_k * base_k
        if single_k >= workload.k:
            continue
        for m_parts, n_parts in pairs:
            single_m = max(
                inner_m,
                align_up(ceil_div(workload.m, m_parts), align_m),
            )
            single_n = max(
                inner_n,
                align_up(ceil_div(workload.n, n_parts), align_n),
            )
            m_chunks = ceil_div(workload.m, single_m)
            n_chunks = ceil_div(workload.n, single_n)
            if seed_k is not None:
                l2_m_count = seed_k["l2MTileCnt"]
                l2_n_count = seed_k["l2NTileCnt"]
                l2_m_block = seed_k["l2MTileBlock"]
                l2_n_block = seed_k["l2NTileBlock"]
            else:
                l2_m_count = 1
                l2_n_count = 1
                l2_m_block = max(1, m_chunks)
                l2_n_block = max(1, n_chunks)
            candidate = {
                "usedCoreNum": min(
                    hardware.aic_cores,
                    workload.max_cores,
                    m_chunks * n_chunks,
                ),
                "singleCoreM": single_m,
                "singleCoreN": single_n,
                "singleCoreK": single_k,
                "baseM": 128,
                "baseN": 128,
                "baseK": base_k,
                "depthA1": depth_a,
                "depthB1": depth_b,
                "stepM": step_m,
                "stepN": step_n,
                "iterateOrder": order,
                "stepKa": step_k,
                "stepKb": step_k,
                "dbL0A": 2,
                "dbL0B": 2,
                "dbL0C": 2,
                "l2MTileCnt": l2_m_count,
                "l2NTileCnt": l2_n_count,
                "l2MTileBlock": l2_m_block,
                "l2NTileBlock": l2_n_block,
                "l2IterateOrder": l2_order,
                "tilingEnable": 2,
            }
            if hard_legal(workload, candidate, hardware):
                candidates.append(candidate)
    return deduplicate_knowledge(candidates)


def deterministic_split_k_applicable(
    workload: Workload,
    seed: Seed,
) -> bool:
    return deterministic_split_k_supported_range(workload)


def deterministic_split_k_supported_range(
    workload: Workload,
) -> bool:
    """Range supported by measured deterministic Split-K holdouts.

    K=16384, 32768, and 49152 improved; K=65536 regressed. The rule is
    therefore deliberately bounded to the measured positive interval.
    """
    return (
        workload.dtype == "fp16"
        and not workload.trans_a
        and not workload.trans_b
        and workload.m == 128
        and workload.n == 128
        and workload.k % 128 == 0
        and 16384 <= workload.k <= 49152
    )


def deterministic_split_k_candidate_space(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[dict[str, int]]:
    if not deterministic_split_k_applicable(workload, seed):
        return []
    candidates: list[dict[str, int]] = []
    seed_k = (
        seed.bank.knowledge
        if template_name(seed.bank.knowledge) == "DETERMINISTIC_SPLIT_K"
        else None
    )
    if seed_k is not None:
        candidates.append(dict(seed_k))
    base_k = 256 // INPUT_BYTES[workload.dtype]
    single_k = 3 * base_k
    k_chunks = ceil_div(workload.k, single_k)
    core_limit = min(
        hardware.aic_cores, workload.max_cores, k_chunks
    )
    core_values = {
        1, 2, 4, 5, 8, 10, 16, core_limit,
        max(1, core_limit - 1),
    }
    layouts = (
        (3, 1, 9, 6, 1, 384, max(128, workload.n), 0),
        (1, 3, 6, 9, 0, max(128, workload.m), 384, 1),
    )
    for step_m, step_n, depth_a, depth_b, order, single_m, single_n, l2_order in layouts:
        for cores in sorted(value for value in core_values if value <= core_limit):
            if seed_k is not None:
                l2_m_count = seed_k["l2MTileCnt"]
                l2_n_count = seed_k["l2NTileCnt"]
                l2_m_block = seed_k["l2MTileBlock"]
                l2_n_block = seed_k["l2NTileBlock"]
            else:
                l2_m_count = 1
                l2_n_count = 1
                l2_m_block = max(1, ceil_div(workload.m, single_m))
                l2_n_block = max(1, ceil_div(workload.n, single_n))
            candidate = {
                "usedCoreNum": cores,
                "singleCoreM": single_m,
                "singleCoreN": single_n,
                "singleCoreK": single_k,
                "baseM": 128,
                "baseN": 128,
                "baseK": base_k,
                "depthA1": depth_a,
                "depthB1": depth_b,
                "stepM": step_m,
                "stepN": step_n,
                "iterateOrder": order,
                "stepKa": 3,
                "stepKb": 3,
                "dbL0A": 2,
                "dbL0B": 2,
                "dbL0C": 2,
                "l2MTileCnt": l2_m_count,
                "l2NTileCnt": l2_n_count,
                "l2MTileBlock": l2_m_block,
                "l2NTileBlock": l2_n_block,
                "l2IterateOrder": l2_order,
                "tilingEnable": 3,
            }
            if hard_legal(workload, candidate, hardware):
                candidates.append(candidate)
    return deduplicate_knowledge(candidates)


def all_template_candidate_spaces(
    workload: Workload,
    seed: Seed,
    raw_rows: list[dict[str, str]],
    hardware: Hardware,
    beam_width: int,
    optimization_scope: str,
) -> dict[str, list[dict[str, int]]]:
    if optimization_scope == "skinny_n_large_k_v1":
        focused = skinny_n_large_k_candidate_space(
            workload, seed, hardware
        )
        return {"BASE": focused} if focused else {}
    if optimization_scope == "official_local_v2":
        candidates = official_seed_order_candidate_space(
            workload, seed, hardware
        )
        candidates.extend(
            skinny_n_large_k_candidate_space(
                workload, seed, hardware
            )
        )
        grouped: dict[str, list[dict[str, int]]] = {}
        for candidate in deduplicate_knowledge(candidates):
            grouped.setdefault(template_name(candidate), []).append(
                candidate
            )
        return grouped

    spaces = {
        "BASE": base_candidate_space(
            workload, seed, raw_rows, hardware, beam_width
        ),
        "SINGLE_CORE_SPLIT_K": single_core_split_k_candidate_space(
            workload, seed, hardware
        ),
        "DETERMINISTIC_SPLIT_K": deterministic_split_k_candidate_space(
            workload, seed, hardware
        ),
        "AL1_FULL_LOAD": al1_candidate_space(
            workload, seed, hardware
        ),
    }
    for candidate in bl1_candidate_space(workload, seed, hardware):
        spaces.setdefault(template_name(candidate), []).append(candidate)
    return {
        template: deduplicate_knowledge(candidates)
        for template, candidates in spaces.items()
        if candidates
    }


def chunk_ceil_sum(total: int, chunk: int, unit: int) -> int:
    count = ceil_div(total, chunk)
    tail = total - (count - 1) * chunk
    return (count - 1) * ceil_div(chunk, unit) + ceil_div(tail, unit)


def chunk_aligned_sum(total: int, chunk: int, alignment: int) -> int:
    return chunk_ceil_sum(total, chunk, alignment) * alignment


def fixpipe_row_sum(
    total: int,
    single: int,
    base: int,
    out_bytes: int,
) -> int:
    def extent_bytes(extent: int) -> int:
        subtiles = ceil_div(extent, base)
        tail = extent - (subtiles - 1) * base
        return (
            (subtiles - 1) * align_up(base * out_bytes, 512)
            + align_up(tail * out_bytes, 512)
        )

    count = ceil_div(total, single)
    tail = total - (count - 1) * single
    return (count - 1) * extent_bytes(single) + extent_bytes(tail)


def analytical_score(
    workload: Workload,
    knowledge: dict[str, int],
    hardware: Hardware,
    callback: CallbackTiling | None = None,
) -> ModelEstimate:
    """Estimate the critical-core MatMulV3 pipeline, not wall-clock time.

    The objective follows the structure used by NPUMeter: model the execution
    sequence of every logical tile, overlap MTE2/MTE1/Cube/FixPipe according to
    buffer depth, and take the slowest core. L2/HBM traffic and ND->NZ vector
    conversion are then added at their shared-resource rates. Absolute cycles
    are intentionally not converted to milliseconds; only ratios between
    candidates of the same workload are used for search.
    """
    in_bytes = INPUT_BYTES[workload.dtype]
    out_bytes = OUTPUT_BYTES[workload.dtype]
    cores = max(1, knowledge["usedCoreNum"])
    base_m = knowledge["baseM"]
    base_n = knowledge["baseN"]
    base_k = knowledge["baseK"]
    single_m = knowledge["singleCoreM"]
    single_n = knowledge["singleCoreN"]
    single_k = knowledge["singleCoreK"]
    native_k = cube_k0(workload.dtype)
    split = split_mode(knowledge)
    full = full_load_mode(knowledge)
    fix = fix_mode(knowledge)

    m_output = ceil_div(workload.m, single_m)
    n_output = ceil_div(workload.n, single_n)
    k_passes = (
        ceil_div(workload.k, single_k)
        if split in (2, 3)
        else 1
    )
    logical_tiles = (
        m_output * n_output * k_passes
        if split == 3
        else m_output * n_output
    )

    # Closed-form axis sums avoid materializing millions of tile coordinates
    # for large shapes. They retain full/tail padding, transaction counts and
    # the number of whole AIC scheduling rounds.
    m_align_sum = chunk_aligned_sum(workload.m, single_m, 16)
    n_align_sum = chunk_aligned_sum(workload.n, single_n, 16)
    k_align_sum = chunk_aligned_sum(workload.k, single_k, native_k)
    m_cube_units = chunk_ceil_sum(workload.m, single_m, 16)
    n_cube_units = chunk_ceil_sum(workload.n, single_n, 16)
    k_cube_units = chunk_ceil_sum(workload.k, single_k, native_k)
    m_subtiles = chunk_ceil_sum(workload.m, single_m, base_m)
    n_subtiles = chunk_ceil_sum(workload.n, single_n, base_n)
    k_subtiles = chunk_ceil_sum(workload.k, single_k, base_k)

    total_cube = float(
        m_cube_units * n_cube_units * k_cube_units
        + 12 * m_subtiles * n_subtiles * k_passes
    )

    # stepM/stepN are the L1 reuse window. Count L2->L1 transfers by the
    # number of such windows, not merely by output-core tiles. This is
    # This also covers specialized families where singleCoreM/N can span
    # multiple base tiles.
    m_l1_groups = chunk_ceil_sum(
        workload.m, single_m, base_m * knowledge["stepM"]
    )
    n_l1_groups = chunk_ceil_sum(
        workload.n, single_n, base_n * knowledge["stepN"]
    )
    l2_a_bytes = float(
        m_align_sum * k_align_sum * n_l1_groups * in_bytes
    )
    l2_b_bytes = float(
        n_align_sum * k_align_sum * m_l1_groups * in_bytes
    )
    l2_c_bytes = float(workload.m * workload.n * out_bytes)

    a_k_packets = chunk_ceil_sum(
        workload.k,
        single_k,
        base_k * knowledge["stepKa"],
    )
    b_k_packets = chunk_ceil_sum(
        workload.k,
        single_k,
        base_k * knowledge["stepKb"],
    )
    total_a_packets = m_subtiles * n_l1_groups * a_k_packets
    total_b_packets = n_subtiles * m_l1_groups * b_k_packets

    # The full-load kernels copy one complete operand into each active AIC's
    # L1 before processing all rounds assigned to that core.
    output_tasks = m_output * n_output
    output_active_cores = max(1, min(cores, output_tasks))
    aligned_a_bytes = (
        align_up(workload.m, 16)
        * align_up(workload.k, native_k)
        * in_bytes
    )
    aligned_b_bytes = (
        align_up(workload.k, native_k)
        * align_up(workload.n, max(16, 32 // in_bytes))
        * in_bytes
    )
    if full == 1:
        l2_a_bytes = float(aligned_a_bytes * output_active_cores)
        l2_b_bytes = float(
            n_align_sum * k_align_sum * m_l1_groups * in_bytes
        )
        total_a_packets = output_active_cores
    elif full == 2:
        l2_a_bytes = float(
            m_align_sum * k_align_sum * n_l1_groups * in_bytes
        )
        l2_b_bytes = float(aligned_b_bytes * output_active_cores)
        total_b_packets = output_active_cores

    total_mte2 = (
        (l2_a_bytes + l2_b_bytes)
        / max(1.0, hardware.l2_bytes_per_cycle_per_core)
        + 12.0 * (total_a_packets + total_b_packets)
    )

    mte1_tiles = m_subtiles * n_subtiles * k_subtiles
    total_mte1 = (
        mte1_tiles
        * (base_m * base_k + base_n * base_k)
        * in_bytes
        / 128.0
        + 4.0 * mte1_tiles
    )
    total_fixpipe = (
        m_align_sum
        * fixpipe_row_sum(
            workload.n, single_n, base_n, out_bytes
        )
        / 64.0
        + 4.0 * m_subtiles * n_subtiles
    )
    fix_workspace = 0
    if full == 2 and fix:
        nz_c_bytes = (
            align_up(workload.m, 16)
            * align_up(workload.n, max(16, 256 // out_bytes))
            * out_bytes
        )
        fix_workspace = 2 * nz_c_bytes
        if fix == 2:
            fix_workspace += aligned_a_bytes
        total_fixpipe += fix_workspace / max(
            1.0, 128.0 * 2.0 * output_active_cores
        )
    total_issue = float(
        2 * mte1_tiles + total_a_packets + total_b_packets
    )

    active_cores = max(1, min(cores, logical_tiles))
    round_balance = (
        ceil_div(logical_tiles, active_cores)
        / max(1.0, logical_tiles / active_cores)
    )
    l2_efficiency, _ = l2_schedule_metrics(
        workload, knowledge, hardware
    )
    # The BASE kernel synchronizes scheduling at every L2 tile boundary.
    # Global task balance alone therefore underestimates idle AIC slots when a
    # legal L2 tail contains fewer than one complete core round.
    round_balance = max(
        round_balance,
        1.0 / max(1.0 / active_cores, l2_efficiency),
    )

    def critical(total: float) -> float:
        return total / active_cores * round_balance

    critical_cube = critical(total_cube)
    critical_mte2 = critical(total_mte2)
    critical_mte1 = critical(total_mte1)
    critical_fixpipe = critical(total_fixpipe)
    issue_cycles = critical(total_issue)
    input_stage = max(critical_mte2, critical_mte1)
    if knowledge["dbL0A"] == 2 and knowledge["dbL0B"] == 2:
        critical_pipeline = max(critical_cube, input_stage)
        critical_pipeline += 0.08 * min(critical_cube, input_stage)
    else:
        critical_pipeline = critical_cube + input_stage
    if knowledge["dbL0C"] == 2:
        critical_pipeline = max(critical_pipeline, critical_fixpipe)
    else:
        critical_pipeline += critical_fixpipe

    hbm_a = workload.m * workload.k * in_bytes
    hbm_b = workload.k * workload.n * in_bytes
    if full == 0:
        hbm_a *= knowledge["l2NTileCnt"]
        hbm_b *= knowledge["l2MTileCnt"]
    hbm_c = workload.m * workload.n * out_bytes
    hbm_bytes = float(hbm_a + hbm_b + hbm_c)

    l2_bytes = float(l2_a_bytes + l2_b_bytes + l2_c_bytes)

    if split == 2:
        # Each output-owning AIC executes K slices sequentially. The first
        # slice writes FP32 and the following IterateAll calls atomically
        # accumulate into the same workspace; AIV then casts once to C.
        workspace_one = (
            align_up(workload.m, 16)
            * align_up(workload.n, max(16, 256 // out_bytes))
            * 4
        )
        workspace_traffic = workspace_one * max(1, 2 * k_passes - 1)
        cast_traffic = workspace_one + workload.m * workload.n * out_bytes
        hbm_bytes += workspace_one
        l2_bytes += workspace_traffic + cast_traffic
        issue_cycles += (
            (k_passes - 1) * m_output * n_output * 12.0
            + workload.m * workload.n / 64.0
        ) / output_active_cores
    elif split == 3:
        # Deterministic Split-K writes one FP32 partial per active K core and
        # reduces those partials in UB before the final cast/store.
        split_active = max(1, min(cores, k_passes))
        workspace_one = (
            align_up(workload.m, 16)
            * align_up(workload.n, 16)
            * 4
        )
        workspace_bytes = workspace_one * split_active
        reduction_bytes = (
            workspace_bytes
            + workspace_one
            + workload.m * workload.n * out_bytes
        )
        hbm_bytes += workspace_bytes
        l2_bytes += reduction_bytes
        issue_cycles += (
            split_active * 32.0
            + workload.m * workload.n * split_active / 64.0
        ) / cores

    if full == 2 and fix:
        # Optimized Fixpipe first writes NZ output to workspace, then the AIV
        # path converts/copies it to the user ND tensor. Vec NZ2ND also stages
        # A in workspace before Cube consumes it.
        hbm_bytes += fix_workspace
        l2_bytes += 2 * fix_workspace
        issue_cycles += output_tasks * (24.0 if fix == 2 else 16.0) / cores

    hbm_cycles = hbm_bytes / max(
        1.0, hardware.hbm_bytes_per_cycle_per_core * active_cores
    )
    l2_cycles = l2_bytes / max(
        1.0, hardware.l2_bytes_per_cycle_per_core * active_cores
    )

    m_extent = min(
        workload.m,
        max(1, knowledge["l2MTileBlock"]) * single_m,
    )
    n_extent = min(
        workload.n,
        max(1, knowledge["l2NTileBlock"]) * single_n,
    )
    working_set = (
        m_extent * workload.k * in_bytes
        + workload.k * n_extent * in_bytes
        + m_extent * n_extent * out_bytes
    )
    safe_l2 = hardware.l2_bytes * (100.0 / 192.0)
    capacity_penalty = max(0.0, working_set / max(1.0, safe_l2) - 1.0)
    m_tail = (
        m_output
        - (knowledge["l2MTileCnt"] - 1)
        * max(1, knowledge["l2MTileBlock"])
    )
    n_tail = (
        n_output
        - (knowledge["l2NTileCnt"] - 1)
        * max(1, knowledge["l2NTileBlock"])
    )
    tail_parallelism = max(
        1,
        min(cores, max(1, m_tail) * max(1, n_tail)),
    )
    tail_smear = 1.0 + 0.20 * (1.0 - tail_parallelism / cores)
    l2_shared_cycles = (hbm_cycles + l2_cycles) * tail_smear
    l2_shared_cycles *= 1.0 + 2.0 * capacity_penalty

    nd2nz_cycles = 0.0
    confidence = "medium"
    if callback is not None and all(
        name in callback.derived
        for name in ("baseAN", "baseAD", "baseBN", "baseBD")
    ):
        # Zero means that the corresponding ND->NZ conversion side/axis does
        # not require an explicit split. The field names use logical MatMul
        # axes (A: M,K and B: N,K), independent of physical transpose.
        base_an = callback.derived["baseAN"] or workload.m
        base_ad = callback.derived["baseAD"] or workload.k
        base_bn = callback.derived["baseBN"] or workload.n
        base_bd = callback.derived["baseBD"] or workload.k
        convert_a = bool(
            callback.derived["baseAN"] or callback.derived["baseAD"]
        )
        convert_b = bool(
            callback.derived["baseBN"] or callback.derived["baseBD"]
        )
        a_vector_tiles = (
            ceil_div(workload.m, base_an)
            * ceil_div(workload.k, base_ad)
            if convert_a
            else 0
        )
        b_vector_tiles = (
            ceil_div(workload.n, base_bn)
            * ceil_div(workload.k, base_bd)
            if convert_b
            else 0
        )
        vector_cores = max(1, 2 * cores)
        nd2nz_bytes = (
            (workload.m * workload.k if convert_a else 0)
            + (workload.k * workload.n if convert_b else 0)
        ) * in_bytes
        nd2nz_cycles = (
            nd2nz_bytes / (128.0 * vector_cores)
            + 6.0 * (a_vector_tiles + b_vector_tiles) / vector_cores
        )
        confidence = "high"

    balance = round_balance * cores / active_cores
    score = (
        max(critical_pipeline, l2_shared_cycles)
        + nd2nz_cycles
        + issue_cycles
    )
    return ModelEstimate(
        cycles=score,
        hbm_bytes=hbm_bytes,
        l2_bytes=l2_bytes,
        cube_cycles=critical_cube,
        mte2_cycles=max(critical_mte2, l2_shared_cycles),
        mte1_cycles=critical_mte1,
        fixpipe_cycles=critical_fixpipe,
        nd2nz_cycles=nd2nz_cycles,
        issue_cycles=issue_cycles,
        balance=balance,
        confidence=confidence,
    )


def l2_schedule_metrics(
    workload: Workload,
    knowledge: dict[str, int],
    hardware: Hardware,
) -> tuple[float, float]:
    """Return per-L2-tile AIC efficiency and safe-L2 working-set ratio."""
    cores = max(1, knowledge["usedCoreNum"])
    m_total = ceil_div(workload.m, knowledge["singleCoreM"])
    n_total = ceil_div(workload.n, knowledge["singleCoreN"])
    m_block = knowledge["l2MTileBlock"]
    n_block = knowledge["l2NTileBlock"]

    if m_block <= 0 or n_block <= 0:
        tasks = max(1, m_total * n_total)
        efficiency = tasks / (ceil_div(tasks, cores) * cores)
        return efficiency, 0.0

    def segments(total: int, count: int, block: int) -> list[tuple[int, int]]:
        tail = total - (count - 1) * block
        result: list[tuple[int, int]] = []
        if count > 1:
            result.append((block, count - 1))
        result.append((tail, 1))
        return result

    scheduled_slots = 0
    useful_tasks = 0
    for m_size, m_count in segments(
        m_total, knowledge["l2MTileCnt"], m_block
    ):
        for n_size, n_count in segments(
            n_total, knowledge["l2NTileCnt"], n_block
        ):
            occurrences = m_count * n_count
            tasks = max(1, m_size * n_size)
            useful_tasks += occurrences * tasks
            scheduled_slots += occurrences * ceil_div(tasks, cores) * cores
    efficiency = useful_tasks / max(1, scheduled_slots)

    m_extent = min(workload.m, m_block * knowledge["singleCoreM"])
    n_extent = min(workload.n, n_block * knowledge["singleCoreN"])
    in_bytes = INPUT_BYTES[workload.dtype]
    out_bytes = OUTPUT_BYTES[workload.dtype]
    working_set = (
        m_extent * workload.k * in_bytes
        + workload.k * n_extent * in_bytes
        + m_extent * n_extent * out_bytes
    )
    safe_l2 = hardware.l2_bytes * (100.0 / 192.0)
    return efficiency, working_set / max(1.0, safe_l2)


def diagnose_bottleneck(
    workload: Workload,
    knowledge: dict[str, int],
    estimate: ModelEstimate,
    hardware: Hardware,
) -> BottleneckProfile:
    cores = max(1, knowledge["usedCoreNum"])
    output_tasks = (
        ceil_div(workload.m, knowledge["singleCoreM"])
        * ceil_div(workload.n, knowledge["singleCoreN"])
    )
    logical_tasks = output_tasks
    if split_mode(knowledge) == 3:
        logical_tasks *= ceil_div(workload.k, knowledge["singleCoreK"])
    active_cores = min(cores, max(1, logical_tasks))
    core_utilization = active_cores / cores
    round_efficiency = logical_tasks / (
        ceil_div(logical_tasks, active_cores) * active_cores
    )
    l2_efficiency, l2_working_set_ratio = l2_schedule_metrics(
        workload, knowledge, hardware
    )
    cube = max(1.0, estimate.cube_cycles)
    input_stage = max(estimate.mte2_cycles, estimate.mte1_cycles)
    input_to_cube = input_stage / cube
    fixpipe_to_cube = estimate.fixpipe_cycles / cube

    operations = 2 * workload.m * workload.n * workload.k
    if (
        operations >= 16 * 1024 * 1024
        and (core_utilization < 0.9 or round_efficiency < 0.85)
    ):
        dominant = "core_grid"
    elif l2_working_set_ratio > 1.05:
        dominant = "l2_capacity"
    elif l2_efficiency < 0.85:
        dominant = "l2_tail"
    elif input_stage >= max(
        estimate.cube_cycles, estimate.fixpipe_cycles
    ):
        dominant = (
            "mte2"
            if estimate.mte2_cycles >= estimate.mte1_cycles
            else "mte1"
        )
    elif estimate.fixpipe_cycles >= estimate.cube_cycles:
        dominant = "fixpipe"
    else:
        dominant = "cube"
    return BottleneckProfile(
        dominant=dominant,
        output_tasks=output_tasks,
        active_cores=active_cores,
        core_utilization=core_utilization,
        round_efficiency=round_efficiency,
        l2_efficiency=l2_efficiency,
        l2_working_set_ratio=l2_working_set_ratio,
        input_to_cube=input_to_cube,
        fixpipe_to_cube=fixpipe_to_cube,
        confidence=estimate.confidence,
    )


def configure_base_candidate(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    base_m: int,
    base_n: int,
    base_k: int,
) -> dict[str, int] | None:
    candidate = dict(seed.bank.knowledge)
    m_parts = ceil_div(workload.m, base_m)
    n_parts = ceil_div(workload.n, base_n)
    candidate.update(
        {
            "usedCoreNum": min(
                workload.max_cores,
                hardware.aic_cores,
                m_parts * n_parts,
            ),
            "singleCoreM": base_m,
            "singleCoreN": base_n,
            "singleCoreK": workload.k,
            "baseM": base_m,
            "baseN": base_n,
            "baseK": base_k,
            "stepM": 1,
            "stepN": 1,
            "iterateOrder": seed.bank.knowledge["iterateOrder"],
            "l2MTileCnt": 1,
            "l2NTileCnt": 1,
            "l2MTileBlock": m_parts,
            "l2NTileBlock": n_parts,
            "l2IterateOrder": 0,
            "tilingEnable": 0,
        }
    )
    depth_a, depth_b, step_a, step_b = official_l1_for_base(
        workload, candidate, hardware
    )
    candidate.update(
        {
            "depthA1": depth_a,
            "depthB1": depth_b,
            "stepKa": step_a,
            "stepKb": step_b,
        }
    )
    return candidate if hard_legal(workload, candidate, hardware) else None


def core_grid_transition_proposal(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    profile: BottleneckProfile,
) -> CandidateProposal | None:
    """Construct one output-grid transition at an AIC round boundary."""
    if (
        template_name(seed.bank.knowledge) != "BASE"
        or profile.dominant != "core_grid"
        or skinny_n_large_k_applicable(workload, seed, hardware)
        or skinny_n_low_k_causal_applicable(
            workload, seed, hardware
        )
    ):
        return None
    cores = min(workload.max_cores, hardware.aic_cores)
    seed_k = seed.bank.knowledge
    seed_tasks = profile.output_tasks
    seed_efficiency = profile.round_efficiency
    seed_padding = (
        ceil_div(workload.m, seed_k["baseM"])
        * seed_k["baseM"]
        * ceil_div(workload.n, seed_k["baseN"])
        * seed_k["baseN"]
        / max(1.0, workload.m * workload.n)
    )

    proposals: list[tuple[float, dict[str, int], float]] = []
    target_rounds = {
        1,
        max(1, ceil_div(seed_tasks, cores)),
    }
    for rounds in target_rounds:
        target_tasks = rounds * cores
        for m_parts in range(1, target_tasks + 1):
            n_parts = ceil_div(target_tasks, m_parts)
            base_m = align_up(ceil_div(workload.m, m_parts), 16)
            base_n = align_up(ceil_div(workload.n, n_parts), 16)
            candidate = configure_base_candidate(
                workload,
                seed,
                hardware,
                base_m,
                base_n,
                seed_k["baseK"],
            )
            if candidate is None:
                continue
            tasks = (
                ceil_div(workload.m, base_m)
                * ceil_div(workload.n, base_n)
            )
            active = min(cores, tasks)
            efficiency = tasks / (ceil_div(tasks, active) * active)
            utilization = active / cores
            padding = (
                ceil_div(workload.m, base_m)
                * base_m
                * ceil_div(workload.n, base_n)
                * base_n
                / max(1.0, workload.m * workload.n)
            )
            padding_growth = max(0.0, padding / seed_padding - 1.0)
            gain = (
                0.65 * (utilization - profile.core_utilization)
                + 0.35 * (efficiency - seed_efficiency)
                - 0.20 * padding_growth
            )
            if gain > 0.03:
                distance = (
                    abs(math.log2(base_m / seed_k["baseM"]))
                    + abs(math.log2(base_n / seed_k["baseN"]))
                )
                proposals.append((gain - 0.01 * distance, candidate, gain))
    if not proposals:
        return None
    _, candidate, gain = max(
        proposals,
        key=lambda item: (item[0], -knowledge_signature(item[1])[4]),
    )
    return CandidateProposal(
        knowledge=candidate,
        guidance="bottleneck_core_grid_round_boundary",
        rationale=(
            "construct M/N tiles at the nearest complete AIC scheduling "
            "round while bounding alignment padding"
        ),
        transition_gain=gain,
    )


def shallow_k_l0_frontier_proposals(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    seed_estimate: ModelEstimate,
) -> list[CandidateProposal]:
    """Construct at most two ROLLER-style L0 capacity-frontier tiles."""
    knowledge = seed.bank.knowledge
    if (
        template_name(knowledge) != "BASE"
        or workload.k > 4 * knowledge["baseK"]
        or workload.k <= knowledge["baseK"]
        or 2 * workload.m * workload.n * workload.k < 16 * 1024 * 1024
    ):
        return []
    target_k = min(
        align_up(workload.k, base_k_alignment(workload)),
        2 * knowledge["baseK"],
    )
    if target_k <= knowledge["baseK"]:
        return []

    cores = min(workload.max_cores, hardware.aic_cores)
    candidates: list[tuple[float, dict[str, int]]] = []
    for target_tasks in (cores, 2 * cores):
        for m_parts in range(1, target_tasks + 1):
            n_parts = ceil_div(target_tasks, m_parts)
            base_m = align_up(ceil_div(workload.m, m_parts), 16)
            base_n = align_up(ceil_div(workload.n, n_parts), 16)
            candidate = configure_base_candidate(
                workload, seed, hardware, base_m, base_n, target_k
            )
            if candidate is None:
                continue
            estimate = analytical_score(workload, candidate, hardware)
            gain = 1.0 - estimate.cycles / max(1.0, seed_estimate.cycles)
            if gain > 0.005:
                candidates.append((gain, candidate))

    proposals: list[CandidateProposal] = []
    for orientation, predicate in (
        ("m_reuse", lambda item: item[1]["baseM"] >= item[1]["baseN"]),
        ("n_reuse", lambda item: item[1]["baseM"] < item[1]["baseN"]),
    ):
        eligible = [item for item in candidates if predicate(item)]
        if not eligible:
            continue
        gain, candidate = max(
            eligible,
            key=lambda item: (
                item[0],
                item[1]["baseM"] * item[1]["baseN"],
            ),
        )
        proposals.append(
            CandidateProposal(
                knowledge=candidate,
                guidance=f"bottleneck_l0_k_frontier_{orientation}",
                rationale=(
                    "raise baseK to the next L0 capacity boundary and derive "
                    "M/N from a complete AIC-round factor frontier"
                ),
                transition_gain=gain,
            )
        )
    return proposals


def l1_packet_frontier_proposal(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    profile: BottleneckProfile,
    seed_estimate: ModelEstimate,
) -> CandidateProposal | None:
    """Solve one continuous L1 packet-allocation frontier, then snap legally."""
    knowledge = seed.bank.knowledge
    if (
        template_name(knowledge) != "BASE"
        or profile.input_to_cube < 0.85
        or knowledge["stepM"] != 1
        or knowledge["stepN"] != 1
    ):
        return None
    k_chunks = ceil_div(workload.k, knowledge["baseK"])
    if k_chunks <= 1:
        return None
    m_tiles = ceil_div(workload.m, knowledge["baseM"])
    n_tiles = ceil_div(workload.n, knowledge["baseN"])
    a_weight = max(1, m_tiles * n_tiles)
    b_weight = max(1, m_tiles * n_tiles)
    in_bytes = INPUT_BYTES[workload.dtype]
    a_step_bytes = (
        2 * knowledge["baseM"] * knowledge["baseK"] * in_bytes
    )
    b_step_bytes = (
        2 * knowledge["baseN"] * knowledge["baseK"] * in_bytes
    )
    ratio = math.sqrt(
        a_weight * b_step_bytes / max(1.0, b_weight * a_step_bytes)
    )
    ratios: set[int] = set()
    if ratio >= 1.0:
        ratios.update({max(1, math.floor(ratio)), max(1, math.ceil(ratio))})
    else:
        inverse = 1.0 / ratio
        ratios.update(
            {-max(1, math.floor(inverse)), -max(1, math.ceil(inverse))}
        )

    seed_packet_cost = (
        a_weight * ceil_div(k_chunks, knowledge["stepKa"])
        + b_weight * ceil_div(k_chunks, knowledge["stepKb"])
    )
    candidates: list[tuple[float, dict[str, int], int]] = []
    for snapped_ratio in ratios:
        if snapped_ratio > 0:
            step_b = effective_l1_bytes(hardware) // (
                a_step_bytes * snapped_ratio + b_step_bytes
            )
            step_a = snapped_ratio * step_b
        else:
            multiplier = -snapped_ratio
            step_a = effective_l1_bytes(hardware) // (
                a_step_bytes + b_step_bytes * multiplier
            )
            step_b = multiplier * step_a
        step_a = max(1, min(k_chunks, step_a))
        step_b = max(1, min(k_chunks, step_b))
        if step_a % step_b and step_b % step_a:
            continue
        candidate = dict(knowledge)
        candidate.update(
            {
                "stepKa": step_a,
                "stepKb": step_b,
                "depthA1": 2 * step_a,
                "depthB1": 2 * step_b,
            }
        )
        if not hard_legal(workload, candidate, hardware):
            continue
        packet_cost = (
            a_weight * ceil_div(k_chunks, step_a)
            + b_weight * ceil_div(k_chunks, step_b)
        )
        packet_gain = 1.0 - packet_cost / max(1.0, seed_packet_cost)
        estimate = analytical_score(workload, candidate, hardware)
        model_gain = 1.0 - estimate.cycles / max(
            1.0, seed_estimate.cycles
        )
        if packet_gain >= 0.10 and model_gain > 0.002:
            candidates.append((model_gain, candidate, packet_cost))
    if not candidates:
        return None
    gain, candidate, _ = max(candidates, key=lambda item: item[0])
    favored = (
        "a"
        if candidate["stepKa"] > knowledge["stepKa"]
        else "b"
    )
    return CandidateProposal(
        knowledge=candidate,
        guidance=f"bottleneck_l1_packet_{favored}",
        rationale=(
            "allocate L1 at the continuous packet-cost optimum, then snap to "
            "the MatMul stepKa/stepKb divisibility contract"
        ),
        transition_gain=gain,
    )


def l2_reuse_order_proposal(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> CandidateProposal | None:
    knowledge = seed.bank.knowledge
    if template_name(knowledge) != "BASE":
        return None
    m_tiles = ceil_div(workload.m, knowledge["singleCoreM"])
    n_tiles = ceil_div(workload.n, knowledge["singleCoreN"])
    if m_tiles <= 1 or n_tiles <= 1:
        return None
    a_bytes = workload.m * workload.k * INPUT_BYTES[workload.dtype]
    b_bytes = workload.k * workload.n * INPUT_BYTES[workload.dtype]
    skew = max(a_bytes, b_bytes) / max(1.0, min(a_bytes, b_bytes))
    if skew < 1.25:
        return None
    # MatMulV3 kernel constants are ROW_FIRST=1 and COL_FIRST=2. Row-first
    # keeps A's M block while N advances; column-first keeps B's N block while
    # M advances. Zero is the official staggered-core mapping.
    desired = 1 if a_bytes >= b_bytes else 2
    if knowledge["l2IterateOrder"] == desired:
        return None
    candidate = dict(knowledge)
    candidate["l2IterateOrder"] = desired
    if not hard_legal(workload, candidate, hardware):
        return None
    return CandidateProposal(
        knowledge=candidate,
        guidance=(
            "bottleneck_l2_reuse_a_row_first"
            if desired == 1
            else "bottleneck_l2_reuse_b_col_first"
        ),
        rationale=(
            "select the kernel's explicit row/column traversal that retains "
            "the larger operand across adjacent output blocks"
        ),
        transition_gain=min(0.10, 0.05 * (1.0 - 1.0 / skew)),
    )


def l2_tail_transition_proposal(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    profile: BottleneckProfile,
) -> CandidateProposal | None:
    knowledge = seed.bank.knowledge
    if (
        template_name(knowledge) != "BASE"
        or (
            profile.l2_efficiency >= 0.90
            and profile.l2_working_set_ratio <= 1.05
        )
    ):
        return None
    m_total = ceil_div(workload.m, knowledge["singleCoreM"])
    n_total = ceil_div(workload.n, knowledge["singleCoreN"])
    seed_m_count = knowledge["l2MTileCnt"]
    seed_n_count = knowledge["l2NTileCnt"]
    candidates: list[tuple[float, float, dict[str, int]]] = []
    for m_count in {
        max(1, seed_m_count - 1),
        seed_m_count,
        seed_m_count + 1,
    }:
        for n_count in {
            max(1, seed_n_count - 1),
            seed_n_count,
            seed_n_count + 1,
        }:
            m_block = ceil_div(m_total, m_count)
            n_block = ceil_div(n_total, n_count)
            candidate = dict(knowledge)
            candidate.update(
                {
                    "l2MTileBlock": m_block,
                    "l2NTileBlock": n_block,
                    "l2MTileCnt": ceil_div(m_total, m_block),
                    "l2NTileCnt": ceil_div(n_total, n_block),
                }
            )
            if (
                knowledge_signature(candidate)
                == knowledge_signature(knowledge)
                or not hard_legal(workload, candidate, hardware)
            ):
                continue
            efficiency, working_ratio = l2_schedule_metrics(
                workload, candidate, hardware
            )
            if working_ratio > max(1.05, profile.l2_working_set_ratio):
                continue
            gain = (
                efficiency - profile.l2_efficiency
                + 0.20
                * max(
                    0.0,
                    profile.l2_working_set_ratio - working_ratio,
                )
            )
            if gain > 0.03:
                candidates.append((gain, working_ratio, candidate))
    if not candidates:
        return None
    gain, _, candidate = max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    action = (
        "capacity"
        if profile.l2_working_set_ratio > 1.05
        else "tail"
    )
    return CandidateProposal(
        knowledge=candidate,
        guidance=f"bottleneck_l2_{action}_transition",
        rationale=(
            "move one L2 tile-count boundary to reduce per-tile idle AIC "
            "slots without exceeding the source-derived safe working set"
        ),
        transition_gain=gain,
    )


def db_transition_proposal(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    profile: BottleneckProfile,
    seed_estimate: ModelEstimate,
) -> CandidateProposal | None:
    knowledge = seed.bank.knowledge
    if (
        template_name(knowledge) != "BASE"
        or knowledge["dbL0C"] == 2
        or profile.fixpipe_to_cube < 0.70
    ):
        return None
    candidate = dict(knowledge)
    candidate["dbL0C"] = 2
    if not hard_legal(workload, candidate, hardware):
        return None
    estimate = analytical_score(workload, candidate, hardware)
    gain = 1.0 - estimate.cycles / max(1.0, seed_estimate.cycles)
    if gain <= 0.002:
        return None
    return CandidateProposal(
        knowledge=candidate,
        guidance="bottleneck_fixpipe_l0c_double_buffer",
        rationale=(
            "enable L0C ping-pong only when Fixpipe is competitive with Cube "
            "and the exact L0C capacity contract still holds"
        ),
        transition_gain=gain,
    )


def broad_base_transition_proposals(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    seed_estimate: ModelEstimate,
    profile: BottleneckProfile,
) -> list[CandidateProposal]:
    proposals: list[CandidateProposal] = []
    core = core_grid_transition_proposal(
        workload, seed, hardware, profile
    )
    if core is not None:
        proposals.append(core)
    proposals.extend(
        shallow_k_l0_frontier_proposals(
            workload, seed, hardware, seed_estimate
        )
    )
    for proposal in (
        l1_packet_frontier_proposal(
            workload,
            seed,
            hardware,
            profile,
            seed_estimate,
        ),
        l2_reuse_order_proposal(workload, seed, hardware),
        l2_tail_transition_proposal(
            workload, seed, hardware, profile
        ),
        db_transition_proposal(
            workload,
            seed,
            hardware,
            profile,
            seed_estimate,
        ),
    ):
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def split_template_proposals(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> list[CandidateProposal]:
    knowledge = seed.bank.knowledge
    family = template_name(knowledge)
    if (
        family == "DETERMINISTIC_SPLIT_K"
        and deterministic_split_k_supported_range(workload)
        and workload.m % 128 == 0
        and workload.n % 128 == 0
        and workload.k % 128 == 0
    ):
        candidate = dict(knowledge)
        candidate["iterateOrder"] = 1 - candidate["iterateOrder"]
        if hard_legal(workload, candidate, hardware):
            return [
                CandidateProposal(
                    knowledge=candidate,
                    guidance="official_seed_iterate_order_ablation",
                    rationale=(
                        "switch the deterministic Split-K MK/NK traversal; "
                        "the installed kernel reads iterateOrder directly"
                    ),
                    transition_gain=0.02,
                    resume_policy="require_existing",
                )
            ]
    if (
        family == "SINGLE_CORE_SPLIT_K"
        and knowledge["l2IterateOrder"] == 0
        and knowledge["singleCoreN"]
        > knowledge["stepN"] * knowledge["baseN"]
    ):
        candidate = dict(knowledge)
        candidate["l2IterateOrder"] = 1
        if hard_legal(workload, candidate, hardware):
            return [
                CandidateProposal(
                    knowledge=candidate,
                    guidance="bottleneck_sc_split_k_inner_n",
                    rationale=(
                        "enable the source-defined inner-N loop only when "
                        "singleCoreN contains more than one legal N step"
                    ),
                    transition_gain=0.01,
                )
            ]
    return []


def bottleneck_guided_candidate_proposals(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    seed_estimate: ModelEstimate,
    profile: BottleneckProfile,
) -> tuple[list[CandidateProposal], str]:
    """Build a bounded action frontier instead of a parameter product."""
    family = template_name(seed.bank.knowledge)
    proposals: list[CandidateProposal] = []
    if family == "BASE":
        if skinny_n_low_k_falsified_applicable(
            workload, seed, hardware
        ):
            return [], "low_k_skinny_n_extrapolation_rejected_by_npu_evidence"
        established = learned_skinny_n_k16384_schedule(
            workload, seed, hardware
        )
        boundary = learned_skinny_n_boundary_k16384_schedule(
            workload, seed, hardware
        )
        transition48 = learned_skinny_n_transition48_k16384_schedule(
            workload, seed, hardware
        )
        boundary64 = learned_skinny_n_boundary64_k16384_schedule(
            workload, seed, hardware
        )
        low_k_causal = skinny_n_low_k_causal_space(
            workload, seed, hardware
        )
        boundary_holdout = workload.workload_id.startswith(
            (
                "skinny_n_boundary_holdout_",
                "skinny_n_boundary64_holdout_",
            )
        )
        focused = (
            established
            or boundary
            or transition48
            or boundary64
            or low_k_causal
        )
        for index, knowledge in enumerate(focused):
            if established:
                guidance = focused_candidate_guidance(knowledge)
                rationale = (
                    "apply the K=16384 one-block-per-AIC schedule selected "
                    "by five independent NPU shapes"
                )
            elif boundary or transition48 or boundary64:
                if transition48:
                    guidance = (
                        "skinny_n_transition48_k16384_base_n64_"
                        "one_block_per_aic"
                    )
                    n_range = "48 crossover"
                    l1_split = "8x32"
                elif boundary64:
                    guidance = (
                        "skinny_n_boundary_k16384_base_n64_"
                        "one_block_per_aic"
                    )
                    n_range = "49..64"
                    l1_split = "8x32"
                else:
                    guidance = (
                        "skinny_n_boundary_k16384_one_block_per_aic"
                    )
                    n_range = "33..47"
                    l1_split = "8x40"
                rationale = (
                    f"apply the N={n_range} boundary schedule: one M block "
                    f"per AIC, one N block, and the L1 {l1_split} split"
                )
            else:
                guidance = (
                    "skinny_n_low_k_l2_partition"
                    if index == 0
                    else "skinny_n_low_k_l1_rebalance"
                )
                rationale = (
                    "compare against the measured one-block-per-AIC lower-K "
                    "schedule by changing L2 grouping first, then L1 depth"
                )
            proposals.append(
                CandidateProposal(
                    knowledge=knowledge,
                    guidance=guidance,
                    rationale=rationale,
                    transition_gain=0.10,
                    resume_policy=(
                        "require_existing"
                        if established or (boundary and not boundary_holdout)
                        else "allow_new"
                    ),
                )
            )
        if focused:
            # Keep one hypothesis per workload. Generic transitions would
            # confound the learned K=16384 rule or the lower-K causal test.
            pass
        else:
            pass
        if proposals:
            pass
        elif workload.workload_id.startswith("npu_smoke_"):
            # Smoke mode is a tiny NPU plumbing check. Keep a single small
            # searched path there so terminal output still proves custom
            # tiling injection works, without reopening the full workload
            # broad BASE search that net_log4 falsified.
            proposals.extend(
                broad_base_transition_proposals(
                    workload,
                    seed,
                    hardware,
                    seed_estimate,
                    profile,
                )
            )
        else:
            # The broad BASE transition frontier was measured on 910B3 and
            # did not generalize: full-size LLM/vision/attention shapes were
            # mostly within noise or slower than the official MatMulV3 path.
            # Keep the transition constructors available for unit tests and
            # explicit research scopes, but do not emit them from the default
            # optimizer until a narrower, holdout-backed family is defined.
            return [], "broad_base_transitions_rejected_by_npu_evidence"
    else:
        proposals.extend(split_template_proposals(workload, seed, hardware))

    unique: list[CandidateProposal] = []
    seen: set[tuple[int, ...]] = set()
    for proposal in sorted(
        proposals,
        key=lambda item: (-item.transition_gain, item.guidance),
    ):
        signature = knowledge_signature(proposal.knowledge)
        if signature == knowledge_signature(seed.bank.knowledge) or signature in seen:
            continue
        seen.add(signature)
        unique.append(proposal)
    if unique:
        return unique[:8], ""
    if family in {
        "AL1_FULL_LOAD",
        "BL1_FULL_LOAD",
        "BL1_FULL_LOAD_FIXPIPE",
        "BL1_FULL_LOAD_VEC_NZ2ND",
    }:
        stop = "full_load_template_has_no_source_supported_free_transition"
    elif family == "DETERMINISTIC_SPLIT_K":
        if (
            workload.m % 128 == 0
            and workload.n % 128 == 0
            and workload.k % 128 == 0
        ):
            stop = "deterministic_split_k_range_rejected_by_npu_evidence"
        else:
            stop = "unaligned_split_k_order_is_excluded_by_negative_npu_evidence"
    elif family == "SINGLE_CORE_SPLIT_K":
        stop = "single_core_split_k_has_no_legal_inner_n_transition"
    else:
        stop = "seed_has_no_diagnosed_transition_above_minimum_gain"
    return [], stop


def knowledge_signature(knowledge: dict[str, int]) -> tuple[int, ...]:
    return tuple(knowledge[field] for field in KNOWLEDGE_FIELDS)


def dominated_by_bank_seed(
    knowledge: dict[str, int],
    seed: Seed,
) -> bool:
    if knowledge["usedCoreNum"] >= seed.bank.knowledge["usedCoreNum"]:
        return False
    return all(
        knowledge[field] == seed.bank.knowledge[field]
        for field in KNOWLEDGE_FIELDS
        if field != "usedCoreNum"
    )


def row_from_state(
    fields: list[str],
    prototype: dict[str, str] | None,
    workload: Workload,
    knowledge: dict[str, int],
    source: str,
    template: str,
    score: float,
    hbm_bytes: float,
    l2_bytes: float,
    seed_key: int,
    guidance: str = "analytical_beam",
    candidate_role: str = "searched",
    estimate: ModelEstimate | None = None,
    bottleneck: str = "",
    rationale: str = "",
    transition_gain: float = 0.0,
    resume_policy: str = "require_existing",
    stop_reason: str = "",
) -> dict[str, str]:
    row = {field: "" for field in fields}
    if prototype:
        row.update(prototype)
    row.update(
        {
            "workload_id": workload.workload_id,
            "m": str(workload.m),
            "n": str(workload.n),
            "k": str(workload.k),
            "dtype": workload.dtype,
            "trans_a": str(int(workload.trans_a)),
            "trans_b": str(int(workload.trans_b)),
            "max_cores": str(workload.max_cores),
            "source": source,
            "candidate_role": candidate_role,
            "source_iteration": "0",
            "valid": "1",
            "error": "",
            "execution_mode": (
                "deterministic_split_k"
                if split_mode(knowledge) == 3
                else "single_core_split_k"
                if split_mode(knowledge) == 2
                else "al1_full_load"
                if full_load_mode(knowledge) == 1
                else "bl1_full_load_fixpipe"
                if full_load_mode(knowledge) == 2
                and fix_mode(knowledge) == 1
                else "bl1_full_load_vec_nz2nd"
                if full_load_mode(knowledge) == 2
                and fix_mode(knowledge) == 2
                else "bl1_full_load"
                if full_load_mode(knowledge) == 2
                else "base_iterate_all"
            ),
            "candidate_single_core_m": str(knowledge["singleCoreM"]),
            "candidate_single_core_n": str(knowledge["singleCoreN"]),
            "candidate_single_core_k": str(knowledge["singleCoreK"]),
            "candidate_base_m": str(knowledge["baseM"]),
            "candidate_base_n": str(knowledge["baseN"]),
            "candidate_base_k": str(knowledge["baseK"]),
            "candidate_traverse": str(knowledge["iterateOrder"]),
            "candidate_db_a": str(int(knowledge["dbL0A"] == 2)),
            "candidate_db_b": str(int(knowledge["dbL0B"] == 2)),
            "candidate_split_k": str(int(split_mode(knowledge) != 0)),
            "used_core_num": str(knowledge["usedCoreNum"]),
            "official_core_num": str(knowledge["usedCoreNum"]),
            "official_m_dim": str(ceil_div(workload.m, knowledge["singleCoreM"])),
            "official_n_dim": str(ceil_div(workload.n, knowledge["singleCoreN"])),
            "m_core_parts": str(ceil_div(workload.m, knowledge["singleCoreM"])),
            "n_core_parts": str(ceil_div(workload.n, knowledge["singleCoreN"])),
            "k_core_parts": str(ceil_div(workload.k, knowledge["singleCoreK"])),
            "single_core_m": str(knowledge["singleCoreM"]),
            "single_core_n": str(knowledge["singleCoreN"]),
            "single_core_k": str(knowledge["singleCoreK"]),
            "base_m": str(knowledge["baseM"]),
            "base_n": str(knowledge["baseN"]),
            "base_k": str(knowledge["baseK"]),
            "depth_a1": str(knowledge["depthA1"]),
            "depth_b1": str(knowledge["depthB1"]),
            "step_m": str(knowledge["stepM"]),
            "step_n": str(knowledge["stepN"]),
            "step_ka": str(knowledge["stepKa"]),
            "step_kb": str(knowledge["stepKb"]),
            "iterate_order": str(knowledge["iterateOrder"]),
            "db_l0a": str(knowledge["dbL0A"]),
            "db_l0b": str(knowledge["dbL0B"]),
            "db_l0c": str(knowledge["dbL0C"]),
            "official_return": "0",
            "tiling_signature": ":".join(map(str, knowledge_signature(knowledge))),
            "tiling_bin": "",
            "search_template": template,
            "search_guidance": guidance,
            "search_bottleneck": bottleneck,
            "search_rationale": rationale,
            "search_transition_gain": f"{transition_gain:.12g}",
            "search_resume_policy": resume_policy,
            "search_stop_reason": stop_reason,
            "search_model_score": f"{score:.12g}",
            "search_model_cycles": (
                f"{estimate.cycles:.12g}" if estimate else ""
            ),
            "search_model_raw_ratio_vs_bank_seed": f"{score:.12g}",
            "search_model_ratio_vs_bank_seed": f"{score:.12g}",
            "search_model_calibration": "1",
            "search_model_confidence": estimate.confidence if estimate else "",
            "search_model_breakdown": estimate.breakdown() if estimate else "",
            "search_hbm_bytes": f"{hbm_bytes:.12g}",
            "search_l2_bytes": f"{l2_bytes:.12g}",
            "search_seed_key": str(seed_key),
            "search_history_match": "",
        }
    )
    for knowledge_name, column in BANK_COLUMNS.items():
        row[column] = str(knowledge[knowledge_name])
    return row


def update_callback_columns(
    state: State,
    callback: CallbackTiling,
    seed: Seed,
    bank_seed_estimate: ModelEstimate,
    hardware: Hardware,
    workload: Workload,
) -> None:
    estimate = analytical_score(
        workload, state.knowledge, hardware, callback
    )
    ratio = estimate.cycles / max(1.0, bank_seed_estimate.cycles)
    state.callback = callback
    state.estimate = estimate
    state.model_score = estimate.cycles
    state.normalized_score = ratio
    state.hbm_bytes = estimate.hbm_bytes
    state.l2_bytes = estimate.l2_bytes
    state.row.update(
        {
            "search_model_score": f"{ratio:.12g}",
            "search_model_cycles": f"{estimate.cycles:.12g}",
            "search_model_raw_ratio_vs_bank_seed": f"{ratio:.12g}",
            "search_model_ratio_vs_bank_seed": f"{ratio:.12g}",
            "search_model_calibration": "1",
            "search_model_confidence": estimate.confidence,
            "search_model_breakdown": estimate.breakdown(),
            "search_hbm_bytes": f"{estimate.hbm_bytes:.12g}",
            "search_l2_bytes": f"{estimate.l2_bytes:.12g}",
            "callback_tiling_sha256": callback.sha256,
            "callback_tiling_bytes": str(len(callback.blob)),
            "callback_tiling_key": str(callback.key),
            "callback_block_dim": str(callback.block_dim),
            "callback_workspace_bytes": str(sum(callback.workspaces)),
            "callback_l2_cache_flag": str(
                callback.derived.get("l2CacheFlag", "")
            ),
            "callback_base_an": str(callback.derived.get("baseAN", "")),
            "callback_base_ad": str(callback.derived.get("baseAD", "")),
            "callback_base_bn": str(callback.derived.get("baseBN", "")),
            "callback_base_bd": str(callback.derived.get("baseBD", "")),
            "callback_kernel_suffix": str(kernel_suffix(callback.key)),
            "callback_kernel_variant": kernel_variant(callback.key),
            "callback_kernel_family": kernel_family(callback.key),
            "callback_derived_diff_vs_default": callback_derived_diff(
                seed.default, callback
            ),
            "callback_derived_diff_vs_bank_seed": callback_derived_diff(
                seed.bank, callback
            ),
        }
    )


def validate_callback(
    workload: Workload,
    state: State,
) -> CallbackTiling:
    callback = invoke_official_callback(workload, state.knowledge)
    observed_family = kernel_family(callback.key)
    if observed_family != state.template:
        raise SearchError(
            "official callback selected a different kernel family: "
            f"{observed_family} expected={state.template} "
            f"suffix={kernel_suffix(callback.key)}"
        )
    observed = callback.knowledge
    mismatches = [
        f"{field}={observed[field]} expected={state.knowledge[field]}"
        for field in KNOWLEDGE_FIELDS
        if observed[field] != state.knowledge[field]
    ]
    if mismatches:
        raise SearchError("official callback changed candidate: " + "; ".join(mismatches))
    return callback


BEAM_LAYERS = (
    (
        "tilingEnable", "usedCoreNum", "singleCoreM",
        "singleCoreN", "singleCoreK",
    ),
    ("baseM", "baseN", "baseK"),
    (
        "stepM", "stepN", "stepKa", "stepKb",
        "depthA1", "depthB1", "iterateOrder",
    ),
    ("dbL0A", "dbL0B", "dbL0C"),
    (
        "l2MTileCnt", "l2NTileCnt", "l2MTileBlock",
        "l2NTileBlock", "l2IterateOrder",
    ),
)

NEIGHBOR_GROUPS = (
    (
        "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
    ),
    (
        "baseM", "baseN", "baseK", "stepM", "stepN",
        "stepKa", "stepKb", "depthA1", "depthB1",
        "dbL0A", "dbL0B", "dbL0C",
    ),
    ("iterateOrder",),
    (
        "l2MTileCnt", "l2NTileCnt", "l2MTileBlock",
        "l2NTileBlock", "l2IterateOrder",
    ),
)


def state_sort_key(state: State) -> tuple[float, tuple[int, ...]]:
    return (
        state.normalized_score,
        knowledge_signature(state.knowledge),
    )


def prefix_signature(
    state: State,
    fields: tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(state.knowledge[field] for field in fields)


def constraint_aware_beam(
    states: list[State],
    width: int,
) -> list[State]:
    """Beam-search legal completions of the coupled MatMulV3 contract.

    Every prefix is represented only by candidates that already satisfy the
    complete kernel-template contract. The minimum analytical score among its
    legal completions is therefore an admissible completion bound for pruning;
    no repair or floating-point rounding can create an invalid tiling.
    """
    if not states:
        return []
    active = list(states)
    prefix_fields: tuple[str, ...] = ()
    for layer_index, layer in enumerate(BEAM_LAYERS, 1):
        prefix_fields += layer
        best_completion: dict[tuple[int, ...], State] = {}
        for state in active:
            prefix = prefix_signature(state, prefix_fields)
            previous = best_completion.get(prefix)
            if previous is None or state_sort_key(state) < state_sort_key(previous):
                best_completion[prefix] = state
        retained_prefixes = {
            prefix
            for prefix, _ in sorted(
                best_completion.items(),
                key=lambda item: state_sort_key(item[1]),
            )[:width]
        }
        active = [
            state
            for state in active
            if prefix_signature(state, prefix_fields) in retained_prefixes
        ]
        if not active:
            break
        for state in active:
            if not state.row.get("search_guidance", "").startswith(
                (
                    "attention_score_",
                    "skinny_n_ablation_",
                    "skinny_n_boundary_",
                    "skinny_n_low_k_",
                    "official_seed_",
                    "bottleneck_",
                )
            ):
                state.guidance = f"constraint_beam_layer_{layer_index}"
                state.row["search_guidance"] = state.guidance
    return sorted(active, key=state_sort_key)[:width]


def semantic_signature(
    state: State,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        prefix_signature(state, fields)
        for fields in NEIGHBOR_GROUPS
    )


def semantic_distance(left: State, right: State) -> int:
    return sum(
        left_group != right_group
        for left_group, right_group in zip(
            semantic_signature(left), semantic_signature(right)
        )
    )


def tabu_lns_search(
    pool: list[State],
    beam: list[State],
    tabu_iterations: int,
    lns_rounds: int,
    limit: int,
) -> list[State]:
    """Improve/diversify a legal beam with discrete Tabu and LNS moves."""
    if not beam:
        return []
    ordered_pool = sorted(pool, key=state_sort_key)
    seed_count = min(4, len(beam))
    iterations_per_seed = max(
        1, ceil_div(max(0, tabu_iterations), seed_count)
    )
    lns_period = max(
        1, ceil_div(iterations_per_seed, max(1, lns_rounds))
    )
    discovered: list[State] = []
    discovered_signatures: set[tuple[int, ...]] = set()

    def remember(state: State, guidance: str) -> None:
        signature = knowledge_signature(state.knowledge)
        if signature in discovered_signatures:
            return
        discovered_signatures.add(signature)
        if not state.row.get("search_guidance", "").startswith(
            (
                "attention_score_",
                "skinny_n_ablation_",
                "skinny_n_boundary_",
                "official_seed_",
                "bottleneck_",
            )
        ):
            state.guidance = guidance
            state.row["search_guidance"] = guidance
        discovered.append(state)

    for seed_index, start in enumerate(beam[:seed_count]):
        current = start
        best = start
        tabu_queue: list[tuple[int, ...]] = []
        tabu: set[tuple[int, ...]] = set()

        def mark_tabu(state: State) -> None:
            signature = knowledge_signature(state.knowledge)
            if signature in tabu:
                return
            tabu.add(signature)
            tabu_queue.append(signature)
            if len(tabu_queue) > 32:
                tabu.discard(tabu_queue.pop(0))

        mark_tabu(current)
        for iteration in range(iterations_per_seed):
            use_lns = lns_rounds > 0 and iteration % lns_period == 0
            current_groups = semantic_signature(current)
            if use_lns:
                first = (iteration // lns_period + seed_index) % len(
                    NEIGHBOR_GROUPS
                )
                relaxed = {first}
                if (iteration // lns_period) % 2:
                    relaxed.add((first + 1) % len(NEIGHBOR_GROUPS))
                neighborhood = [
                    candidate
                    for candidate in ordered_pool
                    if all(
                        index in relaxed
                        or candidate_group == current_groups[index]
                        for index, candidate_group in enumerate(
                            semantic_signature(candidate)
                        )
                    )
                ]
                guidance = "lns_relax_" + "_".join(
                    str(index) for index in sorted(relaxed)
                )
            else:
                neighborhood = [
                    candidate
                    for candidate in ordered_pool
                    if semantic_distance(current, candidate) == 1
                ]
                guidance = "tabu_neighbor"
            if not neighborhood:
                neighborhood = [
                    candidate
                    for candidate in ordered_pool
                    if semantic_distance(current, candidate) <= 2
                ]
            next_state = None
            for candidate in neighborhood:
                signature = knowledge_signature(candidate.knowledge)
                aspiration = state_sort_key(candidate) < state_sort_key(best)
                if signature not in tabu or aspiration:
                    next_state = candidate
                    break
            if next_state is None:
                break
            current = next_state
            if state_sort_key(current) < state_sort_key(best):
                best = current
            mark_tabu(current)
            remember(current, guidance)

    # Preserve the model leaders, then add legal neighborhood alternatives
    # that exercise different coupled parameter groups.
    result: list[State] = []
    seen: set[tuple[int, ...]] = set()
    leader_count = min(len(beam), max(1, limit // 2))
    for state in [
        *beam[:leader_count],
        *discovered,
        *beam[leader_count:],
    ]:
        signature = knowledge_signature(state.knowledge)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(state)
        if len(result) >= limit:
            break
    return result


def bank_seed_control(
    fields: list[str],
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
) -> State:
    estimate = analytical_score(
        workload, seed.bank.knowledge, hardware, seed.bank
    )
    row = row_from_state(
        fields,
        None,
        workload,
        dict(seed.bank.knowledge),
        "official_seed_bank_roundtrip",
        template_name(seed.bank.knowledge),
        1.0,
        estimate.hbm_bytes,
        estimate.l2_bytes,
        seed.key,
        guidance="bank_path_control",
        candidate_role="bank_seed_control",
        estimate=estimate,
    )
    state = State(
        row=row,
        knowledge=dict(seed.bank.knowledge),
        model_score=estimate.cycles,
        normalized_score=1.0,
        hbm_bytes=estimate.hbm_bytes,
        l2_bytes=estimate.l2_bytes,
        template=template_name(seed.bank.knowledge),
        guidance="bank_path_control",
        callback=seed.bank,
        estimate=estimate,
    )
    update_callback_columns(
        state, seed.bank, seed, estimate, hardware, workload
    )
    state.row["search_model_ratio_vs_bank_seed"] = "1"
    state.row["search_model_score"] = "1"
    state.row["rank"] = "0"
    return state


def state_history_key(
    workload: Workload,
    knowledge: dict[str, int],
) -> tuple[str, ...]:
    m_blocks = ceil_div(workload.m, knowledge["singleCoreM"])
    n_blocks = ceil_div(workload.n, knowledge["singleCoreN"])
    return (
        workload.workload_id,
        f"{workload.m}x{workload.n}x{workload.k}",
        workload.dtype,
        template_name(knowledge),
        f"{knowledge['baseM']}x{knowledge['baseN']}x{knowledge['baseK']}",
        (
            f"{knowledge['singleCoreM']}x{knowledge['singleCoreN']}x"
            f"{knowledge['singleCoreK']}"
        ),
        str(knowledge["usedCoreNum"]),
        f"{m_blocks}x{n_blocks}",
        (
            f"{knowledge['l2MTileCnt']}x{knowledge['l2NTileCnt']}"
            f"({knowledge['l2MTileBlock']}x{knowledge['l2NTileBlock']})"
        ),
        str(knowledge["iterateOrder"]),
        f"{knowledge['depthA1']}x{knowledge['depthB1']}",
        f"{knowledge['dbL0A']}x{knowledge['dbL0B']}x{knowledge['dbL0C']}",
        str(knowledge["l2IterateOrder"]),
        str(int(workload.trans_a)),
        str(int(workload.trans_b)),
    )


def load_measurement_history(
    path: Path | None,
    soc: str,
    aic_cores: int,
) -> dict[tuple[str, ...], list[MeasurementEvidence]]:
    if path is None or not path.is_file():
        return {}
    _, rows = read_csv(path)
    valid_rows = [
        row
        for row in rows
        if (
            truthy(row.get("ocr_complete"))
            and row.get("preflight_contract") == "grid9_v1"
            and row.get("soc") == soc
            and int(row.get("aic") or 0) == aic_cores
        )
    ]
    bank_controls: dict[tuple[str, str], float] = {}
    for row in valid_rows:
        if row.get("record_type") != "bank_control":
            continue
        try:
            median_ms = float(row.get("median_ms") or 0)
        except ValueError:
            continue
        if median_ms > 0:
            bank_controls[
                (row.get("workload_id", ""), row.get("run_id", ""))
            ] = median_ms

    history: dict[
        tuple[str, ...], list[MeasurementEvidence]
    ] = {}
    for row in valid_rows:
        if row.get("record_type") != "candidate":
            continue
        try:
            speedup = float(row.get("speedup_vs_official") or 0)
            candidate_ms = float(row.get("median_ms") or 0)
        except ValueError:
            continue
        if speedup <= 0 or candidate_ms <= 0:
            continue
        control_ms = bank_controls.get(
            (row.get("workload_id", ""), row.get("run_id", ""))
        )
        ratio_vs_bank = (
            candidate_ms / control_ms
            if control_ms is not None and control_ms > 0
            else None
        )
        key = (
            row.get("workload_id", ""),
            row.get("shape", ""),
            row.get("dtype", "").lower(),
            row.get("template", ""),
            row.get("T", ""),
            row.get("S", ""),
            row.get("C", ""),
            row.get("G", ""),
            row.get("L2", ""),
            row.get("I", ""),
            row.get("L1", ""),
            row.get("DB", ""),
            row.get("L2O", ""),
            row.get("trans_a", ""),
            row.get("trans_b", ""),
        )
        history.setdefault(key, []).append(
            MeasurementEvidence(
                ratio_vs_official=1.0 / speedup,
                ratio_vs_bank=ratio_vs_bank,
                record_id=row.get("record_key", ""),
            )
        )
    return history


def completed_bottleneck_frontier_coverage(
    path: Path | None,
    soc: str,
    aic_cores: int,
) -> bool:
    """Return whether the 46-workload, 47-candidate campaign completed."""
    if path is None or not path.is_file():
        return False
    _, rows = read_csv(path)
    return any(
        row.get("record_type") == "coverage"
        and row.get("soc") == soc
        and int(row.get("aic") or 0) == aic_cores
        and truthy(row.get("ocr_complete"))
        and row.get("status") == "success"
        and row.get("global_index") == "1-47"
        and row.get("workload_index") == "1-46"
        and "exact_resume_137" in row.get("notes", "")
        for row in rows
    )


def calibrate_from_history(
    workload: Workload,
    states: list[State],
    history: dict[tuple[str, ...], list[MeasurementEvidence]],
    bank_path_diff: str,
) -> tuple[float, int]:
    exact: dict[tuple[str, ...], tuple[float, str]] = {}
    correction_by_key: dict[tuple[str, ...], float] = {}
    for state in states:
        key = state_history_key(workload, state.knowledge)
        records = history.get(key)
        if not records:
            continue
        usable = []
        for record in records:
            if record.ratio_vs_bank is not None:
                usable.append(record.ratio_vs_bank)
            elif not bank_path_diff:
                usable.append(record.ratio_vs_official)
            elif record.ratio_vs_official <= 0.90:
                # A missing same-run bank control must not erase a large,
                # measured official improvement. The 10% guard is well above
                # the observed bank/default drift and is not used to infer
                # marginal wins.
                usable.append(record.ratio_vs_official)
        if not usable:
            continue
        actual_ratio = statistics.median(
            usable
        )
        record_ids = "|".join(
            sorted({
                record.record_id
                for record in records
                if record.record_id
            })
        )
        exact[key] = (actual_ratio, record_ids)
        correction_by_key[key] = (
            actual_ratio / max(1.0e-12, state.normalized_score)
        )

    correction = 1.0
    if len(correction_by_key) >= 2:
        correction = statistics.median(correction_by_key.values())
        # Residual calibration corrects local model bias; larger shifts require
        # a new bank control and are not extrapolated to unseen tilings.
        correction = min(1.15, max(0.85, correction))

    for state in states:
        key = state_history_key(workload, state.knowledge)
        raw_ratio = state.normalized_score
        if key in exact:
            adjusted, record_ids = exact[key]
            state.row["search_history_match"] = record_ids
            confidence = "measured_history"
        else:
            adjusted = raw_ratio * correction
            confidence = (
                "historically_calibrated"
                if len(correction_by_key) >= 2
                else state.row.get("search_model_confidence", "medium")
            )
        state.normalized_score = adjusted
        state.row.update(
            {
                "search_model_raw_ratio_vs_bank_seed": f"{raw_ratio:.12g}",
                "search_model_ratio_vs_bank_seed": f"{adjusted:.12g}",
                "search_model_score": f"{adjusted:.12g}",
                "search_model_calibration": f"{correction:.12g}",
                "search_model_confidence": confidence,
            }
        )
    return correction, len(exact)


def measured_skinny_anchor_count(
    history: dict[tuple[str, ...], list[MeasurementEvidence]],
) -> int:
    anchors: set[str] = set()
    for key, records in history.items():
        try:
            m, n, k = (int(value) for value in key[1].split("x"))
        except (IndexError, ValueError):
            continue
        if m < 3072 or n > 31 or k != 16384:
            continue
        if any(
            record.ratio_vs_bank is not None
            and record.ratio_vs_bank < 0.90
            for record in records
        ):
            anchors.add(key[0])
    return len(anchors)


def state_has_history_match(state: State) -> bool:
    value = str(state.row.get("search_history_match") or "").strip()
    return bool(value) and value.lower() not in {"0", "false", "no", "off"}


def guided_action_frontier(
    workload: Workload,
    seed: Seed,
    hardware: Hardware,
    eligible: list[State],
    model_ratio_limit: float,
    skinny_anchor_count: int,
) -> list[State]:
    action_leaders: dict[str, State] = {}
    for state in eligible:
        preregistered_new = (
            state.row.get("search_resume_policy") == "allow_new"
            and state.guidance.startswith("skinny_n_boundary_")
        )
        if state.normalized_score > model_ratio_limit and not preregistered_new:
            continue
        if preregistered_new and state.normalized_score > model_ratio_limit:
            state.row["search_model_confidence"] = "preregistered_new"
            state.row["search_stop_reason"] = (
                "source_guided_allow_new_candidate_despite_proxy_rejection"
            )
        previous = action_leaders.get(state.guidance)
        if (
            previous is None
            or state_sort_key(state) < state_sort_key(previous)
        ):
            action_leaders[state.guidance] = state

    evidence_transfer: list[State] = []
    if (
        skinny_anchor_count >= 3
        and skinny_n_low_k_causal_applicable(
            workload, seed, hardware
        )
    ):
        evidence_transfer = [
            state
            for state in eligible
            if state.guidance.startswith("skinny_n_low_k_")
            and not state_has_history_match(state)
        ]
    elif (
        not action_leaders
        and skinny_anchor_count >= 3
        and skinny_n_large_k_applicable(workload, seed, hardware)
    ):
        evidence_transfer = [
            state
            for state in eligible
            if not state_has_history_match(state)
        ][:1]

    for transfer in evidence_transfer:
        if transfer.guidance not in action_leaders:
            transfer.row["search_model_confidence"] = "evidence_transfer"
            transfer.row["search_rationale"] = (
                transfer.row.get("search_rationale", "")
                + "; retain the preregistered causal transition because at "
                "least three K=16384 skinny-N shapes beat both controls"
            ).lstrip("; ")
            transfer.row["search_stop_reason"] = (
                "evidence_transfer_candidate_despite_proxy_rejection"
            )
            action_leaders[transfer.guidance] = transfer

    return sorted(
        action_leaders.values(),
        key=lambda state: (
            0
            if state.row.get("search_model_confidence")
            == "measured_history"
            else 1,
            -float(state.row.get("search_transition_gain") or 0),
            state.normalized_score,
            state.guidance,
        ),
    )


def completed_frontier_candidate_allowed(state: State) -> bool:
    """Keep measured schedules or explicitly preregistered new experiments."""
    return (
        state_has_history_match(state)
        or state.row.get("search_resume_policy") == "allow_new"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-output", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--tabu-iters", type=int, default=64)
    parser.add_argument("--lns-rounds", type=int, default=8)
    parser.add_argument("--model-ratio-limit", type=float, default=1.03)
    parser.add_argument(
        "--optimization-scope",
        choices=(
            "bottleneck_guided_v1",
            "official_local_v2",
            "skinny_n_large_k_v1",
            "all_templates_validation",
        ),
        default="bottleneck_guided_v1",
    )
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--l0a-bytes", type=int, required=True)
    parser.add_argument("--l0b-bytes", type=int, required=True)
    parser.add_argument("--l0c-bytes", type=int, required=True)
    parser.add_argument("--l1-bytes", type=int, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument(
        "--l2-bytes-per-cycle-per-core", type=float, required=True
    )
    parser.add_argument(
        "--hbm-bytes-per-cycle-per-core", type=float, required=True
    )
    args = parser.parse_args()
    hardware = Hardware(
        aic_cores=args.aic_cores,
        l0a_bytes=args.l0a_bytes,
        l0b_bytes=args.l0b_bytes,
        l0c_bytes=args.l0c_bytes,
        l1_bytes=args.l1_bytes,
        l2_bytes=args.l2_bytes,
        l2_bytes_per_cycle_per_core=args.l2_bytes_per_cycle_per_core,
        hbm_bytes_per_cycle_per_core=args.hbm_bytes_per_cycle_per_core,
    )
    if (
        args.top_k <= 0
        or args.beam_width <= 0
        or args.tabu_iters < 0
        or args.lns_rounds < 0
        or args.model_ratio_limit <= 0
        or any(
            value <= 0
            for value in (
                hardware.aic_cores,
                hardware.l0a_bytes,
                hardware.l0b_bytes,
                hardware.l0c_bytes,
                hardware.l1_bytes,
                hardware.l2_bytes,
                hardware.l2_bytes_per_cycle_per_core,
                hardware.hbm_bytes_per_cycle_per_core,
            )
        )
    ):
        raise SearchError("search limits and platform capacities must be positive")

    from tbe.common.platform import set_current_compile_soc_info
    from tbe.common.utils import op_tiling

    set_current_compile_soc_info(args.soc)
    op_tiling._RT_BANK_CACHE = {}

    raw_fields, raw_rows = read_csv(args.raw_candidates)
    output_fields = ["rank", *(field for field in raw_fields if field != "rank")]
    for column in EXTRA_COLUMNS:
        if column not in output_fields:
            output_fields.append(column)
    workloads = load_workloads(args.workloads)
    history = load_measurement_history(
        args.history, args.soc, hardware.aic_cores
    )
    completed_frontier = completed_bottleneck_frontier_coverage(
        args.history, args.soc, hardware.aic_cores
    )
    skinny_anchor_count = measured_skinny_anchor_count(history)

    # Capture both the no-bank default and the exact same 23 fields after the
    # RuntimeKb injection path. MatMulV3 derives ND->NZ fields after the bank
    # lookup, so these two blobs are not assumed to be identical.
    seeds: dict[str, Seed] = {}
    for workload in workloads:
        if workload.dtype in DTYPE_NAME:
            seeds[workload.workload_id] = parse_seed(workload)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        grouped.setdefault(row["workload_id"], []).append(row)

    selected_rows: list[dict[str, str]] = []
    all_destination = None
    all_writer = None
    if args.all_output:
        args.all_output.parent.mkdir(parents=True, exist_ok=True)
        all_destination = args.all_output.open(
            "w", newline="", encoding="utf-8"
        )
        all_writer = csv.DictWriter(
            all_destination,
            fieldnames=output_fields,
            extrasaction="ignore",
        )
        all_writer.writeheader()

    def write_all(row: dict[str, str]) -> None:
        if all_writer is not None:
            all_writer.writerow(row)

    for workload in workloads:
        if workload.dtype not in DTYPE_NAME:
            continue
        seed = seeds[workload.workload_id]
        bank_path_diff = callback_derived_diff(
            seed.default, seed.bank
        )
        bank_seed_estimate = analytical_score(
            workload, seed.bank.knowledge, hardware, seed.bank
        )
        bottleneck = diagnose_bottleneck(
            workload,
            seed.bank.knowledge,
            bank_seed_estimate,
            hardware,
        )
        seed_score = bank_seed_estimate.cycles
        prototype = next(iter(grouped.get(workload.workload_id, [])), None)
        states: list[State] = []
        proposal_by_signature: dict[
            tuple[int, ...], CandidateProposal
        ] = {}
        stop_reason = ""
        if args.optimization_scope == "bottleneck_guided_v1":
            proposals, stop_reason = bottleneck_guided_candidate_proposals(
                workload,
                seed,
                hardware,
                bank_seed_estimate,
                bottleneck,
            )
            proposal_by_signature = {
                knowledge_signature(proposal.knowledge): proposal
                for proposal in proposals
            }
            template_spaces: dict[str, list[dict[str, int]]] = {}
            for proposal in proposals:
                template_spaces.setdefault(
                    template_name(proposal.knowledge), []
                ).append(proposal.knowledge)
        else:
            template_spaces = all_template_candidate_spaces(
                workload,
                seed,
                grouped.get(workload.workload_id, []),
                hardware,
                args.beam_width,
                args.optimization_scope,
            )
        control = bank_seed_control(
            output_fields, workload, seed, hardware
        )
        control.row["search_bottleneck"] = bottleneck.dominant
        control.row["search_rationale"] = bottleneck.summary()
        control.row["search_resume_policy"] = "require_existing"
        control.row["search_stop_reason"] = stop_reason
        selected_rows.append(dict(control.row))
        write_all(control.row)

        callback_limit = max(args.top_k * 3, args.beam_width)
        callback_beams: dict[str, list[State]] = {}
        search_counts: dict[str, tuple[int, int, int]] = {}
        seed_signature = knowledge_signature(seed.bank.knowledge)
        for template, candidates in sorted(template_spaces.items()):
            template_states: list[State] = []
            for knowledge in candidates:
                if (
                    not hard_legal(workload, knowledge, hardware)
                    or knowledge_signature(knowledge) == seed_signature
                ):
                    continue
                estimate = analytical_score(
                    workload, knowledge, hardware
                )
                normalized = estimate.cycles / max(seed_score, 1.0)
                proposal = proposal_by_signature.get(
                    knowledge_signature(knowledge)
                )
                if proposal is not None:
                    guidance = proposal.guidance
                else:
                    guidance = (
                        focused_candidate_guidance(knowledge)
                        if args.optimization_scope == "skinny_n_large_k_v1"
                        else (
                            official_local_guidance(knowledge, seed)
                            if args.optimization_scope == "official_local_v2"
                            else "constraint_generated"
                        )
                    )
                row = row_from_state(
                    output_fields,
                    prototype,
                    workload,
                    knowledge,
                    (
                        "cann81_bottleneck_guided_v1"
                        if args.optimization_scope == "bottleneck_guided_v1"
                        else "cann81_skinny_n_v1_beam_tabu_lns"
                        if args.optimization_scope == "skinny_n_large_k_v1"
                        else (
                            "cann81_official_local_v2"
                            if args.optimization_scope == "official_local_v2"
                            else "cann81_constraint_beam_tabu_lns"
                        )
                    ),
                    template,
                    normalized,
                    estimate.hbm_bytes,
                    estimate.l2_bytes,
                    seed.key,
                    guidance=guidance,
                    estimate=estimate,
                    bottleneck=bottleneck.dominant,
                    rationale=(
                        proposal.rationale if proposal is not None else ""
                    ),
                    transition_gain=(
                        proposal.transition_gain
                        if proposal is not None
                        else 0.0
                    ),
                    resume_policy=(
                        proposal.resume_policy
                        if proposal is not None
                        else "require_existing"
                    ),
                    stop_reason=stop_reason,
                )
                template_states.append(
                    State(
                        row=row,
                        knowledge=knowledge,
                        model_score=estimate.cycles,
                        normalized_score=normalized,
                        hbm_bytes=estimate.hbm_bytes,
                        l2_bytes=estimate.l2_bytes,
                        template=template,
                        guidance=guidance,
                        estimate=estimate,
                    )
                )
            template_states.sort(key=state_sort_key)
            beam = constraint_aware_beam(
                template_states, args.beam_width
            )
            if args.optimization_scope in {
                "bottleneck_guided_v1",
                "official_local_v2",
                "skinny_n_large_k_v1",
            }:
                frontier = beam
            else:
                frontier = tabu_lns_search(
                    template_states,
                    beam,
                    args.tabu_iters,
                    args.lns_rounds,
                    callback_limit,
                )
            if frontier:
                callback_beams[template] = frontier
            search_counts[template] = (
                len(template_states), len(beam), len(frontier)
            )
            states.extend(template_states)
        states.sort(key=state_sort_key)

        accepted: list[State] = []
        seen: set[tuple[int, ...]] = set()
        failures = 0
        first_failure = ""

        def try_callback(state: State) -> bool:
            nonlocal failures, first_failure
            signature = knowledge_signature(state.knowledge)
            if signature in seen:
                return False
            seen.add(signature)
            try:
                callback = validate_callback(workload, state)
            except Exception as exception:
                failures += 1
                if not first_failure:
                    first_failure = str(exception).replace("\n", " ")[:240]
                return False
            update_callback_columns(
                state,
                callback,
                seed,
                bank_seed_estimate,
                hardware,
                workload,
            )
            accepted.append(state)
            return True

        # First secure one valid representative of every generated template.
        for template in sorted(callback_beams):
            for state in callback_beams[template]:
                if try_callback(state):
                    break

        remaining = sorted(
            (
                state
                for template_states in callback_beams.values()
                for state in template_states
                if knowledge_signature(state.knowledge) not in seen
            ),
            key=lambda state: (
                state.normalized_score,
                knowledge_signature(state.knowledge),
            ),
        )
        callback_target = (
            len(states)
            if args.optimization_scope == "bottleneck_guided_v1"
            else args.top_k
        )
        for state in remaining:
            try_callback(state)
            if len(accepted) >= callback_target:
                break

        history_correction, history_matches = calibrate_from_history(
            workload,
            accepted,
            history,
            bank_path_diff,
        )
        for state in accepted:
            if state_has_history_match(state):
                state.row["search_resume_policy"] = "require_existing"
        accepted.sort(
            key=state_sort_key
        )
        callback_valid = len(accepted)
        eligible = [
            state
            for state in accepted
            if not dominated_by_bank_seed(state.knowledge, seed)
        ]
        if (
            completed_frontier
            and args.optimization_scope == "bottleneck_guided_v1"
        ):
            eligible = [
                state
                for state in eligible
                if completed_frontier_candidate_allowed(state)
            ]
        chosen: list[State] = []
        chosen_signatures: set[tuple[int, ...]] = set()
        if args.optimization_scope == "bottleneck_guided_v1":
            # A transition frontier retains at most one representative per
            # diagnosed action. Exact measured regressions are not sent back;
            # preregistered allow_new actions may override the proxy model.
            ordered_candidates = guided_action_frontier(
                workload,
                seed,
                hardware,
                eligible,
                args.model_ratio_limit,
                skinny_anchor_count,
            )
        else:
            template_leaders: list[State] = []
            leader_templates: set[str] = set()
            for state in eligible:
                if state.template in leader_templates:
                    continue
                leader_templates.add(state.template)
                template_leaders.append(state)
            preferred = [
                state
                for state in eligible
                if state.normalized_score <= args.model_ratio_limit
            ]
            ordered_candidates = [
                *template_leaders, *preferred, *eligible
            ]
        for state in ordered_candidates:
            signature = knowledge_signature(state.knowledge)
            if signature in chosen_signatures:
                continue
            chosen_signatures.add(signature)
            chosen.append(state)
            if len(chosen) >= args.top_k:
                break

        for rank, state in enumerate(chosen, 1):
            if (
                completed_frontier
                and args.optimization_scope == "bottleneck_guided_v1"
                and state.row.get("search_resume_policy") != "allow_new"
                and not state.guidance.startswith("skinny_n_low_k_")
            ):
                state.row["search_resume_policy"] = "require_existing"
            state.row["rank"] = str(rank)
            selected_rows.append(dict(state.row))
        for rank, state in enumerate(states, 1):
            state.row["rank"] = str(rank)
            write_all(state.row)
        if all_destination is not None:
            all_destination.flush()

        print(
            f"template_search: {workload.workload_id} "
            f"scope={args.optimization_scope} "
            f"official={template_name(seed.knowledge)} "
            f"bank_seed_diff="
            f"{bank_path_diff or 'none'} "
            f"families="
            f"{','.join(f'{name}:{counts[0]}/{counts[1]}/{counts[2]}' for name, counts in search_counts.items()) or 'none'} "
            f"generated={len(states)} callback_valid={callback_valid} "
            f"selected_for_npu={len(chosen)} "
            f"preferred_model_ratio={args.model_ratio_limit:.3g} "
            f"history_matches={history_matches} "
            f"history_correction={history_correction:.4g} "
            f"callback_rejected={failures} "
            f"{bottleneck.summary()} "
            f"actions="
            f"{','.join(state.guidance for state in chosen) or 'none'} "
            f"stop={stop_reason or 'frontier_selected'}"
            f"{f' first_rejection={first_failure}' if first_failure else ''}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)
    if all_destination is not None:
        all_destination.close()
    print(
        f"template_search_completed: workloads={len(workloads)} "
        f"searched_selected="
        f"{sum(row.get('candidate_role') == 'searched' for row in selected_rows)} "
        f"bank_seed_controls="
        f"{sum(row.get('candidate_role') == 'bank_seed_control' for row in selected_rows)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SearchError, OSError, ValueError, KeyError) as exception:
        print(f"fatal: {exception}", flush=True)
        raise SystemExit(1)
