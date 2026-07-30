from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from tiling_search.contracts import template_of
from tiling_search.domain import KNOWLEDGE_FIELDS, Schedule, Workload


DTYPE_NAME = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}


class CallbackError(RuntimeError):
    pass


@dataclass(frozen=True)
class CallbackTiling:
    key: int
    block_dim: int
    workspaces: tuple[int, ...]
    blob: bytes
    sha256: str
    schedule: Schedule


def _tensor_desc(name: str, shape: list[int], dtype: str) -> dict:
    return {
        "name": name,
        "shape": shape,
        "ori_shape": shape,
        "format": "ND",
        "ori_format": "ND",
        "dtype": DTYPE_NAME[dtype],
    }


def _decode_tiling_enable(tiling_key: int) -> int:
    # The public key suffix is FIX,SPECIAL,FULL,SPLIT,MIX. RuntimeKb stores
    # split + 10*full + 1000*fix in the 23-field tuning record.
    suffix = tiling_key % 100000
    split = (suffix // 10) % 10
    full = (suffix // 100) % 10
    fix = (suffix // 10000) % 10
    return split + 10 * full + 1000 * fix


def _parse_result(result: dict) -> CallbackTiling:
    blob = bytes(result["tiling_data"])
    if len(blob) < 220 or len(blob) % 4:
        raise CallbackError(
            f"unexpected MatMulV3 tiling_data size={len(blob)}"
        )
    cube = struct.unpack_from("<50i", blob, 0)
    l2 = struct.unpack_from("<5I", blob, 200)
    schedule = Schedule.from_mapping(
        {
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
            "tilingEnable": _decode_tiling_enable(int(result["tiling_key"])),
        }
    )
    return CallbackTiling(
        key=int(result["tiling_key"]),
        block_dim=int(result["block_dim"]),
        workspaces=tuple(int(value) for value in result.get("workspaces") or [0]),
        blob=blob,
        sha256=hashlib.sha256(blob).hexdigest(),
        schedule=schedule,
    )


def invoke(
    workload: Workload,
    injected: Schedule | None = None,
) -> CallbackTiling:
    from tbe.common.utils import op_tiling

    shape_a = (
        [workload.k, workload.m]
        if workload.trans_a
        else [workload.m, workload.k]
    )
    shape_b = (
        [workload.n, workload.k]
        if workload.trans_b
        else [workload.k, workload.n]
    )
    attrs = [
        {
            "name": "transpose_x1",
            "dtype": "bool",
            "value": workload.trans_a,
        },
        {
            "name": "transpose_x2",
            "dtype": "bool",
            "value": workload.trans_b,
        },
        {"name": "offset_x", "dtype": "int", "value": 0},
        {"name": "enable_hf32", "dtype": "bool", "value": False},
    ]
    op_tiling._RT_BANK_CACHE = (
        {} if injected is None else injected.as_dict()
    )
    result = op_tiling.do_op_tiling(
        "MatMulV3",
        {},
        [
            _tensor_desc("x1", shape_a, workload.dtype),
            _tensor_desc("x2", shape_b, workload.dtype),
            None,
            None,
        ],
        [_tensor_desc("y", [workload.m, workload.n], workload.dtype)],
        attrs=attrs,
    )
    return _parse_result(result)


def exact_roundtrip(
    workload: Workload,
    schedule: Schedule,
) -> CallbackTiling:
    callback = invoke(workload, schedule)
    mismatches = [
        f"{field}={callback.schedule[field]} expected={schedule[field]}"
        for field in KNOWLEDGE_FIELDS
        if callback.schedule[field] != schedule[field]
    ]
    if mismatches:
        raise CallbackError("callback changed candidate: " + "; ".join(mismatches))
    expected = template_of(schedule)
    observed = template_of(callback.schedule)
    if observed != expected:
        raise CallbackError(
            f"callback selected template={observed.value}, "
            f"expected={expected.value}"
        )
    return callback
