"""Small command-line entry point for the parameter-only simulator."""

from __future__ import annotations

import argparse
import json

from .hardware import ascend_910b3
from .operators import (
    elementwise_add,
    flash_attention_forward,
    flash_attention_score_grad,
    gather_elements,
    matmul,
    reduce_sum,
    scatter_add,
    scatter_elements,
)
from .schedule import ScheduleSpace, SearchPolicy
from .solver import solve


def _shape(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.lower().replace("x", ",").split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("shape must contain positive dimensions")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic NPU tiling cycle simulator")
    parser.add_argument(
        "operator",
        choices=(
            "matmul", "add", "reduce", "gather", "scatter",
            "flash-attention", "flash-attention-grad",
        ),
    )
    parser.add_argument("--shape", type=_shape, required=True)
    parser.add_argument("--source-shape", type=_shape)
    parser.add_argument("--axis", type=int, default=-1)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--index-dtype", default="int32")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-evaluations", type=int, default=100_000)
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    args = parser.parse_args()

    if args.operator == "matmul":
        if len(args.shape) != 3:
            parser.error("matmul --shape must be M,N,K")
        operator = matmul(*args.shape, args.dtype, trans_a=args.trans_a, trans_b=args.trans_b)
    elif args.operator == "add":
        operator = elementwise_add(args.shape, args.dtype)
    elif args.operator == "reduce":
        operator = reduce_sum(args.shape, args.axis, args.dtype)
    elif args.operator == "gather":
        if args.source_shape is None:
            parser.error("gather requires --source-shape")
        operator = gather_elements(
            args.source_shape, args.shape, args.axis, args.dtype, args.index_dtype
        )
    elif args.operator == "scatter":
        operator = scatter_elements(
            args.source_shape or args.shape,
            args.shape,
            args.axis,
            args.dtype,
            args.index_dtype,
            reduction="add",
        )
    elif args.operator == "flash-attention":
        if len(args.shape) != 5:
            parser.error("flash-attention --shape must be B,H,Q,KV,D")
        operator = flash_attention_forward(*args.shape, args.dtype)
    else:
        if len(args.shape) != 6:
            parser.error("flash-attention-grad --shape must be B,QH,KVH,Q,KV,D")
        operator = flash_attention_score_grad(*args.shape, args.dtype)

    result = solve(
        operator,
        ascend_910b3(),
        ScheduleSpace(),
        SearchPolicy(top_k=args.top_k, max_evaluations=args.max_evaluations),
    )
    payload = {
        "operator": operator.name,
        "hardware": "Ascend910B3",
        "evaluated": result.evaluated,
        "legal": result.legal,
        "rejected": result.rejected,
        "exhaustive": result.exhaustive,
        "ranked": [item.as_dict() for item in result.ranked],
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0 if result.ranked else 2


if __name__ == "__main__":
    raise SystemExit(main())
