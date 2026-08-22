#!/usr/bin/env python3
"""Deterministic non-MatMul workloads for original-source candidate collection.

This is a semantic workload catalog, not a tiling generator. Candidate counts
come solely from the pinned source inventory: FASG has eight registrations;
FusedInferAttentionScore, GatherElementsV2 and ScatterElementsV2 choose only
the source path their original predicates admit.  No Cartesian tiling search is
present here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any


FORMAL_LATENCY_TARGET = 20_000
FASG_NATIVE_STRATEGIES = 8
# The complete source strategy registry is audited for every FASG workload.
# These three original strategies have overlapping, documented capability for
# the source-multi-tiling lattice below: fp16, BNSD, equal Q/KV head counts,
# and both sequence lengths below 1024.  They are not synthetic variants.
FASG_MULTI_TILING_SOURCE_CLASSES = (
    "FlashAttentionScoreGradTilingS1s2Bn2",
    "FlashAttentionScoreGradTilingS1s2Bn2gs1s2",
    "FlashAttentionScoreGradTilingS1s2Bn2gs1s2SameAb",
)
FASG_MULTI_TILING_RESERVE_SHAPES = 11_000


def row(op: str, index: int, tags: str, **parameters: Any) -> dict[str, Any]:
    return {
        "workload_id": f"{op}_{index:03d}",
        "op": op,
        "coverage": tags.split(","),
        **parameters,
    }


def attention_grad_workloads() -> list[dict[str, Any]]:
    # These are explicit geometry families, not a generated random grid. Every
    # tuple keeps headDim 16-aligned and qHeads divisible by kvHeads.
    cases = [
        (1, 1, 1, 64, 64, 64, "fp16", "BNSD", "small,aligned"),
        (1, 4, 4, 128, 128, 64, "fp16", "BNSD", "m_head,aligned"),
        (2, 8, 8, 256, 256, 128, "fp16", "BNSD", "m_batch,m_head"),
        (1, 8, 8, 127, 193, 128, "fp16", "BNSD", "q_tail,kv_tail"),
        (2, 4, 4, 257, 129, 64, "fp16", "BNSD", "q_tail,kv_tail"),
        (4, 2, 2, 64, 512, 128, "fp16", "BNSD", "m_batch,long_kv"),
        (1, 16, 16, 512, 512, 64, "fp16", "BNSD", "many_heads,long_seq"),
        (2, 8, 8, 128, 1024, 64, "fp16", "BNSD", "long_kv"),
        (1, 1, 1, 64, 64, 128, "fp32", "BNSD", "fp32,aligned"),
        (1, 4, 4, 129, 257, 64, "fp32", "BNSD", "fp32,tail"),
        (2, 8, 8, 256, 256, 128, "fp32", "BNSD", "fp32,m_batch"),
        (1, 4, 4, 128, 128, 64, "bf16", "BNSD", "bf16,aligned"),
        (1, 1, 1, 64, 64, 64, "fp16", "SBH", "sbh,small"),
        (1, 4, 4, 128, 128, 64, "fp16", "SBH", "sbh,m_head"),
        (2, 8, 8, 257, 129, 128, "fp16", "SBH", "sbh,tail"),
        (1, 4, 4, 128, 512, 128, "fp16", "SBH", "sbh,long_kv"),
        (1, 1, 1, 64, 64, 128, "fp32", "SBH", "sbh,fp32"),
        (2, 4, 4, 129, 257, 64, "fp32", "SBH", "sbh,fp32,tail"),
        (1, 8, 8, 384, 384, 64, "fp16", "BNSD", "seq_384"),
        (2, 16, 16, 96, 320, 64, "fp16", "BNSD", "head_parallel"),
        (4, 4, 4, 192, 192, 128, "fp16", "BNSD", "batch_parallel"),
        (1, 2, 2, 33, 65, 192, "fp16", "BNSD", "d192,tail"),
        (1, 8, 8, 1024, 128, 64, "fp16", "BNSD", "long_q,short_kv"),
        (2, 1, 1, 63, 1025, 128, "fp16", "BNSD", "long_kv_boundary"),
        (1, 8, 2, 128, 256, 64, "fp16", "BNSD", "gqa"),
        (2, 16, 1, 257, 129, 128, "fp16", "BNSD", "mqa,tail"),
        (1, 4, 4, 64, 64, 64, "fp16", "BSND", "bsnd,aligned"),
        (2, 8, 2, 129, 257, 128, "fp16", "BSND", "bsnd,gqa,tail"),
        (1, 8, 8, 512, 128, 64, "fp32", "BSND", "bsnd,fp32,long_q"),
        (1, 4, 4, 64, 64, 128, "fp16", "BSH", "bsh,aligned"),
        (2, 8, 2, 257, 129, 64, "fp16", "BSH", "bsh,gqa,tail"),
        (1, 16, 1, 128, 1024, 64, "fp16", "BSH", "bsh,mqa,long_kv"),
        # Additional, independently reviewed boundary families.
        (1, 2, 2, 96, 96, 64, "fp16", "BNSD", "small,seq_96"),
        (1, 2, 2, 192, 384, 64, "fp16", "BNSD", "mid,asymmetric"),
        (2, 8, 4, 384, 768, 128, "fp16", "BNSD", "gqa,long_kv"),
        (1, 16, 4, 768, 256, 64, "fp16", "BNSD", "gqa,long_q"),
        (2, 32, 8, 128, 128, 64, "fp16", "BNSD", "many_heads,gqa"),
        (1, 4, 1, 511, 257, 128, "fp16", "BNSD", "mqa,dual_tail"),
        (1, 8, 8, 1024, 1024, 128, "fp16", "BNSD", "long_square"),
        (2, 4, 2, 33, 513, 64, "fp16", "BNSD", "gqa,tail_long_kv"),
        (1, 8, 2, 256, 512, 192, "fp16", "BNSD", "gqa,d192"),
        (1, 4, 4, 384, 128, 128, "bf16", "BNSD", "bf16,long_q"),
        (2, 8, 8, 128, 384, 128, "fp32", "BNSD", "fp32,asymmetric"),
        (1, 4, 4, 256, 768, 64, "fp16", "SBH", "sbh,long_kv"),
        (2, 8, 4, 384, 192, 64, "fp16", "SBH", "sbh,gqa"),
        (1, 16, 16, 192, 192, 128, "fp16", "SBH", "sbh,many_heads"),
        (1, 4, 1, 129, 513, 64, "fp16", "SBH", "sbh,mqa,tail"),
        (2, 8, 8, 96, 320, 64, "fp16", "BSND", "bsnd,asymmetric"),
        (1, 8, 2, 512, 256, 128, "fp16", "BSND", "bsnd,gqa,long_q"),
        (1, 4, 4, 257, 513, 64, "fp16", "BSH", "bsh,dual_tail"),
        (2, 16, 4, 128, 768, 128, "fp16", "BSH", "bsh,gqa,long_kv"),
        (1, 8, 8, 384, 384, 128, "fp32", "BSH", "bsh,fp32,seq_384"),
        (1, 2, 2, 65, 33, 64, "fp16", "BNSD", "inverse_tail"),
        (4, 8, 2, 64, 256, 64, "fp16", "BNSD", "batch,gqa"),
        (1, 32, 32, 64, 128, 64, "fp16", "BNSD", "head_count_32"),
        (1, 8, 8, 1536, 256, 64, "fp16", "BNSD", "very_long_q"),
        (1, 8, 8, 256, 1536, 64, "fp16", "BNSD", "very_long_kv"),
        (2, 4, 4, 511, 511, 128, "fp16", "BNSD", "square_tail"),
        (1, 8, 1, 96, 384, 128, "fp16", "BSH", "bsh,mqa"),
        (1, 8, 2, 192, 768, 64, "fp16", "BSND", "bsnd,gqa,ratio4"),
        (1, 4, 4, 128, 128, 192, "fp16", "BNSD", "d192,aligned"),
        (2, 16, 16, 320, 320, 64, "fp16", "BNSD", "seq_320"),
        (1, 4, 4, 256, 256, 256, "fp16", "BNSD", "d256"),
    ]
    return [row("flash_attention_score_grad", index, tags + "," + dtype + "," + layout.lower(),
                dtype=dtype, layout=layout, batch=batch, q_heads=q_heads, kv_heads=kv_heads,
                q_seq=q_seq, kv_seq=kv_seq, head_dim=head_dim)
            for index, (batch, q_heads, kv_heads, q_seq, kv_seq, head_dim, dtype, layout, tags)
            in enumerate(cases)]


def attention_grad_multi_tiling_reserve(start_index: int) -> list[dict[str, Any]]:
    """Return a fixed, legal geometry lattice for source multi-tiling.

    This is a shape-coverage lattice, not a product of *tiling fields*.  All
    levels below were reviewed against the three source ``IsCapable`` rules:
    fp16, BNSD, Q heads equal KV heads, 16-aligned head dimension, and Q/KV
    sequence length strictly below 1024.  A coprime walk gives each factor a
    balanced deterministic distribution in every prefix while keeping the
    requested reserve finite and without random sampling.
    """
    batches = (1, 2, 4)
    heads = (1, 2, 4, 8, 16)
    sequences = (16, 32, 48, 63, 64, 65, 80, 96, 112, 127,
                 128, 129, 160, 192, 224, 256, 320, 384, 448, 512)
    head_dims = (64, 128)
    total = len(batches) * len(heads) * len(sequences) * len(sequences) * len(head_dims)
    if FASG_MULTI_TILING_RESERVE_SHAPES > total:
        raise ValueError("FASG multi-tiling reserve exceeds its reviewed shape lattice")
    # 119 and 12,000 are coprime: the walk visits distinct legal geometries
    # before cycling, and is deterministic across resumes and machines.
    if total != 12_000:
        raise ValueError("unexpected FASG reviewed lattice cardinality")
    output: list[dict[str, Any]] = []
    for ordinal in range(FASG_MULTI_TILING_RESERVE_SHAPES):
        code = (ordinal * 119) % total
        head_dim = head_dims[code % len(head_dims)]
        code //= len(head_dims)
        kv_seq = sequences[code % len(sequences)]
        code //= len(sequences)
        q_seq = sequences[code % len(sequences)]
        code //= len(sequences)
        q_heads = heads[code % len(heads)]
        code //= len(heads)
        batch = batches[code]
        output.append(row(
            "flash_attention_score_grad", start_index + ordinal,
            "source_multi_tiling,fp16,bnsd,reviewed_lattice,"
            "q_heads_equal_kv_heads,q_seq_lt1024,kv_seq_lt1024",
            dtype="fp16", layout="BNSD", batch=batch, q_heads=q_heads,
            kv_heads=q_heads, q_seq=q_seq, kv_seq=kv_seq, head_dim=head_dim,
            source_candidate_requirement="at_least_two_distinct_original_source_tilings",
        ))
    return output


def fused_attention_workloads() -> list[dict[str, Any]]:
    # Explicit decode and prefill cases. The source itself chooses IFA for its
    # original decode predicate and PFA otherwise; this catalog never forces one
    # branch onto a shape owned by the other.
    cases = [
        (1, 1, 1, 1, 64, 64, "fp16", "BNSD", "decode,small"),
        (1, 8, 8, 1, 128, 64, "fp16", "BNSD", "decode,mha"),
        (2, 16, 16, 1, 512, 128, "fp16", "BNSD", "decode,batch"),
        (1, 8, 2, 1, 1024, 128, "fp16", "BNSD", "decode,gqa"),
        (4, 16, 1, 1, 257, 64, "fp16", "BNSD", "decode,mqa,tail"),
        (1, 32, 4, 1, 2048, 64, "fp16", "BNSD", "decode,heads"),
        (1, 8, 8, 16, 64, 64, "fp16", "BNSD", "prefill,short"),
        (2, 16, 4, 32, 127, 128, "fp16", "BNSD", "prefill,gqa,tail"),
        (1, 8, 1, 64, 256, 128, "fp16", "BNSD", "prefill,mqa"),
        (2, 8, 2, 128, 512, 64, "fp16", "BNSD", "prefill,long_q"),
        (1, 16, 16, 257, 129, 64, "fp16", "BNSD", "prefill,dual_tail"),
        (1, 8, 2, 64, 1024, 128, "fp16", "BNSD", "prefill,long_kv"),
        (1, 8, 8, 1, 128, 64, "bf16", "BNSD", "bf16,decode"),
        (2, 16, 4, 1, 257, 128, "bf16", "BNSD", "bf16,gqa,tail"),
        (1, 8, 1, 1, 1024, 128, "bf16", "BNSD", "bf16,mqa,long_kv"),
        (4, 16, 16, 1, 512, 64, "bf16", "BNSD", "bf16,batch"),
        (1, 4, 4, 32, 32, 256, "fp16", "BNSD", "prefill,d256"),
        (1, 8, 2, 96, 320, 192, "fp16", "BNSD", "prefill,d192"),
        (2, 32, 8, 16, 2048, 64, "fp16", "BNSD", "prefill,heads"),
        (1, 16, 1, 384, 4096, 64, "fp16", "BNSD", "prefill,very_long_kv"),
        (8, 8, 8, 1, 64, 128, "fp16", "BNSD", "decode,large_batch"),
        (2, 16, 2, 63, 65, 64, "fp16", "BNSD", "prefill,tail"),
        (1, 8, 1, 256, 8192, 64, "fp16", "BNSD", "prefill,kv8192"),
        (1, 32, 4, 64, 512, 128, "fp16", "BNSD", "prefill,gqa"),
        (1, 8, 8, 1, 128, 64, "fp16", "BSND", "bsnd,decode"),
        (2, 16, 4, 32, 257, 128, "fp16", "BSND", "bsnd,prefill,gqa"),
        (1, 8, 1, 64, 1024, 64, "fp16", "BSND", "bsnd,prefill,mqa"),
        (1, 8, 2, 1, 513, 128, "bf16", "BSND", "bsnd,bf16,decode"),
        (1, 8, 8, 1, 128, 64, "fp16", "BSH", "bsh,decode"),
        (2, 16, 4, 32, 257, 128, "fp16", "BSH", "bsh,prefill,gqa"),
        (1, 8, 1, 64, 1024, 64, "fp16", "BSH", "bsh,prefill,mqa"),
        (1, 8, 2, 1, 513, 128, "bf16", "BSH", "bsh,bf16,decode"),
    ]
    extras = [
        (1, 4, 4, 1, 64, 64, "fp16", "BNSD", "decode,small4"),
        (1, 16, 4, 1, 384, 64, "fp16", "BNSD", "decode,gqa"),
        (2, 8, 1, 1, 1536, 128, "fp16", "BNSD", "decode,mqa,long_kv"),
        (1, 32, 32, 1, 128, 64, "fp16", "BNSD", "decode,many_heads"),
        (1, 8, 8, 8, 128, 64, "fp16", "BNSD", "prefill,seq8"),
        (1, 4, 2, 192, 384, 128, "fp16", "BNSD", "prefill,gqa"),
        (2, 16, 16, 384, 192, 64, "fp16", "BNSD", "prefill,long_q"),
        (1, 8, 1, 96, 768, 64, "fp16", "BNSD", "prefill,mqa"),
        (1, 8, 8, 256, 256, 128, "fp16", "BNSD", "prefill,square"),
        (1, 4, 4, 257, 513, 64, "fp16", "BNSD", "prefill,dual_tail"),
        (2, 32, 8, 64, 512, 64, "fp16", "BNSD", "prefill,heads"),
        (1, 4, 1, 1, 257, 192, "fp16", "BNSD", "decode,mqa,d192"),
        (1, 8, 8, 1, 64, 256, "fp16", "BNSD", "decode,d256"),
        (1, 4, 4, 128, 512, 64, "fp16", "BSND", "bsnd,prefill"),
        (1, 8, 2, 1, 1024, 64, "fp16", "BSND", "bsnd,decode,gqa"),
        (2, 8, 8, 64, 256, 128, "fp16", "BSH", "bsh,prefill"),
        (1, 8, 1, 1, 512, 64, "fp16", "BSH", "bsh,decode,mqa"),
        (1, 16, 4, 32, 1024, 128, "fp16", "BSH", "bsh,prefill,gqa"),
        (1, 4, 4, 1, 128, 64, "bf16", "BNSD", "bf16,decode"),
        (2, 8, 2, 1, 768, 128, "bf16", "BNSD", "bf16,decode,gqa"),
        (1, 16, 1, 1, 2048, 64, "bf16", "BNSD", "bf16,decode,mqa"),
        (4, 8, 8, 1, 256, 64, "bf16", "BNSD", "bf16,decode,batch"),
        (1, 8, 8, 1, 192, 64, "fp16", "BNSD", "decode,kv192"),
        (1, 8, 2, 64, 320, 128, "fp16", "BNSD", "prefill,kv320"),
        (2, 4, 4, 128, 128, 64, "fp16", "BNSD", "prefill,batch"),
        (1, 8, 4, 16, 512, 192, "fp16", "BNSD", "prefill,gqa,d192"),
        (1, 4, 4, 1, 65, 64, "fp16", "BNSD", "decode,tail"),
        (1, 4, 4, 65, 33, 64, "fp16", "BNSD", "prefill,inverse_tail"),
        (1, 8, 1, 1, 8192, 64, "fp16", "BNSD", "decode,kv8192"),
        (1, 16, 16, 512, 512, 128, "fp16", "BNSD", "prefill,long_square"),
        (2, 16, 4, 96, 384, 64, "fp16", "BSND", "bsnd,prefill,gqa"),
        (2, 16, 4, 96, 384, 64, "fp16", "BSH", "bsh,prefill,gqa"),
    ]
    return [row("fused_infer_attention_score", index, tags + "," + dtype + "," + layout.lower(),
                dtype=dtype, layout=layout, batch=batch, q_heads=q_heads, kv_heads=kv_heads,
                q_seq=q_seq, kv_seq=kv_seq, head_dim=head_dim)
            for index, (batch, q_heads, kv_heads, q_seq, kv_seq, head_dim, dtype, layout, tags)
            in enumerate(cases + extras)]


def index_workloads(op: str) -> list[dict[str, Any]]:
    cases = [
        ([64], 0, [17], "rank1,index_tail"),
        ([31, 65], 0, [17, 65], "rank2,first_axis"),
        ([31, 65], 1, [31, 19], "rank2,last_axis"),
        ([8, 64, 128], 0, [3, 64, 128], "rank3,first_axis"),
        ([8, 64, 128], 1, [8, 17, 128], "rank3,middle_axis"),
        ([7, 33, 65], 2, [7, 33, 19], "rank3,last_axis,tail"),
        ([4, 17, 33, 65], 1, [4, 7, 33, 65], "rank4,inner_axis"),
        ([2, 64, 128, 256], 3, [2, 64, 128, 63], "rank4,last_axis,large"),
        ([1, 4097, 63], 1, [1, 257, 63], "large_axis,long_index"),
        ([16, 32, 64], -1, [16, 32, 1], "negative_axis,single_axis"),
    ]
    modes = (("fp16", "int32"), ("bf16", "int64"), ("fp32", "int32"), ("int32", "int64"))
    output: list[dict[str, Any]] = []
    for case_index, (shape, axis, index_shape, tags) in enumerate(cases):
        rotated = modes[case_index % len(modes):] + modes[:case_index % len(modes)]
        for dtype, index_dtype in rotated:
            extra = "reduce_assign" if op == "scatter_elements" else ""
            output.append(row(op, len(output), ",".join(value for value in (tags, dtype, index_dtype, extra) if value),
                              dtype=dtype, index_dtype=index_dtype, shape=shape, axis=axis,
                              index_shape=index_shape, **({"reduce": 0} if op == "scatter_elements" else {})))
    return output


def catalog() -> list[dict[str, Any]]:
    attention = attention_grad_workloads()
    output = attention + attention_grad_multi_tiling_reserve(len(attention))
    output += fused_attention_workloads()
    output += index_workloads("gather_elements")
    output += index_workloads("scatter_elements")
    validate(output)
    return output


def native_attempts(workload: dict[str, Any]) -> int:
    if workload["op"] == "flash_attention_score_grad":
        # The registry has eight original strategies (reported separately),
        # but only these three have overlapping source capability for the
        # controlled BNSD/fp16 comparison lane.  Calling a strategy that the
        # original IsCapable predicate rules out is neither a viable tiling nor
        # useful data, so it is not launched for every shape.
        return len(FASG_MULTI_TILING_SOURCE_CLASSES)
    return 1


def validate(workloads: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for item in workloads:
        if item["workload_id"] in ids:
            raise ValueError("duplicate workload id: {}".format(item["workload_id"]))
        ids.add(item["workload_id"])
        if item["op"] in ("flash_attention_score_grad", "fused_infer_attention_score"):
            if item["q_heads"] % item["kv_heads"] or item["head_dim"] % 16:
                raise ValueError("illegal attention geometry: {}".format(item["workload_id"]))
        else:
            rank = len(item["shape"])
            axis = item["axis"] + rank if item["axis"] < 0 else item["axis"]
            if axis < 0 or axis >= rank or len(item["index_shape"]) != rank:
                raise ValueError("illegal index geometry: {}".format(item["workload_id"]))
            if any(item["index_shape"][i] > item["shape"][i] for i in range(rank) if i != axis):
                raise ValueError("illegal non-axis index shape: {}".format(item["workload_id"]))


def audit(workloads: list[dict[str, Any]]) -> dict[str, Any]:
    per_op = Counter(item["op"] for item in workloads)
    attempts = Counter()
    for item in workloads:
        attempts[item["op"]] += native_attempts(item)
    tags: dict[str, Counter[str]] = defaultdict(Counter)
    for item in workloads:
        tags[item["op"]].update(item["coverage"])
    return {
        "schema": "non_matmul_source_candidate_catalog_v1",
        "generation": "reviewed_explicit_families_plus_fixed_legal_shape_lattice_no_random_no_tile_enumeration",
        "matmul_included": False,
        "semantic_workloads": len(workloads),
        "workloads_per_op": dict(sorted(per_op.items())),
        "source_discovery_upper_bound": sum(attempts.values()),
        "full_fasg_original_strategy_registry_count": FASG_NATIVE_STRATEGIES,
        "attempts_per_op": dict(sorted(attempts.items())),
        "formal_latency_target": FORMAL_LATENCY_TARGET,
        "formal_latency_count_rule": "count only output-validated executions; FlashAttentionScoreGrad requires at least two distinct original raw tilings for one semantic shape",
        "fasg_multi_tiling_source_classes": list(FASG_MULTI_TILING_SOURCE_CLASSES),
        "fasg_multi_tiling_reserve_shapes": FASG_MULTI_TILING_RESERVE_SHAPES,
        "blocked_without_matching_910b_source": ["transpose", "gather_v2"],
        "coverage_tags_per_op": {name: dict(sorted(counter.items())) for name, counter in sorted(tags.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit workload rows as JSONL")
    args = parser.parse_args()
    rows = catalog()
    if args.audit:
        print(json.dumps(audit(rows), ensure_ascii=False, sort_keys=True))
    if args.json:
        for item in rows:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))
    if not args.audit and not args.json:
        parser.error("choose --audit and/or --json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
