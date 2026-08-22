#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict


OPS = (
    "matmul",
    "transpose",
    "gather_v2",
    "gather_elements",
    "scatter_elements",
    "flash_attention_score_grad",
    "fused_infer_attention_score",
)


def product(shape):
    value = 1
    for dim in shape:
        value *= dim
    return value


def row(op, index, tags, **parameters):
    return {
        "workload_id": f"{op}_{index:03d}",
        "op": op,
        "preflight": index == 0,
        "coverage": tags.split(","),
        **parameters,
    }


def matmul_workloads():
    geometries = [
        (16, 16, 16, "single_cube,aligned"),
        (31, 47, 65, "all_axis_tail,small"),
        (64, 64, 256, "aligned,k_deep"),
        (128, 256, 512, "base_128x256,aligned"),
        (256, 128, 512, "base_256x128,aligned"),
        (32, 384, 4096, "skinny_m,k_deep"),
        (384, 32, 4096, "skinny_n,k_deep"),
        (16, 1024, 8192, "underfilled_m,wide_n,k_deep"),
        (1024, 16, 8192, "underfilled_n,tall_m,k_deep"),
        (160, 224, 1536, "multi_core_tail,mixed_alignment"),
        (511, 383, 4097, "large_tail,k_tail"),
        (1024, 1024, 256, "many_output_tiles,k_shallow"),
    ]
    modes = [
        ("fp16", 0, 0, "fp16,nn"),
        ("bf16", 0, 1, "bf16,nt"),
        ("fp32", 1, 0, "fp32,tn"),
        ("fp16", 1, 1, "fp16,tt"),
    ]
    result = []
    for geometry_index, (m, n, k, geometry_tags) in enumerate(geometries):
        rotated = modes[geometry_index % len(modes):] + modes[:geometry_index % len(modes)]
        for dtype, trans_a, trans_b, mode_tags in rotated:
            result.append(row(
                "matmul", len(result), f"{geometry_tags},{mode_tags}",
                dtype=dtype, m=m, n=n, k=k, trans_a=trans_a, trans_b=trans_b,
            ))
    return result


def transpose_workloads():
    cases = [
        ([4, 2], [1, 0], "rank2,official_preflight,axis_swap"),
        ([31, 65], [1, 0], "rank2,tail,axis_swap"),
        ([4, 64, 128], [0, 2, 1], "rank3,aligned,last_axis_swap"),
        ([3, 31, 65], [2, 0, 1], "rank3,tail,cyclic"),
        ([2, 17, 33, 65], [0, 2, 3, 1], "rank4,tail,nchw_to_nhwc"),
        ([2, 17, 33, 65], [0, 3, 1, 2], "rank4,tail,nhwc_to_nchw"),
        ([4, 32, 64, 128], [3, 2, 1, 0], "rank4,aligned,reverse"),
        ([2, 3, 17, 33, 65], [0, 2, 4, 1, 3], "rank5,tail,multi_axis"),
        ([1, 4097, 63], [0, 2, 1], "rank3,large_axis,tail"),
        ([8, 16, 32, 64], [0, 1, 2, 3], "rank4,identity,tensor_move"),
    ]
    dtypes = ("fp32", "fp16", "bf16", "int8")
    result = []
    for case_index, (shape, perm, tags) in enumerate(cases):
        for dtype in dtypes[case_index % 4:] + dtypes[:case_index % 4]:
            result.append(row(
                "transpose", len(result), f"{tags},{dtype}",
                dtype=dtype, shape=shape, perm=perm,
            ))
    return result


def gather_v2_workloads():
    cases = [
        ([64], 0, [17], "rank1,index_tail"),
        ([31, 65], 0, [7], "rank2,first_axis,tail"),
        ([31, 65], 1, [9], "rank2,last_axis,tail"),
        ([8, 64, 128], 1, [17], "rank3,middle_axis,aligned"),
        ([7, 33, 65], 2, [3, 5], "rank3,last_axis,index_rank2"),
        ([4, 17, 33, 65], 0, [5, 3], "rank4,first_axis,index_rank2"),
        ([4, 17, 33, 65], 2, [31], "rank4,inner_axis,tail"),
        ([2, 64, 128, 256], 3, [63], "rank4,last_axis,large_aligned"),
        ([1, 4097, 63], 1, [257], "large_axis,long_index,tail"),
        ([16, 32, 64], -1, [1], "negative_axis,single_index"),
    ]
    modes = (
        ("fp16", "int32"),
        ("bf16", "int64"),
        ("fp32", "int32"),
        ("int32", "int64"),
    )
    result = []
    for case_index, (shape, axis, index_shape, tags) in enumerate(cases):
        rotated = modes[case_index % 4:] + modes[:case_index % 4]
        for dtype, index_dtype in rotated:
            result.append(row(
                "gather_v2", len(result), f"{tags},{dtype},{index_dtype}",
                dtype=dtype, index_dtype=index_dtype, shape=shape,
                axis=axis, index_shape=index_shape,
            ))
    return result


def gather_elements_workloads():
    cases = [
        ([64], 0, [17], "rank1,index_tail"),
        ([31, 65], 0, [17, 65], "rank2,first_axis,partial_axis"),
        ([31, 65], 1, [31, 19], "rank2,last_axis,partial_axis"),
        ([8, 64, 128], 0, [3, 64, 128], "rank3,first_axis,aligned"),
        ([8, 64, 128], 1, [8, 17, 128], "rank3,middle_axis,aligned"),
        ([7, 33, 65], 2, [7, 33, 19], "rank3,last_axis,tail"),
        ([4, 17, 33, 65], 1, [4, 7, 33, 65], "rank4,inner_axis,tail"),
        ([2, 64, 128, 256], 3, [2, 64, 128, 63], "rank4,last_axis,large"),
        ([1, 4097, 63], 1, [1, 257, 63], "large_axis,long_index"),
        ([16, 32, 64], -1, [16, 32, 1], "negative_axis,single_axis_value"),
    ]
    modes = (
        ("fp16", "int32"),
        ("bf16", "int64"),
        ("fp32", "int32"),
        ("int32", "int64"),
    )
    result = []
    for case_index, (shape, axis, index_shape, tags) in enumerate(cases):
        rotated = modes[case_index % 4:] + modes[:case_index % 4]
        for dtype, index_dtype in rotated:
            result.append(row(
                "gather_elements", len(result), f"{tags},{dtype},{index_dtype}",
                dtype=dtype, index_dtype=index_dtype, shape=shape,
                axis=axis, index_shape=index_shape,
            ))
    return result


def scatter_elements_workloads():
    cases = [
        ([64], 0, [17], "rank1,index_tail"),
        ([31, 65], 0, [17, 65], "rank2,first_axis"),
        ([31, 65], 1, [31, 19], "rank2,last_axis"),
        ([8, 64, 128], 0, [3, 64, 128], "rank3,first_axis,aligned"),
        ([8, 64, 128], 1, [8, 17, 128], "rank3,middle_axis,aligned"),
        ([7, 33, 65], 2, [7, 33, 19], "rank3,last_axis,tail"),
        ([4, 17, 33, 65], 1, [4, 7, 33, 65], "rank4,inner_axis,tail"),
        ([2, 64, 128, 256], 3, [2, 64, 128, 63], "rank4,last_axis,large"),
        ([1, 4097, 63], 1, [1, 257, 63], "large_axis,long_update"),
        ([16, 32, 64], -1, [16, 32, 1], "negative_axis,single_axis_value"),
    ]
    modes = (
        ("fp16", "int32", 0, "assign"),
        ("fp32", "int64", 1, "add"),
        ("int32", "int32", 2, "multiply"),
        ("fp16", "int64", 1, "add"),
    )
    result = []
    for case_index, (shape, axis, index_shape, tags) in enumerate(cases):
        rotated = modes[case_index % 4:] + modes[:case_index % 4]
        for dtype, index_dtype, reduce, reduce_tag in rotated:
            result.append(row(
                "scatter_elements", len(result),
                f"{tags},{dtype},{index_dtype},reduce_{reduce_tag}",
                dtype=dtype, index_dtype=index_dtype, shape=shape,
                axis=axis, index_shape=index_shape, reduce=reduce,
            ))
    return result


def flash_attention_score_grad_workloads():
    cases = [
        (1, 1, 64, 64, 64, "fp16", "BNSD", "small,aligned"),
        (1, 4, 128, 128, 64, "fp16", "BNSD", "multi_head,aligned"),
        (2, 8, 256, 256, 128, "fp16", "BNSD", "multi_batch,multi_head"),
        (1, 8, 127, 193, 128, "fp16", "BNSD", "q_tail,kv_tail,asymmetric_seq"),
        (2, 4, 257, 129, 64, "fp16", "BNSD", "q_tail,kv_tail,asymmetric_seq"),
        (4, 2, 64, 512, 128, "fp16", "BNSD", "multi_batch,long_kv"),
        (1, 16, 512, 512, 64, "fp16", "BNSD", "many_heads,long_seq"),
        (2, 8, 128, 1024, 64, "fp16", "BNSD", "long_kv,asymmetric_seq"),
        (1, 1, 64, 64, 128, "fp32", "BNSD", "fp32,aligned"),
        (1, 4, 129, 257, 64, "fp32", "BNSD", "fp32,tail,asymmetric_seq"),
        (2, 8, 256, 256, 128, "fp32", "BNSD", "fp32,multi_batch"),
        (1, 4, 128, 128, 64, "bf16", "BNSD", "bf16,aligned"),
        (1, 1, 64, 64, 64, "fp16", "SBH", "sbh,small"),
        (1, 4, 128, 128, 64, "fp16", "SBH", "sbh,multi_head"),
        (2, 8, 257, 129, 128, "fp16", "SBH", "sbh,multi_batch,tail"),
        (1, 4, 128, 512, 128, "fp16", "SBH", "sbh,long_kv"),
        (1, 1, 64, 64, 128, "fp32", "SBH", "sbh,fp32"),
        (2, 4, 129, 257, 64, "fp32", "SBH", "sbh,fp32,tail"),
        (1, 8, 384, 384, 64, "fp16", "BNSD", "sequence_384"),
        (2, 16, 96, 320, 64, "fp16", "BNSD", "head_parallelism,asymmetric_seq"),
        (4, 4, 192, 192, 128, "fp16", "BNSD", "batch_parallelism"),
        (1, 2, 33, 65, 192, "fp16", "BNSD", "head_dim_192,tail"),
        (1, 8, 1024, 128, 64, "fp16", "BNSD", "long_q,short_kv"),
        (2, 1, 63, 1025, 128, "fp16", "BNSD", "long_kv_boundary"),
    ]
    result = []
    for index, (batch, heads, q_seq, kv_seq, head_dim, dtype, layout, tags) in enumerate(cases):
        result.append(row(
            "flash_attention_score_grad", index, f"{tags},{dtype},{layout.lower()}",
            dtype=dtype, layout=layout, batch=batch, q_heads=heads, kv_heads=heads,
            q_seq=q_seq, kv_seq=kv_seq, head_dim=head_dim,
        ))
    gqa_and_layout_cases = [
        (1, 8, 2, 128, 256, 64, "fp16", "BNSD", "gqa,asymmetric_heads"),
        (2, 16, 1, 257, 129, 128, "fp16", "BNSD", "mqa,tail,asymmetric_heads"),
        (1, 4, 4, 64, 64, 64, "fp16", "BSND", "bsnd,aligned"),
        (2, 8, 2, 129, 257, 128, "fp16", "BSND", "bsnd,gqa,tail"),
        (1, 8, 8, 512, 128, 64, "fp32", "BSND", "bsnd,fp32,long_q"),
        (1, 4, 4, 64, 64, 128, "fp16", "BSH", "bsh,aligned"),
        (2, 8, 2, 257, 129, 64, "fp16", "BSH", "bsh,gqa,tail"),
        (1, 16, 1, 128, 1024, 64, "fp16", "BSH", "bsh,mqa,long_kv"),
    ]
    for batch, q_heads, kv_heads, q_seq, kv_seq, head_dim, dtype, layout, tags in gqa_and_layout_cases:
        result.append(row(
            "flash_attention_score_grad", len(result), f"{tags},{dtype},{layout.lower()}",
            dtype=dtype, layout=layout, batch=batch, q_heads=q_heads, kv_heads=kv_heads,
            q_seq=q_seq, kv_seq=kv_seq, head_dim=head_dim,
        ))
    return result


def fused_infer_attention_score_workloads():
    cases = [
        (1, 1, 1, 1, 64, 64, "fp16", "mha,decode,small"),
        (1, 8, 8, 1, 128, 64, "fp16", "mha,decode"),
        (2, 16, 16, 1, 512, 128, "fp16", "mha,multi_batch,decode"),
        (1, 8, 2, 1, 1024, 128, "fp16", "gqa,decode,long_kv"),
        (4, 16, 1, 1, 257, 64, "fp16", "mqa,multi_batch,kv_tail"),
        (1, 32, 4, 1, 2048, 64, "fp16", "gqa,many_heads,long_kv"),
        (1, 8, 8, 16, 64, 64, "fp16", "mha,prefill_short"),
        (2, 16, 4, 32, 127, 128, "fp16", "gqa,prefill,kv_tail"),
        (1, 8, 1, 64, 256, 128, "fp16", "mqa,prefill"),
        (2, 8, 2, 128, 512, 64, "fp16", "gqa,prefill,long_q"),
        (1, 16, 16, 257, 129, 64, "fp16", "mha,q_tail,kv_tail"),
        (1, 8, 2, 64, 1024, 128, "fp16", "gqa,prefill,long_kv"),
        (1, 8, 8, 1, 128, 64, "bf16", "bf16,mha,decode"),
        (2, 16, 4, 1, 257, 128, "bf16", "bf16,gqa,decode,kv_tail"),
        (1, 8, 1, 1, 1024, 128, "bf16", "bf16,mqa,decode,long_kv"),
        (4, 16, 16, 1, 512, 64, "bf16", "bf16,mha,multi_batch"),
        (1, 4, 4, 32, 32, 256, "fp16", "large_head_dim,prefill"),
        (1, 8, 2, 96, 320, 192, "fp16", "head_dim_192,gqa"),
        (2, 32, 8, 16, 2048, 64, "fp16", "many_heads,long_kv"),
        (1, 16, 1, 384, 4096, 64, "fp16", "mqa,very_long_kv"),
        (8, 8, 8, 1, 64, 128, "fp16", "large_batch,decode"),
        (2, 16, 2, 63, 65, 64, "fp16", "gqa,all_seq_tail"),
        (1, 8, 1, 256, 8192, 64, "fp16", "mqa,kv_8192"),
        (1, 32, 4, 64, 512, 128, "fp16", "gqa,many_heads,prefill"),
    ]
    result = []
    for index, (batch, q_heads, kv_heads, q_seq, kv_seq, head_dim, dtype, tags) in enumerate(cases):
        result.append(row(
            "fused_infer_attention_score", index, f"{tags},{dtype},bnsd",
            dtype=dtype, layout="BNSD", batch=batch, q_heads=q_heads,
            kv_heads=kv_heads, q_seq=q_seq, kv_seq=kv_seq,
            head_dim=head_dim,
        ))
    additional_layout_cases = [
        (1, 8, 8, 1, 128, 64, "fp16", "BSND", "bsnd,mha,decode"),
        (2, 16, 4, 32, 257, 128, "fp16", "BSND", "bsnd,gqa,prefill,tail"),
        (1, 8, 1, 64, 1024, 64, "fp16", "BSND", "bsnd,mqa,long_kv"),
        (1, 8, 2, 1, 513, 128, "bf16", "BSND", "bsnd,bf16,gqa,decode"),
        (1, 8, 8, 1, 128, 64, "fp16", "BSH", "bsh,mha,decode"),
        (2, 16, 4, 32, 257, 128, "fp16", "BSH", "bsh,gqa,prefill,tail"),
        (1, 8, 1, 64, 1024, 64, "fp16", "BSH", "bsh,mqa,long_kv"),
        (1, 8, 2, 1, 513, 128, "bf16", "BSH", "bsh,bf16,gqa,decode"),
    ]
    for batch, q_heads, kv_heads, q_seq, kv_seq, head_dim, dtype, layout, tags in additional_layout_cases:
        result.append(row(
            "fused_infer_attention_score", len(result), f"{tags},{dtype},{layout.lower()}",
            dtype=dtype, layout=layout, batch=batch, q_heads=q_heads,
            kv_heads=kv_heads, q_seq=q_seq, kv_seq=kv_seq, head_dim=head_dim,
        ))
    return result


def catalog():
    workloads = []
    workloads.extend(matmul_workloads())
    workloads.extend(transpose_workloads())
    workloads.extend(gather_v2_workloads())
    workloads.extend(gather_elements_workloads())
    workloads.extend(scatter_elements_workloads())
    workloads.extend(flash_attention_score_grad_workloads())
    workloads.extend(fused_infer_attention_score_workloads())
    validate(workloads)
    return workloads


def normalized_axis(axis, rank):
    return axis + rank if axis < 0 else axis


def validate(workloads):
    ids = set()
    preflight_counts = Counter()
    for workload in workloads:
        workload_id = workload["workload_id"]
        if workload_id in ids:
            raise ValueError(f"duplicate workload_id: {workload_id}")
        ids.add(workload_id)
        op = workload["op"]
        if op not in OPS:
            raise ValueError(f"unknown op: {op}")
        if workload["preflight"]:
            preflight_counts[op] += 1
        if not workload["coverage"] or any(not tag for tag in workload["coverage"]):
            raise ValueError(f"missing coverage tag: {workload_id}")
        if op == "matmul":
            if min(workload["m"], workload["n"], workload["k"]) <= 0:
                raise ValueError(f"illegal matmul dimension: {workload_id}")
        elif op == "transpose":
            shape, perm = workload["shape"], workload["perm"]
            if sorted(perm) != list(range(len(shape))):
                raise ValueError(f"illegal permutation: {workload_id}")
        elif op in ("gather_v2", "gather_elements", "scatter_elements"):
            rank = len(workload["shape"])
            axis = normalized_axis(workload["axis"], rank)
            if not 0 <= axis < rank:
                raise ValueError(f"illegal axis: {workload_id}")
            if op != "gather_v2":
                index_shape = workload["index_shape"]
                if len(index_shape) != rank:
                    raise ValueError(f"rank mismatch: {workload_id}")
                if any(index_shape[i] > workload["shape"][i] for i in range(rank) if i != axis):
                    raise ValueError(f"illegal non-axis index extent: {workload_id}")
        elif op == "flash_attention_score_grad":
            if workload["q_heads"] % workload["kv_heads"]:
                raise ValueError(f"q_heads not divisible by kv_heads: {workload_id}")
            if workload["head_dim"] % 16 or workload["layout"] not in ("BNSD", "BSND", "BSH", "SBH"):
                raise ValueError(f"illegal attention-grad geometry: {workload_id}")
        elif op == "fused_infer_attention_score":
            if workload["q_heads"] % workload["kv_heads"]:
                raise ValueError(f"q_heads not divisible by kv_heads: {workload_id}")
            if workload["head_dim"] % 16 or workload["head_dim"] > 512:
                raise ValueError(f"illegal head_dim: {workload_id}")
            if workload["dtype"] == "bf16" and workload["q_seq"] != 1:
                raise ValueError(f"bf16 prefill is excluded for CANN 8.1: {workload_id}")
        elements = []
        if "shape" in workload:
            elements.append(product(workload["shape"]))
        if elements and max(elements) > 134217728:
            raise ValueError(f"tensor exceeds campaign bound: {workload_id}")
    if set(preflight_counts) != set(OPS) or any(preflight_counts[op] != 1 for op in OPS):
        raise ValueError(f"exactly one preflight is required per op: {preflight_counts}")


def audit(workloads):
    counts = Counter(item["op"] for item in workloads)
    tags = defaultdict(Counter)
    for item in workloads:
        tags[item["op"]].update(item["coverage"])
    return {
        "schema": "multi_op_explicit_workloads_v1",
        "generation": "manually_reviewed_deterministic_coverage_no_random",
        "total_workloads": len(workloads),
        "workloads_per_op": dict(sorted(counts.items())),
        "preflight_per_op": dict(sorted(Counter(
            item["op"] for item in workloads if item["preflight"]
        ).items())),
        "coverage_tags_per_op": {
            op: dict(sorted(tag_counts.items())) for op, tag_counts in sorted(tags.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    workloads = catalog()
    if args.audit:
        print(json.dumps(audit(workloads), ensure_ascii=False, sort_keys=True))
        return
    for workload in workloads:
        if not args.preflight or workload["preflight"]:
            print(json.dumps(workload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
