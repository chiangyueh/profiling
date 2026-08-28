#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    # +++ BEGIN: official Gitee MatMulV3 emits FP16; colleague direct kernel emits FP32.
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    # +++ END: route-specific output dtype.
    args = parser.parse_args()

    record = {"variant": args.variant}
    if not args.output.is_file():
        record.update(status="missing_output")
        print(json.dumps(record, separators=(",", ":")))
        return 1

    # +++ BEGIN: decode each route using its real output contract.
    dtype = np.float16 if args.dtype == "fp16" else np.float32
    output = np.fromfile(args.output, dtype=dtype)
    golden = np.fromfile(args.golden, dtype=dtype)
    # +++ END: route-specific output decoding.
    if output.size != golden.size:
        record.update(
            status="wrong_output_size",
            output_elements=int(output.size),
            golden_elements=int(golden.size),
        )
        print(json.dumps(record, separators=(",", ":")))
        return 1

    close = np.isclose(output, golden, rtol=1e-6, atol=1e-9, equal_nan=True)
    bad = np.flatnonzero(~close)
    record["elements"] = int(golden.size)
    record["mismatches"] = int(bad.size)
    record["error_ratio"] = float(bad.size / golden.size)
    if bad.size:
        index = int(bad[0])
        record.update(
            status="wrong_output",
            first_error={
                "index": index,
                "expected": float(golden[index]),
                "actual": float(output[index]),
            },
        )
    else:
        record["status"] = "passed"
    print(json.dumps(record, separators=(",", ":")))
    return 0 if bad.size == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
