#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import argparse
import numpy as np

# for float32
relative_tol = 1e-6
absolute_tol = 1e-9
error_tol = 1e-4


def verify_result(output: str, golden: str, dtype_name: str = "fp32") -> bool:
    # +++ BEGIN: official CANN MatMulV3 returns FP16; the retained colleague
    # direct-launch program returns FP32.
    dtype = np.float16 if dtype_name == "fp16" else np.float32
    # +++ END: route-specific output contract.
    output = np.fromfile(output, dtype=dtype).reshape(-1)
    golden = np.fromfile(golden, dtype=dtype).reshape(-1)
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data))
        if index == 100:
            break
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == '__main__':
    # +++ BEGIN: optional dtype flag; no flag preserves the colleague verifier's
    # original FP32 behavior used by 4.log.
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("golden")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    args = parser.parse_args()
    # +++ END: route-specific verifier input.
    try:
        res = verify_result(args.output, args.golden, args.dtype)
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass")
    except Exception as e:
        print(e)
        sys.exit(1)
