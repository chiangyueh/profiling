#!/usr/bin/env python3
import sys

import numpy as np


def main() -> int:
    output = np.fromfile(sys.argv[1], dtype=np.float16)
    golden = np.fromfile(sys.argv[2], dtype=np.float16)
    if output.size != golden.size:
        print(f"wrong output size: output={output.size} golden={golden.size}")
        return 1

    bad = np.flatnonzero(~np.isclose(output, golden, rtol=1e-3, atol=1e-3, equal_nan=True))
    if bad.size:
        first = int(bad[0])
        print(
            f"wrong output: index={first} expected={float(golden[first])} "
            f"actual={float(output[first])} mismatches={bad.size}/{golden.size}"
        )
        return 1
    print("test pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
