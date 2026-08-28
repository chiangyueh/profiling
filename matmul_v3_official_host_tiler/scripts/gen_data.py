#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np


def main() -> None:
    m = int(os.environ.get("MM_M", 512))
    n = int(os.environ.get("MM_N", 512))
    k = int(os.environ.get("MM_K", 512))
    print(f"[gen_data] M={m} N={n} K={k}")

    x1 = np.random.randint(1, 10, (m, k)).astype(np.float16)
    x2 = np.random.randint(1, 10, (k, n)).astype(np.float16)
    golden = np.matmul(x1.astype(np.float32), x2.astype(np.float32)).astype(np.float16)

    Path("input").mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(parents=True, exist_ok=True)
    x1.tofile("input/x1_gm.bin")
    x2.tofile("input/x2_gm.bin")
    golden.tofile("output/golden.bin")


if __name__ == "__main__":
    main()
