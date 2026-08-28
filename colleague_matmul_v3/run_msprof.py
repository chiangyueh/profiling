import os
import subprocess
import sys
import pandas as pd
from pathlib import Path

CSV = "configs.csv"
RUNNER = "./run.sh"
RUN_MODE = "npu"
SOC = "Ascend910B3"

def get_time() -> float:
    try:
        opprofs = list(Path('.').glob("OPPROF_*"))
        if not opprofs:
            return float("inf")
        opprof = max(opprofs, key=lambda d: d.stat().st_mtime)
        latency = pd.read_csv(f"{opprof}/OpBasicInfo.csv")['Task Duration(us)'].iloc[0]
    except Exception:
        latency = float("inf")
    return latency

if not os.path.isfile(CSV):
    print(f"CSV не найден: '{CSV}'")
    sys.exit(1)

results = []
df = pd.read_csv(CSV)
for idx, row in df.iterrows():
    form = int(row["form"])
    env = dict(os.environ)
    env.update({k: str(int(v)) for k, v in {
        "MM_M": form, "MM_N": form, "MM_K": form,
        "MM_BASE_M": row["MM_BASE_M"], "MM_BASE_N": row["MM_BASE_N"], "MM_BASE_K": row["MM_BASE_K"],
        "MM_SINGLE_M": row["MM_SINGLE_M"], "MM_SINGLE_N": row["MM_SINGLE_N"],
        "MM_STEP_M": row["MM_STEP_M"], "MM_STEP_N": row["MM_STEP_N"],
        "MM_STEP_Ka": row["MM_STEP_Ka"], "MM_STEP_Kb": row["MM_STEP_Kb"],
        "MM_ITER_ORDER": 0, "MM_OP_TILING": 0,
    }.items()})
    print("=" * 60)
    print(f"[{idx}] form={form}^3 cores={int(row['usedCoreNum'])} | "
          f"baseK={int(row['MM_BASE_K'])} singleM={int(row['MM_SINGLE_M'])} singleN={int(row['MM_SINGLE_N'])} "
          f"stepKa={int(row['MM_STEP_Ka'])} stepKb={int(row['MM_STEP_Kb'])} | cache_duration={row['duration']}")
    print("-" * 60)
    subprocess.run(["bash", RUNNER, "-r", RUN_MODE, "-v", SOC], env=env)
    results.append(get_time())

df['duration real'] = results
df.to_csv('results.csv', index=False)