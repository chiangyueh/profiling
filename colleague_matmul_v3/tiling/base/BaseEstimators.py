from __future__ import annotations

from .Base import BaseAlgo, BaseParam
from pathlib import Path
import os
import shutil
import subprocess
import json
import pandas as pd


class BaseAlgoMsprof(BaseAlgo):
    def _run_estimator(self, params: list[BaseParam]) -> float:
        env = dict(os.environ)
        for param in params:
            env[param.name] = str(param.value)
        for d in Path(".").glob("OPPROF_*"):
            shutil.rmtree(d)
        subprocess.run(["bash", self.runner, "-r", "sim"], env=env,
                               capture_output=True, text=True)
        subprocess.run(["bash", self.runner, "-r", "cpu"], env=env,
                       capture_output=True, text=True)

        dur = self._get_time()
        return dur

    def _get_time(self) -> float:
        try:
            opprofs = list(Path('.').glob("OPPROF_*"))
            if not opprofs:
                return float("inf")
            opprof = max(opprofs, key=lambda d: d.stat().st_mtime)
            traces = list(opprof.rglob("trace.json"))
            latency = 0.0
            found = False
            for tj in traces:
                with open(tj) as f:
                    events = json.load(f).get("traceEvents", [])
                ts = [e["ts"] for e in events if "ts" in e]
                if not ts:
                    continue
                span = max(e["ts"] + e.get("dur", 0) for e in events if "ts" in e) - min(ts)
                latency = max(latency, span)
                found = True
            if not found:
                return float("inf")
        except Exception:
            latency = float("inf")
        return latency


class BaseAlgoReal(BaseAlgo):
    def _run_estimator(self, params: list[BaseParam]) -> float:
        env = dict(os.environ)
        for param in params:
            env[param.name] = str(param.value)

        cceprint = Path("cceprint")
        if cceprint.exists():
            for f in cceprint.glob("*.cce"):
                f.unlink()
        for d in Path(".").glob("OPPROF_*"):
            shutil.rmtree(d)
        output_path = Path("output/output.bin")
        output_path.unlink(missing_ok=True)
        completed = subprocess.run(
            ["bash", self.runner, "-r", "npu", "-v", "Ascend910B3"],
            env=env,
            capture_output=True,
            text=True,
        )
        self._last_run = {
            "status": "passed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }

        if completed.returncode != 0:
            return float("inf")

        dur = self._get_time()
        if dur == float("inf"):
            self._last_run["status"] = "timing_missing"
        return dur
    
    def _get_time(self) -> float:
        try:
            opprofs = list(Path('.').glob("OPPROF_*"))
            if not opprofs:
                return float("inf")
            opprof = max(opprofs, key=lambda d: d.stat().st_mtime)
            latency = pd.read_csv(f"{opprof}/OpBasicInfo.csv")['Task Duration(us)'].iloc[0]
        except Exception:
            latency = float("inf")
        return latency
