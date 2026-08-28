from __future__ import annotations

from .Base import BaseAlgo, BaseParam
from pathlib import Path
import math
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
        max_attempts = max(1, int(os.environ.get("MATMUL_AUDIT_PROFILE_ATTEMPTS", "2")))
        history = []
        for attempt in range(1, max_attempts + 1):
            cceprint = Path("cceprint")
            if cceprint.exists():
                for f in cceprint.glob("*.cce"):
                    f.unlink()
            for d in Path(".").glob("OPPROF_*"):
                shutil.rmtree(d)
            Path("output/output.bin").unlink(missing_ok=True)
            completed = subprocess.run(
                ["bash", self.runner, "-r", "npu", "-v", "Ascend910B3"],
                env=env,
                capture_output=True,
                text=True,
            )
            attempt_status = "failed" if completed.returncode else "timing_missing"
            duration = float("inf")
            if completed.returncode == 0:
                duration = self._get_time()
                if math.isfinite(duration):
                    attempt_status = "passed"
            history.append({
                "attempt": attempt,
                "status": attempt_status,
                "return_code": completed.returncode,
            })
            self._last_run = {
                "status": attempt_status,
                "return_code": completed.returncode,
                "profile_attempts": history,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
            if attempt_status == "passed":
                return duration
        return float("inf")
    
    def _get_time(self) -> float:
        try:
            opprofs = list(Path('.').glob("OPPROF_*"))
            if not opprofs:
                return float("inf")
            opprof = max(opprofs, key=lambda d: d.stat().st_mtime)
            latency = float(pd.read_csv(f"{opprof}/OpBasicInfo.csv")['Task Duration(us)'].iloc[0])
            if not math.isfinite(latency):
                return float("inf")
        except Exception:
            latency = float("inf")
        return latency
