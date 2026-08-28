from __future__ import annotations

from typing import Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import os
import numpy as np


@dataclass
class BaseParam:
    name: str
    value: int
    is_const: bool
    domain: list = field(default_factory=list)
    index: int = 0

    def __post_init__(self):
        if self.is_const and not self.domain:
            self.domain = [self.value]
        self.index = self.domain.index(self.value)

    def __repr__(self) -> str:
        return f"name={self.name}, value={self.value}, is_const={self.is_const}"
    
    def update(self, index: int) -> None:
        self.index = index
        self.value = self.domain[index]


@dataclass
class BaseResult:
    duration: float
    params: list[BaseParam]

    def __repr__(self) -> str:
        return f"duration={self.duration}\nparams={[f'{param.name}={param.value}' for param in self.params]}\n"


class BaseValidator:
    def is_valid(self, params: list[BaseParam]) -> bool:
        raise NotImplementedError

    def get_combinations(self, num: int, const_params: list[BaseParam]) -> list[list[BaseParam]]:
        raise NotImplementedError

    def repair(self, params: list[BaseParam]) -> list[BaseParam]:
        raise NotImplementedError


class BaseAlgo:
    def __init__(self,
                 is_stop: Callable[[list[BaseResult]], bool],
                 validator: BaseValidator,
                 runner: str = "./run.sh",
                 verbose: bool = False,
                 cache_path: str = "msprof_cache.json") -> None:
        self.is_stop = is_stop
        self.validator = validator
        self.runner = runner
        self.verbose = verbose
        self.cache_path = Path(cache_path)
        self._cache = self._load_cache()
        self._audit_index = 0

    def __call__(self, *args, **kwargs) -> list[BaseResult]:
        k = 1
        results = [self.run(*args, **kwargs)]
        print(f"STEP={k}, RESULTS={results}")
        while not self.is_stop(results):
            results.append(self.run(*args, **kwargs))
            k += 1
            print(f"STEP={k}, RESULTS={results}")
        return results

    def run(self, *args, **kwargs) -> BaseResult:
        raise NotImplementedError
    
    def _run_estimator(self, params: list[BaseParam]) -> float:
        raise NotImplementedError
    
    def _key(self, params: list[BaseParam]) -> str:
        return ";".join(f"{n}={v}" for n, v in sorted((p.name, p.value) for p in params))

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    raw = json.load(f)
                return {k: (float("inf") if v == "inf" else float(v)) for k, v in raw.items()}
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        merged = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    merged = json.load(f)
            except Exception:
                merged = {}
        for k, v in self._cache.items():
            merged[k] = "inf" if v == float("inf") else v
        tmp = self.cache_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=4)
        tmp.replace(self.cache_path) 

    def _duration(self, params: list[BaseParam]) -> float:
        proposed = {p.name: int(p.value) for p in params}
        audit_enabled = bool(os.environ.get("MATMUL_AUDIT_LOG_DIR"))
        key = self._key(params)
        if not audit_enabled and key in self._cache:
            if self.verbose:
                print(f"CACHE HIT: {BaseResult(self._cache[key], params)}")
            return self._cache[key]
        
        if not self.validator.is_valid(params):
            params = self.validator.repair(params)  
        legacy_valid = self.validator.is_valid(params)
        if legacy_valid:
            if audit_enabled:
                # Import lazily to avoid BaseValidator's module import cycle.
                from tiling.validators.CostModelValidator import CostModelValidator
                filter_verdict = CostModelValidator().classify(params)
            else:
                filter_verdict = None
            dur = self._run_estimator(params)
            measured_dur = dur
            numeric = self._numeric_result()
            if numeric["status"] != "correct":
                dur = float('inf')
        else:
            dur = float("inf")
            measured_dur = float("inf")
            numeric = {
                "status": "not_executed",
                "reason": "legacy_resource_validator_could_not_repair_candidate",
            }
            filter_verdict = None

        if audit_enabled:
            run_info = getattr(self, "_last_run", {}) if legacy_valid else {}
            if not legacy_valid:
                classification = "legacy_rejected_not_audited"
            elif run_info.get("status") == "failed":
                classification = (
                    "yes_execution_failed"
                    if filter_verdict.valid
                    else "no_execution_failed"
                )
            elif numeric["status"] == "correct":
                classification = (
                    "yes_correct"
                    if filter_verdict.valid
                    else "no_correct_filter_false_negative"
                )
            elif numeric["status"] == "wrong":
                classification = (
                    "yes_wrong_filter_false_positive"
                    if filter_verdict.valid
                    else "no_wrong"
                )
            else:
                classification = (
                    "yes_execution_failed"
                    if filter_verdict.valid
                    else "no_execution_failed"
                )
            record = {
                "schema": "matmul_cost_filter_audit_v1",
                "record_type": "candidate_audit",
                "run_id": os.environ.get("MATMUL_AUDIT_RUN_ID"),
                "candidate_index": self._audit_index,
                "algorithm": getattr(self, "audit_algorithm", type(self).__name__),
                "proposed_params": proposed,
                "executed_params": {p.name: int(p.value) for p in params},
                "legacy_resource_filter": "yes" if legacy_valid else "no",
                "cost_model_filter": None if filter_verdict is None else {
                    "predicted": filter_verdict.predicted,
                    "rules": list(filter_verdict.rules),
                    "materialized_tiling": filter_verdict.tiling,
                },
                "execution": {
                    **run_info,
                    "latency_us": measured_dur if math.isfinite(measured_dur) else None,
                },
                "numeric": numeric,
                "classification": classification,
            }
            self._append_audit(record)
            predicted = (
                filter_verdict.predicted if filter_verdict is not None else "not_classified"
            )
            materialized = filter_verdict.tiling if filter_verdict is not None else {}
            shape_text = (
                f"{materialized.get('M', 'NA')}x"
                f"{materialized.get('N', 'NA')}x"
                f"{materialized.get('K', 'NA')}"
            )
            latency = record["execution"]["latency_us"]
            latency_text = "NA" if latency is None else f"{latency:.6f}"
            print(
                "MATMUL_FILTER_AUDIT_CANDIDATE "
                f"algorithm={record['algorithm']} index={self._audit_index} "
                f"shape={shape_text} "
                f"filter={predicted} actual={numeric['status']} "
                f"latency_us={latency_text} classification={classification}",
                flush=True,
            )
            self._audit_index += 1
            
        if self.verbose:
            print(f"RUN: {BaseResult(dur, params)}")
            
        if not audit_enabled:
            self._cache[key] = dur
            self._save_cache()
        
        return dur
    
    
    def _numeric_result(self,
                        relative_tol: float = 1e-6,
                        absolute_tol: float = 1e-9,
                        error_tol: float = 0.0) -> dict:
        output_path = Path("./output/output.bin")
        golden_path = Path("./output/golden.bin")
        if not output_path.is_file():
            return {"status": "not_available", "reason": "output_missing"}
        if not golden_path.is_file():
            return {"status": "not_available", "reason": "golden_missing"}
        try:
            output = np.fromfile(output_path, dtype=np.float32).reshape(-1)
            golden = np.fromfile(golden_path, dtype=np.float32).reshape(-1)
        except Exception as error:
            return {"status": "not_available", "reason": f"read_failed:{error}"}
        if output.size != golden.size or golden.size == 0:
            return {
                "status": "wrong",
                "reason": "element_count_mismatch",
                "output_elements": int(output.size),
                "golden_elements": int(golden.size),
            }

        close = np.isclose(
            output,
            golden,
            rtol=relative_tol,
            atol=absolute_tol,
            equal_nan=True,
        )
        different = np.flatnonzero(~close)
        mismatch_count = int(different.size)
        error_ratio = mismatch_count / int(golden.size)
        result = {
            "status": "correct" if error_ratio <= error_tol else "wrong",
            "elements": int(golden.size),
            "mismatches": mismatch_count,
            "error_ratio": error_ratio,
            "rtol": relative_tol,
            "atol": absolute_tol,
            "accepted_error_ratio": error_tol,
        }
        if mismatch_count:
            index = int(different[0])
            result["first_error"] = {
                "index": index,
                "expected": float(golden[index]),
                "actual": float(output[index]),
            }
        if self.verbose:
            print(f"IS_RIGHT: {result['status'] == 'correct'}, error ratio = {error_ratio}")
        return result

    def _is_right(self,
                  absolute_tol: float = 1e-9,
                  error_tol: float = 0.0) -> bool:
        return self._numeric_result(
            absolute_tol=absolute_tol,
            error_tol=error_tol,
        )["status"] == "correct"

    @staticmethod
    def _append_audit(record: dict) -> None:
        log_dir = Path(os.environ["MATMUL_AUDIT_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        max_bytes = 50 * 1024 * 1024
        index = 1
        while True:
            path = log_dir / f"{index}.log"
            if not path.exists() or path.stat().st_size + len(line.encode("utf-8")) <= max_bytes:
                with path.open("a", encoding="utf-8") as output:
                    output.write(line)
                return
            index += 1
