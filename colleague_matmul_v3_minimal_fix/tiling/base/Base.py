from typing import Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
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
        key = self._key(params)
        if key in self._cache:
            if self.verbose:
                print(f"CACHE HIT: {BaseResult(self._cache[key], params)}")
            return self._cache[key]
        
        if not self.validator.is_valid(params):
            params = self.validator.repair(params)  
        if self.validator.is_valid(params):
            dur = self._run_estimator(params)
            if not self._is_right():
                dur = float('inf')
        else:
            dur = float("inf")
            
        if self.verbose:
            print(f"RUN: {BaseResult(dur, params)}")
            
        self._cache[key] = dur
        self._save_cache()
        
        return dur
    
    
    def _is_right(self,
                  absolute_tol: float = 1e-9,
                  error_tol: float = 1e-4) -> bool:
        output = np.fromfile("./output/output.bin", dtype=np.float32).reshape(-1)
        golden = np.fromfile("./output/golden.bin", dtype=np.float32).reshape(-1)
        different_elements_num = np.abs(output - golden >= absolute_tol).sum()
        error_ratio = different_elements_num / golden.size
        if self.verbose:
            print(f"IS_RIGHT: {error_ratio <= error_tol}, error ratio = {error_ratio}")
        return error_ratio <= error_tol