from __future__ import annotations

from dataclasses import dataclass

@dataclass
class AscendLimits:
    max_cores: int
    L0A_size: int
    L0B_size: int
    L0C_size: int
    L1_size: int
    
@dataclass
class MatmulLimits(AscendLimits):
    domains: dict[str, list[int]]
    dtype_size: int
    
