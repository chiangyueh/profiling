from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


KNOWLEDGE_FIELDS = (
    "usedCoreNum",
    "singleCoreM",
    "singleCoreN",
    "singleCoreK",
    "baseM",
    "baseN",
    "baseK",
    "depthA1",
    "depthB1",
    "stepM",
    "stepN",
    "iterateOrder",
    "stepKa",
    "stepKb",
    "dbL0A",
    "dbL0B",
    "dbL0C",
    "l2MTileCnt",
    "l2NTileCnt",
    "l2MTileBlock",
    "l2NTileBlock",
    "l2IterateOrder",
    "tilingEnable",
)

INPUT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}
OUTPUT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


class Template(str, Enum):
    BASE = "BASE"
    SINGLE_CORE_SPLIT_K = "SINGLE_CORE_SPLIT_K"
    DETERMINISTIC_SPLIT_K = "DETERMINISTIC_SPLIT_K"
    AL1_FULL_LOAD = "AL1_FULL_LOAD"
    BL1_FULL_LOAD = "BL1_FULL_LOAD"
    BL1_FULL_LOAD_FIXPIPE = "BL1_FULL_LOAD_FIXPIPE"
    BL1_FULL_LOAD_VEC_NZ2ND = "BL1_FULL_LOAD_VEC_NZ2ND"


@dataclass(frozen=True)
class Workload:
    workload_id: str
    m: int
    n: int
    k: int
    dtype: str
    trans_a: bool
    trans_b: bool
    max_cores: int

    def identity(self) -> tuple[int, int, int, str, bool, bool, int]:
        return (
            self.m,
            self.n,
            self.k,
            self.dtype,
            self.trans_a,
            self.trans_b,
            self.max_cores,
        )


@dataclass(frozen=True)
class Hardware:
    aic_cores: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    l1_bytes: int
    l2_bytes: int
    l2_bytes_per_cycle_per_core: float
    hbm_bytes_per_cycle_per_core: float

    @property
    def effective_l1_bytes(self) -> int:
        return ((self.l1_bytes + 1023) // 1024) * 1024


@dataclass(frozen=True)
class Schedule:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(KNOWLEDGE_FIELDS):
            raise ValueError(
                f"schedule has {len(self.values)} fields, "
                f"expected {len(KNOWLEDGE_FIELDS)}"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, int]) -> "Schedule":
        missing = [field for field in KNOWLEDGE_FIELDS if field not in values]
        if missing:
            raise ValueError(f"schedule is missing fields: {','.join(missing)}")
        return cls(tuple(int(values[field]) for field in KNOWLEDGE_FIELDS))

    @classmethod
    def from_signature(cls, signature: str) -> "Schedule":
        try:
            values = tuple(int(value) for value in signature.split(":"))
        except ValueError as exception:
            raise ValueError("invalid schedule signature") from exception
        return cls(values)

    def __getitem__(self, field: str) -> int:
        try:
            return self.values[KNOWLEDGE_FIELDS.index(field)]
        except ValueError as exception:
            raise KeyError(field) from exception

    def as_dict(self) -> dict[str, int]:
        return dict(zip(KNOWLEDGE_FIELDS, self.values))

    def replace(self, **updates: int) -> "Schedule":
        values = self.as_dict()
        unknown = set(updates).difference(KNOWLEDGE_FIELDS)
        if unknown:
            raise ValueError(f"unknown schedule fields: {sorted(unknown)}")
        values.update({name: int(value) for name, value in updates.items()})
        return Schedule.from_mapping(values)

    def signature(self) -> tuple[int, ...]:
        return self.values

    def signature_text(self) -> str:
        return ":".join(str(value) for value in self.values)


@dataclass(frozen=True)
class BehaviorTarget:
    template: Template | None = None
    l0_occupancy: float | None = None
    l1_occupancy: float | None = None
    core_rounds: float | None = None
    k_passes: float | None = None
    padding_efficiency: float | None = None
    l2_working_set_ratio: float | None = None
    split_reduction_ratio: float | None = None
    full_load_resident_ratio: float | None = None
    origin: str = "coverage"


@dataclass(frozen=True)
class Candidate:
    schedule: Schedule
    template: Template
    source: str
    rationale: str
    acquisition: float = 0.0
    parent_signatures: tuple[tuple[int, ...], ...] = ()
    behavior_key: tuple[object, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def with_selection(
        self,
        *,
        acquisition: float,
        behavior_key: tuple[object, ...],
        metrics: Mapping[str, float],
    ) -> "Candidate":
        return Candidate(
            schedule=self.schedule,
            template=self.template,
            source=self.source,
            rationale=self.rationale,
            acquisition=acquisition,
            parent_signatures=self.parent_signatures,
            behavior_key=behavior_key,
            metrics=dict(metrics),
        )


@dataclass(frozen=True)
class MeasuredObservation:
    workload: Workload
    schedule: Schedule
    ratio_vs_official: float
    ratio_vs_bank: float
    source: str
    record_id: str
    status_vs_official: str = ""
    status_vs_bank: str = ""
    verified: bool = False
    structured_verified: bool = False
    bank_schedule: Schedule | None = None

    @property
    def measured_ratio(self) -> float:
        return max(self.ratio_vs_official, self.ratio_vs_bank)

    @property
    def is_winner(self) -> bool:
        if self.status_vs_official or self.status_vs_bank:
            return (
                self.status_vs_official == "improved"
                and self.status_vs_bank == "improved"
            )
        return (
            self.ratio_vs_official <= 0.99
            and self.ratio_vs_bank <= 0.99
        )

    @property
    def is_verified_winner(self) -> bool:
        return self.verified and self.is_winner

    @property
    def is_regression(self) -> bool:
        if self.status_vs_official or self.status_vs_bank:
            return (
                self.status_vs_official == "regressed"
                or self.status_vs_bank == "regressed"
            )
        return (
            self.ratio_vs_official >= 1.01
            or self.ratio_vs_bank >= 1.01
        )


@dataclass(frozen=True)
class GenerationBudget:
    raw_attempts: int = 12000
    legal_candidates: int = 5000
    behavior_candidates: int = 192
    callback_candidates: int = 96
    npu_candidates: int = 32


@dataclass(frozen=True)
class SolverReport:
    template: Template
    raw_generated: int
    common_legal: int
    template_legal: int
    emitted: int
    failure_reasons: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[Candidate, ...]
    callback_candidates: tuple[Candidate, ...]
    reports: tuple[SolverReport, ...]
    excluded_fingerprints: int
    observation_count: int
    behavior_bins: int
    legal_candidates: int = 0
    draft_candidates: int = 0
