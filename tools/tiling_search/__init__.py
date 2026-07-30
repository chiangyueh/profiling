"""Hardware-constrained MatMulV3 candidate search.

This package is intentionally independent from the legacy candidate
constructors in refine_matmul_v3_candidates.py.  The legacy module may adapt
its Workload/Hardware objects into these types, but imports never flow in the
opposite direction.
"""

from .domain import (
    KNOWLEDGE_FIELDS,
    BehaviorTarget,
    Candidate,
    GenerationBudget,
    Hardware,
    MeasuredObservation,
    Schedule,
    SearchResult,
    SolverReport,
    Template,
    Workload,
)
from .orchestrator import CandidateEngine, SearchConfig

__all__ = [
    "KNOWLEDGE_FIELDS",
    "BehaviorTarget",
    "Candidate",
    "CandidateEngine",
    "GenerationBudget",
    "Hardware",
    "MeasuredObservation",
    "Schedule",
    "SearchConfig",
    "SearchResult",
    "SolverReport",
    "Template",
    "Workload",
]
