"""Hardware-constrained MatMulV3 candidate search."""

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
from .one_shot import OneShotDecision, select_one_shot_candidate
from .racing import RacingPlan, TemplateEvidence, plan_template_race

__all__ = [
    "KNOWLEDGE_FIELDS",
    "BehaviorTarget",
    "Candidate",
    "CandidateEngine",
    "GenerationBudget",
    "Hardware",
    "MeasuredObservation",
    "OneShotDecision",
    "RacingPlan",
    "Schedule",
    "SearchConfig",
    "SearchResult",
    "SolverReport",
    "Template",
    "TemplateEvidence",
    "Workload",
    "plan_template_race",
    "select_one_shot_candidate",
]
