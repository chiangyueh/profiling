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
from .one_shot import (
    BankRelativeEffectModel,
    BankRelativePrediction,
    BankRelativeValidation,
    OneShotDecision,
    select_calibration_candidates,
    select_one_shot_candidate,
    validate_bank_relative_selector,
)
from .racing import RacingPlan, TemplateEvidence, plan_template_race
from .ranking import PairwiseLatencyRanker, PairwiseLatencyPrediction

__all__ = [
    "KNOWLEDGE_FIELDS",
    "BehaviorTarget",
    "BankRelativeEffectModel",
    "BankRelativePrediction",
    "BankRelativeValidation",
    "Candidate",
    "CandidateEngine",
    "GenerationBudget",
    "Hardware",
    "MeasuredObservation",
    "OneShotDecision",
    "PairwiseLatencyPrediction",
    "PairwiseLatencyRanker",
    "RacingPlan",
    "Schedule",
    "SearchConfig",
    "SearchResult",
    "SolverReport",
    "Template",
    "TemplateEvidence",
    "Workload",
    "plan_template_race",
    "select_calibration_candidates",
    "select_one_shot_candidate",
    "validate_bank_relative_selector",
]
