from __future__ import annotations

import math
from dataclasses import dataclass

from .behavior import BehaviorVector
from .contracts import template_of
from .domain import Hardware, Schedule, Template, Workload


SPLIT_K_TEMPLATES = {
    Template.SINGLE_CORE_SPLIT_K,
    Template.DETERMINISTIC_SPLIT_K,
}


@dataclass(frozen=True)
class TemplateCompetition:
    same_template: bool
    hardware_opportunity: bool
    evidence_opportunity: bool
    competitive: bool
    bank_active_cores: float
    candidate_active_cores: float
    active_core_gain: float
    compute_floor_ratio: float
    output_overhead_ratio: float
    conservative_floor_ratio: float
    reason: str


def compare_templates(
    workload: Workload,
    bank: Schedule,
    candidate: Schedule,
    bank_vector: BehaviorVector,
    candidate_vector: BehaviorVector,
    hardware: Hardware,
    *,
    effect_samples: int,
    effect_support: float,
    effect_upper_ratio: float,
) -> TemplateCompetition:
    """Determine whether a template switch has a hardware-level opportunity.

    This is not an applicability gate. Every schedule has already passed the
    common and template kernel contracts. The result only controls whether a
    cross-template schedule may consume the single one-shot measurement.
    """

    bank_template = template_of(bank)
    candidate_template = template_of(candidate)
    same_template = bank_template == candidate_template
    bank_active = max(1.0, bank_vector.metrics["active_cores"])
    candidate_active = max(
        1.0, candidate_vector.metrics["active_cores"]
    )
    active_gain = candidate_active / bank_active

    bank_padding = max(
        1.0e-6, bank_vector.metrics["padding_efficiency"]
    )
    candidate_padding = max(
        1.0e-6, candidate_vector.metrics["padding_efficiency"]
    )
    compute_floor = (
        bank_active
        / candidate_active
        * bank_padding
        / candidate_padding
    )
    bank_write = max(
        1.0, bank_vector.metrics["output_write_multiplier"]
    )
    candidate_write = max(
        1.0, candidate_vector.metrics["output_write_multiplier"]
    )
    output_overhead = candidate_write / bank_write
    # Partial-output traffic is not free, but it overlaps with Cube work. A
    # bounded linear term is more realistic than treating every partial write
    # as an additional full MatMul.
    conservative_floor = compute_floor * (
        1.0 + 0.04 * max(0.0, output_overhead - 1.0)
    )

    core_limit = max(
        1.0, min(workload.max_cores, hardware.aic_cores)
    )
    bank_underfilled = bank_active < 0.80 * core_limit
    parallelism_gain = (
        bank_underfilled
        and active_gain >= 1.25
        and conservative_floor <= 0.95
    )
    # A BASE schedule can also be a valid escape from an incumbent split-K
    # kernel when it removes reduction traffic without losing parallelism.
    reduction_escape = (
        bank_template in SPLIT_K_TEMPLATES
        and candidate_template == Template.BASE
        and candidate_active >= bank_active
        and conservative_floor <= 1.02
    )
    hardware_opportunity = parallelism_gain or reduction_escape
    evidence_opportunity = (
        effect_samples >= 3
        and effect_support >= 0.20
        and math.isfinite(effect_upper_ratio)
        and effect_upper_ratio <= 0.99
        and conservative_floor <= 1.05
    )
    competitive = (
        same_template or hardware_opportunity or evidence_opportunity
    )
    if same_template:
        reason = "same_template"
    elif parallelism_gain:
        reason = "bank_underfilled_parallelism_gain"
    elif reduction_escape:
        reason = "split_reduction_escape"
    elif evidence_opportunity:
        reason = "paired_cross_template_upper_bound"
    else:
        reason = "no_cross_template_advantage"
    return TemplateCompetition(
        same_template=same_template,
        hardware_opportunity=hardware_opportunity,
        evidence_opportunity=evidence_opportunity,
        competitive=competitive,
        bank_active_cores=bank_active,
        candidate_active_cores=candidate_active,
        active_core_gain=active_gain,
        compute_floor_ratio=compute_floor,
        output_overhead_ratio=output_overhead,
        conservative_floor_ratio=conservative_floor,
        reason=reason,
    )
