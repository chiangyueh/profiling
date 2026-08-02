from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping, Sequence

from .contracts import template_of
from .domain import Candidate, MeasuredObservation, Template, Workload


@dataclass(frozen=True)
class TemplateEvidence:
    template: Template
    samples: int
    best_ratio: float
    robust_ratio: float
    winners: int
    regressions: int


@dataclass(frozen=True)
class RacingPlan:
    budget: int
    template_quotas: Mapping[Template, int]
    evidence: tuple[TemplateEvidence, ...]
    state: str


def _same_workload_evidence(
    workload: Workload,
    observations: Sequence[MeasuredObservation],
) -> tuple[TemplateEvidence, ...]:
    grouped: dict[Template, list[MeasuredObservation]] = defaultdict(list)
    for observation in observations:
        if (
            observation.workload.identity() != workload.identity()
            or not observation.verified
            or not observation.structured_verified
            or observation.source == "runtime_rejected"
        ):
            continue
        grouped[template_of(observation.schedule)].append(observation)

    evidence = []
    for template, records in grouped.items():
        ratios = sorted(record.measured_ratio for record in records)
        # A single unusually fast point must not eliminate other templates.
        # The median of the best three is optimistic enough for exploration,
        # but requires repeat evidence before declaring a clear leader.
        robust_window = ratios[: min(3, len(ratios))]
        evidence.append(
            TemplateEvidence(
                template=template,
                samples=len(records),
                best_ratio=ratios[0],
                robust_ratio=median(robust_window),
                winners=sum(record.is_verified_winner for record in records),
                regressions=sum(record.is_regression for record in records),
            )
        )
    return tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.robust_ratio,
                item.best_ratio,
                item.template.value,
            ),
        )
    )


def _allocate_quotas(
    templates: Sequence[Template],
    availability: Mapping[Template, int],
    priority: Sequence[Template],
    budget: int,
    *,
    exploration_floor: int,
) -> dict[Template, int]:
    quotas = {template: 0 for template in templates}
    remaining = budget

    # Retain a small counterfactual budget for every callback-accepted
    # template. This is what prevents a weak cost model from making template
    # elimination irreversible.
    for template in templates:
        amount = min(exploration_floor, availability[template], remaining)
        quotas[template] += amount
        remaining -= amount
        if remaining == 0:
            return {key: value for key, value in quotas.items() if value}

    weighted_priority = [
        template for template in priority if template in quotas
    ]
    weighted_priority.extend(
        template for template in templates
        if template not in weighted_priority
    )
    cursor = 0
    while remaining > 0 and weighted_priority:
        template = weighted_priority[cursor % len(weighted_priority)]
        cursor += 1
        if quotas[template] >= availability[template]:
            if all(
                quotas[item] >= availability[item]
                for item in weighted_priority
            ):
                break
            continue
        # The leading template receives two slots for each one assigned to
        # another family, while the runner-up remains represented.
        weight = 2 if template == weighted_priority[0] else 1
        amount = min(
            weight,
            availability[template] - quotas[template],
            remaining,
        )
        quotas[template] += amount
        remaining -= amount
    return {key: value for key, value in quotas.items() if value}


def plan_template_race(
    workload: Workload,
    candidates: Iterable[Candidate],
    observations: Sequence[MeasuredObservation],
    max_budget: int,
) -> RacingPlan:
    candidate_list = list(candidates)
    generated_availability = Counter(
        candidate.template for candidate in candidate_list
    )
    safe_availability = Counter(
        candidate.template
        for candidate in candidate_list
        if not (
            candidate.metrics.get("runtime_risk_score", 0.5) >= 0.75
            and candidate.metrics.get("runtime_risk_support", 0.0) >= 0.25
        )
    )
    # Runtime risk is learned from strict NPU preflight. Keep one uncertain
    # probe per generated template, but let safer templates consume the rest
    # of the budget instead of forcing quota-filling from known bad regions.
    availability = Counter(
        {
            template: min(
                count,
                safe_availability[template] + 1,
            )
            for template, count in generated_availability.items()
        }
    )
    templates = tuple(sorted(availability, key=lambda item: item.value))
    evidence = _same_workload_evidence(workload, observations)
    evidence_by_template = {item.template: item for item in evidence}
    observed_templates = [
        item.template for item in evidence if item.template in availability
    ]
    unobserved_templates = [
        template for template in templates
        if template not in evidence_by_template
    ]

    clear_winner = False
    if evidence:
        leader = evidence[0]
        runner_up = evidence[1] if len(evidence) > 1 else None
        all_templates_sampled = all(
            template in evidence_by_template
            and evidence_by_template[template].samples >= 2
            for template in templates
        )
        clear_winner = (
            all_templates_sampled
            and leader.samples >= 2
            and leader.winners >= 2
            and leader.best_ratio <= 0.98
            and leader.robust_ratio <= 0.99
            and (
                runner_up is None
                or runner_up.best_ratio >= leader.robust_ratio + 0.03
            )
        )

    if not evidence:
        budget = max_budget
        state = "balanced_cold_start"
        exploration_floor = max(1, budget // max(1, len(templates)))
    elif clear_winner:
        # Stage 1 measures 16 schedules. Retain at least 16 stage-2
        # schedules so a stopped race still supplies the promised 32-point
        # NPU experiment while avoiding the full 40 when evidence is decisive.
        budget = min(max_budget, max(16, 2 * len(templates)))
        state = "clear_template_leader"
        exploration_floor = 1
    else:
        budget = max_budget
        state = "templates_competitive"
        exploration_floor = 2

    budget = min(budget, sum(availability.values()))
    priority = [*observed_templates, *unobserved_templates]
    quotas = _allocate_quotas(
        templates,
        availability,
        priority,
        budget,
        exploration_floor=exploration_floor,
    )
    return RacingPlan(
        budget=sum(quotas.values()),
        template_quotas=quotas,
        evidence=evidence,
        state=state,
    )
