from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .behavior import (
    FeedbackCostModel,
    draft_behavior_coverage,
    select_behavior_coverage,
)
from .bank_structure import schedules_execution_equivalent
from .contracts import (
    common_hardware_contract,
    template_kernel_contract,
    template_of,
)
from .domain import (
    Candidate,
    GenerationBudget,
    Hardware,
    MeasuredObservation,
    Schedule,
    SearchResult,
    SolverReport,
    Workload,
)
from .feedback import (
    Fingerprint,
    bank_relative_transfer_candidates,
    feedback_mutations,
    feedback_targets,
    fingerprint,
    local_anchor_mutations,
)
from .solvers import (
    Al1FullLoadSolver,
    BaseExplorationSolver,
    BaseSolver,
    Bl1FullLoadSolver,
    DeterministicSplitKSolver,
    SingleCoreSplitKSolver,
)


@dataclass(frozen=True)
class SearchConfig:
    budget: GenerationBudget = GenerationBudget()
    include_exploration: bool = True


class CandidateEngine:
    """Independent hardware-contract candidate generator."""

    def __init__(
        self,
        *,
        config: SearchConfig | None = None,
        observations: Sequence[MeasuredObservation] = (),
        exclusions: set[Fingerprint] | None = None,
        solvers: Sequence[object] | None = None,
    ) -> None:
        self.config = config or SearchConfig()
        self.observations = tuple(observations)
        self.exclusions = exclusions or set()
        self._cost_models: dict[Hardware, FeedbackCostModel] = {}
        if solvers is not None:
            selected_solvers = tuple(solvers)
        else:
            deployment_solvers = (
                BaseSolver(),
                SingleCoreSplitKSolver(),
                DeterministicSplitKSolver(),
            )
            exploration_solvers = (
                BaseExplorationSolver(),
                DeterministicSplitKSolver(explore_low_core=True),
                Al1FullLoadSolver(),
                Bl1FullLoadSolver(),
            )
            selected_solvers = (
                *deployment_solvers,
                *(
                    exploration_solvers
                    if self.config.include_exploration
                    else ()
                ),
            )
        self.solvers = selected_solvers

    def generate(
        self,
        workload: Workload,
        hardware: Hardware,
        *,
        local_anchor: Schedule | None = None,
    ) -> SearchResult:
        budget = self.config.budget
        cost_model = self._cost_models.get(hardware)
        if cost_model is None:
            cost_model = FeedbackCostModel(self.observations, hardware)
            self._cost_models[hardware] = cost_model
        targets = feedback_targets(workload, hardware, self.observations)
        iterators = []
        stats = []
        for solver in self.solvers:
            source = getattr(solver, "source", "")
            if not source:
                raise ValueError(
                    f"{type(solver).__name__} must declare candidate source"
                )
            solver_targets = [
                target
                for target in targets
                if target.template is None
                or target.template == solver.template
            ]
            iterators.append(
                iter(solver.generate(workload, hardware, solver_targets))
            )
            stats.append(
                {
                    "template": solver.template,
                    "source": source,
                    "raw": 0,
                    "common": 0,
                    "template_legal": 0,
                    "emitted": 0,
                    "failures": Counter(),
                }
            )

        independent: list[Candidate] = []
        seen: set[tuple[int, ...]] = set()
        active = list(range(len(iterators)))
        total_raw = 0
        total_legal = 0
        while (
            active
            and total_raw < budget.raw_attempts
            and total_legal < budget.legal_candidates
        ):
            remaining = []
            for index in active:
                if (
                    total_raw >= budget.raw_attempts
                    or total_legal >= budget.legal_candidates
                ):
                    break
                try:
                    schedule = next(iterators[index])
                except StopIteration:
                    continue
                remaining.append(index)
                stats[index]["raw"] += 1
                total_raw += 1
                common = common_hardware_contract(
                    workload, schedule, hardware
                )
                if not common.valid:
                    stats[index]["failures"].update(common.violations)
                    continue
                stats[index]["common"] += 1
                specific = template_kernel_contract(
                    workload, schedule, hardware
                )
                if not specific.valid:
                    stats[index]["failures"].update(specific.violations)
                    continue
                stats[index]["template_legal"] += 1
                signature = schedule.signature()
                if signature in seen:
                    continue
                seen.add(signature)
                source = stats[index]["source"]
                rationale = (
                    "contract-coupled deployment policy"
                    if source == "contract_coupled_policy"
                    else "independent hardware and template contract solver"
                )
                candidate = Candidate(
                    schedule=schedule,
                    template=template_of(schedule),
                    source=source,
                    rationale=rationale,
                )
                independent.append(candidate)
                stats[index]["emitted"] += 1
                total_legal += 1
            active = remaining

        measured_mutations = feedback_mutations(
            workload, hardware, self.observations
        )
        transferred = bank_relative_transfer_candidates(
            workload,
            hardware,
            local_anchor,
            independent,
            self.observations,
        )
        local_candidates = local_anchor_mutations(
            workload, hardware, local_anchor
        )
        expanded = [
            # Preserve feedback provenance when the same schedule also appears
            # in the independent stream. Stage two can then distinguish a
            # measured structural hypothesis from a generic coverage probe.
            *transferred,
            *measured_mutations,
            *independent,
            *local_candidates,
        ]
        filtered: list[Candidate] = []
        excluded = 0
        seen.clear()
        for candidate in expanded:
            signature = candidate.schedule.signature()
            if signature in seen:
                continue
            seen.add(signature)
            if fingerprint(
                workload, candidate.schedule
            ) in self.exclusions:
                excluded += 1
                continue
            filtered.append(candidate)

        draft_limit = max(
            budget.behavior_candidates,
            budget.callback_candidates * 4,
        )
        draft_pool = draft_behavior_coverage(
            workload,
            filtered,
            hardware,
            draft_limit,
        )
        behavior_pool = select_behavior_coverage(
            workload,
            draft_pool,
            self.observations,
            hardware,
            budget.behavior_candidates,
            template_probe_floor=8,
            allow_risky_template_probes=True,
            cost_model=cost_model,
        )
        selected = select_behavior_coverage(
            workload,
            behavior_pool,
            self.observations,
            hardware,
            budget.callback_candidates,
            template_probe_floor=3,
            allow_risky_template_probes=True,
            cost_model=cost_model,
        )
        protected_reproductions = [
            candidate
            for candidate in filtered
            if (
                local_anchor is not None
                and candidate.source != "local_bank_anchor"
                and schedules_execution_equivalent(
                    local_anchor, candidate.schedule
                )
            )
        ][:1]
        protected_signatures = {
            candidate.schedule.signature()
            for candidate in protected_reproductions
        }
        protected_local = [
            candidate
            for candidate in filtered
            if (
                candidate.source == "local_bank_anchor"
                and candidate.schedule.signature()
                not in protected_signatures
            )
        ][: max(1, budget.callback_candidates // 4)]
        protected_signatures.update(
            candidate.schedule.signature()
            for candidate in protected_local
        )
        protected_feedback = [
            candidate
            for candidate in filtered
            if (
                candidate.source
                in {
                    "feedback_winner_transfer",
                    "feedback_winner_mutation",
                    "feedback_regression_counterfactual",
                }
                and candidate.schedule.signature()
                not in protected_signatures
            )
        ][: max(1, budget.callback_candidates // 3)]
        protected_signatures.update(
            candidate.schedule.signature()
            for candidate in protected_feedback
        )
        protected_policy = [
            candidate
            for candidate in filtered
            if (
                candidate.source == "contract_coupled_policy"
                and candidate.schedule.signature()
                not in protected_signatures
            )
        ][: max(1, budget.callback_candidates // 4)]
        protected_signatures.update(
            candidate.schedule.signature()
            for candidate in protected_policy
        )
        selected = [
            *protected_reproductions,
            *protected_feedback,
            *protected_local,
            *protected_policy,
            *(
                candidate
                for candidate in selected
                if candidate.schedule.signature()
                not in protected_signatures
            ),
        ][: budget.callback_candidates]
        reports = tuple(
            SolverReport(
                template=item["template"],
                raw_generated=item["raw"],
                common_legal=item["common"],
                template_legal=item["template_legal"],
                emitted=item["emitted"],
                failure_reasons=tuple(
                    sorted(item["failures"].items())
                ),
            )
            for item in stats
        )
        return SearchResult(
            candidates=tuple(behavior_pool),
            callback_candidates=tuple(selected),
            reports=reports,
            excluded_fingerprints=excluded,
            observation_count=len(self.observations),
            behavior_bins=len(
                {candidate.behavior_key for candidate in selected}
            ),
            legal_candidates=len(filtered),
            draft_candidates=len(draft_pool),
        )
