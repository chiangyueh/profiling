from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .behavior import select_behavior_coverage
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
    feedback_mutations,
    feedback_targets,
    fingerprint,
    local_anchor_mutations,
)
from .solvers import (
    Al1FullLoadSolver,
    BaseSolver,
    Bl1FullLoadSolver,
    DeterministicSplitKSolver,
    SingleCoreSplitKSolver,
)


@dataclass(frozen=True)
class SearchConfig:
    budget: GenerationBudget = GenerationBudget()


class CandidateEngine:
    """Independent candidate generator with no legacy fallback path."""

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
        self.solvers = tuple(
            solvers
            or (
                BaseSolver(),
                SingleCoreSplitKSolver(),
                DeterministicSplitKSolver(),
                Al1FullLoadSolver(),
                Bl1FullLoadSolver(),
            )
        )

    def generate(
        self,
        workload: Workload,
        hardware: Hardware,
        *,
        local_anchor: Schedule | None = None,
    ) -> SearchResult:
        budget = self.config.budget
        targets = feedback_targets(workload, hardware, self.observations)
        iterators = []
        stats = []
        for solver in self.solvers:
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
                candidate = Candidate(
                    schedule=schedule,
                    template=template_of(schedule),
                    source="contract_global",
                    rationale=(
                        "independent hardware and template contract solver"
                    ),
                )
                independent.append(candidate)
                stats[index]["emitted"] += 1
                total_legal += 1
            active = remaining

        expanded = [
            *independent,
            *feedback_mutations(
                workload, hardware, self.observations
            ),
            *local_anchor_mutations(
                workload, hardware, local_anchor
            ),
        ]
        filtered: list[Candidate] = []
        excluded = 0
        seen.clear()
        for candidate in expanded:
            signature = candidate.schedule.signature()
            if signature in seen:
                continue
            seen.add(signature)
            if fingerprint(workload, candidate.schedule) in self.exclusions:
                excluded += 1
                continue
            filtered.append(candidate)

        behavior_pool = select_behavior_coverage(
            workload,
            filtered,
            self.observations,
            hardware,
            budget.behavior_candidates,
        )
        selected = select_behavior_coverage(
            workload,
            behavior_pool,
            self.observations,
            hardware,
            budget.callback_candidates,
        )
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
        )
