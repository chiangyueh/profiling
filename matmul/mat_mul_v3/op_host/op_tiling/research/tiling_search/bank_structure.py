from __future__ import annotations

from dataclasses import dataclass

from .contracts import template_of
from .domain import KNOWLEDGE_FIELDS, Schedule, Template


BANK_SUBSYSTEM_FIELDS = {
    "template": frozenset({"tilingEnable"}),
    "dispatch": frozenset(
        {
            "usedCoreNum",
            "singleCoreM",
            "singleCoreN",
            "singleCoreK",
        }
    ),
    "l0": frozenset(
        {
            "baseM",
            "baseN",
            "baseK",
            "dbL0A",
            "dbL0B",
            "dbL0C",
        }
    ),
    "l1": frozenset(
        {
            "depthA1",
            "depthB1",
            "stepM",
            "stepN",
            "stepKa",
            "stepKb",
        }
    ),
    "traversal": frozenset({"iterateOrder"}),
    "l2": frozenset(
        {
            "l2MTileCnt",
            "l2NTileCnt",
            "l2MTileBlock",
            "l2NTileBlock",
            "l2IterateOrder",
        }
    ),
}

_FIELD_TO_SUBSYSTEM = {
    field: subsystem
    for subsystem, fields in BANK_SUBSYSTEM_FIELDS.items()
    for field in fields
}

# These fields are encoded in the common 23-field record but are not consumed
# by the selected kernel implementation. The single-core split-K block reads
# only calOrder from L2cacheTilePara; deterministic split-K does not read that
# structure at all.
_SINGLE_CORE_SPLIT_K_IGNORED = frozenset(
    {
        "l2MTileCnt",
        "l2NTileCnt",
        "l2MTileBlock",
        "l2NTileBlock",
    }
)
_DETERMINISTIC_SPLIT_K_IGNORED = frozenset(
    {
        *_SINGLE_CORE_SPLIT_K_IGNORED,
        "l2IterateOrder",
    }
)

if set(_FIELD_TO_SUBSYSTEM) != set(KNOWLEDGE_FIELDS):
    missing = set(KNOWLEDGE_FIELDS) - set(_FIELD_TO_SUBSYSTEM)
    duplicate_or_unknown = set(_FIELD_TO_SUBSYSTEM) - set(KNOWLEDGE_FIELDS)
    raise RuntimeError(
        "bank subsystem schema does not cover the 23-field record: "
        f"missing={sorted(missing)} "
        f"unknown={sorted(duplicate_or_unknown)}"
    )


@dataclass(frozen=True)
class BankTransition:
    changed_fields: frozenset[str]
    changed_subsystems: frozenset[str]

    @property
    def execution_subsystems(self) -> frozenset[str]:
        return self.changed_subsystems - {"traversal", "l2"}

    @property
    def preserves_execution_structure(self) -> bool:
        return not self.execution_subsystems

    @property
    def risk_tier(self) -> int:
        if self.preserves_execution_structure:
            return 0
        if (
            "template" not in self.execution_subsystems
            and len(self.execution_subsystems) == 1
        ):
            return 1
        if "template" not in self.execution_subsystems:
            return 2
        return 3


def kernel_ignored_fields(schedule: Schedule) -> frozenset[str]:
    template = template_of(schedule)
    if template == Template.SINGLE_CORE_SPLIT_K:
        return _SINGLE_CORE_SPLIT_K_IGNORED
    if template == Template.DETERMINISTIC_SPLIT_K:
        return _DETERMINISTIC_SPLIT_K_IGNORED
    return frozenset()


def schedules_execution_equivalent(
    left: Schedule,
    right: Schedule,
) -> bool:
    if template_of(left) != template_of(right):
        return False
    ignored = kernel_ignored_fields(left)
    return all(
        left_value == right_value or field in ignored
        for field, left_value, right_value in zip(
            KNOWLEDGE_FIELDS,
            left.values,
            right.values,
        )
    )


def bank_transition(bank: Schedule, candidate: Schedule) -> BankTransition:
    ignored = (
        kernel_ignored_fields(bank)
        if template_of(bank) == template_of(candidate)
        else frozenset()
    )
    changed_fields = frozenset(
        field
        for field, bank_value, candidate_value in zip(
            KNOWLEDGE_FIELDS,
            bank.values,
            candidate.values,
        )
        if bank_value != candidate_value and field not in ignored
    )
    return BankTransition(
        changed_fields=changed_fields,
        changed_subsystems=frozenset(
            _FIELD_TO_SUBSYSTEM[field] for field in changed_fields
        ),
    )


def subsystem_mask_distance(
    left: BankTransition,
    right: BankTransition,
) -> float:
    union = left.changed_subsystems | right.changed_subsystems
    if not union:
        return 0.0
    return len(left.changed_subsystems ^ right.changed_subsystems) / len(
        union
    )
