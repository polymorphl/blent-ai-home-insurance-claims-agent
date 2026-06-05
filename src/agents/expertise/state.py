from typing import TypedDict


class ExpertiseState(TypedDict, total=False):
    verdict: dict
    severity: str | None
    cost_range: tuple[int, int] | None
    compensable_amount: tuple[int, int] | None
    deductible_applied: int | None
    ceiling_applied: int | None
    summary: str | None
    report: dict | None
