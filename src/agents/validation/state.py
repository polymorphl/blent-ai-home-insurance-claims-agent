from typing import TypedDict


class ValidationState(TypedDict, total=False):
    """State for the claim validation workflow."""
    claim: dict
    conformity_errors: list[str]
    coverage_errors: list[str]
    verdict: dict | None
    today_override: str | None
