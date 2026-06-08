from src.agents.declaration.tools import REQUIRED_FIELDS, completeness_score


def declaration_completeness(extracted: dict) -> dict:
    """Presence of the 4 required claim fields in the extracted claim."""
    per_field = {f: extracted.get(f) is not None for f in REQUIRED_FIELDS}
    n = sum(per_field.values())
    return {
        "score": completeness_score(extracted),
        "per_field": per_field,
        "fields_ok": f"{n}/{len(REQUIRED_FIELDS)}",
    }


def validation_factual(verdict: dict, expected: dict) -> dict:
    """Verdict status correctness + contract-fact (coverage) correctness."""
    status = verdict.get("status")
    status_correct = status == expected.get("status")
    coverage = verdict.get("coverage")
    if expected.get("status") == "approved":
        coverage_correct = (
            coverage is not None
            and coverage.get("ceiling") == expected.get("ceiling")
            and coverage.get("deductible") == expected.get("deductible")
        )
    else:
        coverage_correct = coverage is None
    return {
        "status": status,
        "status_correct": status_correct,
        "coverage_correct": coverage_correct,
    }


def orchestration_provider(chosen: str | None, expected: str) -> dict:
    """Provider selection precision."""
    return {"provider": chosen, "correct": chosen == expected}
