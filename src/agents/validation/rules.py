from datetime import date, datetime, timedelta

COVERAGE_RULES: dict[str, dict] = {
    "water_damage": {"ceiling": 25000, "deductible": 150, "deadline_days": 5},
    "fire": {"ceiling": 100000, "deductible": 300, "deadline_days": 5},
    "theft": {"ceiling": 20000, "deductible": 200, "deadline_days": 2},
}


def business_days_since(date_str: str, today: date | None = None) -> int:
    """Count business days (Mon–Fri) elapsed from date_str up to today."""
    incident = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = today or date.today()
    if incident >= today:
        return 0
    count = 0
    current = incident
    while current < today:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count


_REQUIRED_CLAIM_FIELDS = ["date", "incident_type", "description", "has_photos"]


def check_conformity(claim: dict) -> list[str]:
    """Return a list with at most one conformity error (fail-fast)."""
    for field in _REQUIRED_CLAIM_FIELDS:
        if claim.get(field) is None:
            return [f"Dossier incomplet : {field} manquant."]
    if claim.get("has_photos") is False:
        return [f"Des photos sont requises pour un sinistre de type {claim['incident_type']}."]
    return []
