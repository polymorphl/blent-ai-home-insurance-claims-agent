REQUIRED_FIELDS = ["date", "incident_type", "description", "has_photos"]

EXTRACT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_claim_fields",
        "description": (
            "Extract insurance claim fields from the policyholder's message. "
            "Use null for fields not mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": ["string", "null"],
                    "description": "Date of incident in ISO format (YYYY-MM-DD), or null if not mentioned.",
                },
                "incident_type": {
                    "type": ["string", "null"],
                    "enum": ["water_damage", "fire", "theft"],
                    "description": (
                        "Type of incident: water_damage (leak, infiltration, flooding), "
                        "fire (fire, explosion, smoke), theft (burglary, robbery, vandalism), "
                        "or null if unclear."
                    ),
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "Description of the damage observed, or null if not mentioned.",
                },
                "has_photos": {
                    "type": ["boolean", "null"],
                    "description": (
                        "True if photo attachments are listed in the message, "
                        "false if explicitly none, null if not mentioned."
                    ),
                },
            },
            "required": ["date", "incident_type", "description", "has_photos"],
        },
    },
}


def validate_date(date_str: str | None) -> str | None:
    """Return the date if it falls within a plausible range (last 5 years to today), else None."""
    if date_str is None:
        return None
    from datetime import date, datetime
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = date.today()
        five_years_ago = today.replace(year=today.year - 5)
        if five_years_ago <= parsed <= today:
            return date_str
    except ValueError:
        pass
    return None


def merge_claim_data(existing: dict, new_fields: dict) -> dict:
    """Merge new extracted fields into existing claim data, keeping non-null values."""
    result = dict(existing)
    for field in REQUIRED_FIELDS:
        if new_fields.get(field) is not None:
            result[field] = new_fields[field]
    return result


def completeness_score(claim: dict) -> float:
    """Calculate the fraction of required fields that have non-null values."""
    present = sum(1 for f in REQUIRED_FIELDS if claim.get(f) is not None)
    return present / len(REQUIRED_FIELDS)
