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
