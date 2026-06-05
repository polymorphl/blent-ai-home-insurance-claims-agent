COST_TABLE: dict[str, dict[str, tuple[int, int]]] = {
    "water_damage": {
        "low": (200, 1500),
        "medium": (1500, 8000),
        "high": (8000, 20000),
    },
    "fire": {
        "low": (1000, 10000),
        "medium": (10000, 40000),
        "high": (40000, 90000),
    },
    "theft": {
        "low": (200, 2000),
        "medium": (2000, 8000),
        "high": (8000, 18000),
    },
}


def estimate_costs(
    incident_type: str, severity: str, ceiling: int, deductible: int
) -> dict:
    """Map incident type + severity to cost range and apply guarantee rules."""
    if severity == "unknown" or incident_type not in COST_TABLE:
        return {
            "cost_range": (0, 0),
            "compensable_amount": (0, 0),
            "deductible_applied": deductible,
            "ceiling_applied": ceiling,
        }
    cost_low, cost_high = COST_TABLE[incident_type][severity]
    comp_low = max(0, min(ceiling, cost_low - deductible))
    comp_high = max(0, min(ceiling, cost_high - deductible))
    return {
        "cost_range": (cost_low, cost_high),
        "compensable_amount": (comp_low, comp_high),
        "deductible_applied": deductible,
        "ceiling_applied": ceiling,
    }
