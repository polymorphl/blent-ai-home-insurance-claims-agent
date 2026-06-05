from src.agents.expertise.state import ExpertiseState
from src.agents.expertise.rules import COST_TABLE, estimate_costs

def test_expertise_state_accepts_all_fields():
    state: ExpertiseState = {
        "verdict": {},
        "severity": "medium",
        "cost_range": (1500, 8000),
        "compensable_amount": (1350, 7850),
        "deductible_applied": 150,
        "ceiling_applied": 25000,
        "summary": "Résumé.",
        "report": None,
    }
    assert state["severity"] == "medium"
    assert state["cost_range"] == (1500, 8000)

def test_cost_table_covers_all_incident_types():
    assert set(COST_TABLE.keys()) == {"water_damage", "fire", "theft"}

def test_cost_table_all_severities_present():
    for incident_type in COST_TABLE:
        assert set(COST_TABLE[incident_type].keys()) == {"low", "medium", "high"}

def test_cost_table_tuples_are_ordered():
    for incident_type, severities in COST_TABLE.items():
        for level, (low, high) in severities.items():
            assert low < high, f"{incident_type}/{level}: low must be < high"

def test_cost_table_severity_ranges_escalate():
    for incident_type, severities in COST_TABLE.items():
        assert severities["low"][1] <= severities["medium"][0] or \
               severities["low"][0] < severities["medium"][0], \
               f"{incident_type}: medium range should start higher than low"

# estimate_costs: normal cases

def test_estimate_costs_water_damage_medium():
    result = estimate_costs("water_damage", "medium", ceiling=25000, deductible=150)
    assert result["cost_range"] == (1500, 8000)
    assert result["compensable_amount"] == (1350, 7850)
    assert result["deductible_applied"] == 150
    assert result["ceiling_applied"] == 25000

def test_estimate_costs_fire_high():
    result = estimate_costs("fire", "high", ceiling=100000, deductible=300)
    assert result["cost_range"] == (40000, 90000)
    assert result["compensable_amount"] == (39700, 89700)

def test_estimate_costs_theft_low():
    result = estimate_costs("theft", "low", ceiling=20000, deductible=200)
    assert result["cost_range"] == (200, 2000)
    assert result["compensable_amount"] == (0, 1800)

def test_estimate_costs_deductible_exceeds_low_cost():
    # water_damage low: (200, 1500), deductible=150 → comp_low = max(0, 200-150) = 50
    result = estimate_costs("water_damage", "low", ceiling=25000, deductible=150)
    assert result["compensable_amount"][0] == 50

def test_estimate_costs_ceiling_caps_high_end():
    # fire high: (40000, 90000), ceiling=50000 → comp_high = min(50000, 90000-300) = 50000
    result = estimate_costs("fire", "high", ceiling=50000, deductible=300)
    assert result["compensable_amount"][1] == 50000

def test_estimate_costs_unknown_severity_returns_zeros():
    result = estimate_costs("water_damage", "unknown", ceiling=25000, deductible=150)
    assert result["cost_range"] == (0, 0)
    assert result["compensable_amount"] == (0, 0)

def test_estimate_costs_unknown_incident_type_returns_zeros():
    result = estimate_costs("earthquake", "medium", ceiling=80000, deductible=380)
    assert result["cost_range"] == (0, 0)
    assert result["compensable_amount"] == (0, 0)

def test_estimate_costs_preserves_deductible_and_ceiling():
    result = estimate_costs("water_damage", "unknown", ceiling=25000, deductible=150)
    assert result["deductible_applied"] == 150
    assert result["ceiling_applied"] == 25000

from src.agents.expertise.prompts import EXPERTISE_SUMMARY_PROMPT

def test_expertise_summary_prompt_has_required_placeholders():
    required = [
        "{incident_type}", "{description}", "{date}", "{severity}",
        "{cost_low}", "{cost_high}", "{comp_low}", "{comp_high}",
        "{deductible}", "{ceiling}",
    ]
    for placeholder in required:
        assert placeholder in EXPERTISE_SUMMARY_PROMPT, \
            f"Missing placeholder: {placeholder}"

def test_expertise_summary_prompt_delegates_to_advisor():
    lower = EXPERTISE_SUMMARY_PROMPT.lower()
    assert "conseiller" in lower
