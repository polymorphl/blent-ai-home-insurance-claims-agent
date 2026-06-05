from src.agents.expertise.state import ExpertiseState

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
