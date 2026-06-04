from src.agents.validation.state import ValidationState


def test_validation_state_has_required_keys():
    state: ValidationState = {
        "claim": {"date": "2025-09-10", "incident_type": "fire",
                  "description": "Incendie.", "has_photos": True},
        "conformity_errors": [],
        "coverage_errors": [],
        "verdict": None,
        "today_override": None,
    }
    assert state["claim"]["incident_type"] == "fire"
    assert state["verdict"] is None
