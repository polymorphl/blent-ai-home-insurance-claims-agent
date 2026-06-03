from src.agents.declaration.tools import merge_claim_data, completeness_score, REQUIRED_FIELDS


def test_merge_keeps_existing_non_none():
    existing = {"date": "2025-03-15", "incident_type": "theft", "description": None, "has_photos": None}
    new = {"date": None, "incident_type": None, "description": "Appareils volés.", "has_photos": None}
    result = merge_claim_data(existing, new)
    assert result["date"] == "2025-03-15"
    assert result["incident_type"] == "theft"
    assert result["description"] == "Appareils volés."
    assert result["has_photos"] is None


def test_merge_overwrites_none_with_value():
    existing = {"date": None, "incident_type": None, "description": None, "has_photos": None}
    new = {"date": "2025-09-10", "incident_type": "fire", "description": "Incendie.", "has_photos": True}
    assert merge_claim_data(existing, new) == {
        "date": "2025-09-10",
        "incident_type": "fire",
        "description": "Incendie.",
        "has_photos": True,
    }


def test_merge_with_empty_existing():
    result = merge_claim_data({}, {"date": "2025-01-01", "incident_type": "theft", "description": None, "has_photos": None})
    assert result["date"] == "2025-01-01"
    assert result["incident_type"] == "theft"


def test_completeness_score_all_present():
    claim = {"date": "2025-09-10", "incident_type": "fire", "description": "Incendie.", "has_photos": True}
    assert completeness_score(claim) == 1.0


def test_completeness_score_all_missing():
    assert completeness_score({"date": None, "incident_type": None, "description": None, "has_photos": None}) == 0.0


def test_completeness_score_partial():
    claim = {"date": "2025-03-15", "incident_type": "theft", "description": None, "has_photos": None}
    assert completeness_score(claim) == 0.5


def test_required_fields_has_four_fields():
    assert set(REQUIRED_FIELDS) == {"date", "incident_type", "description", "has_photos"}
