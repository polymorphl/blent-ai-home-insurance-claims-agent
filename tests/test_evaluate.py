from scripts.evaluate_declaration import evaluate_example


def test_evaluate_example_all_fields_present():
    result = {
        "final_claim": {
            "date": "2025-09-10",
            "incident_type": "fire",
            "description": "Incendie.",
            "has_photos": True,
            "photo_filenames": [],
            "conversation_history": [],
            "turns_count": 1,
        }
    }
    expected = {
        "date": "2025-09-10",
        "incident_type": "fire",
        "description": "Incendie.",
        "has_photos": True,
    }
    metrics = evaluate_example(result, expected)
    assert metrics["score"] == 1.0
    assert metrics["turns_count"] == 1
    assert all(metrics["per_field"].values())


def test_evaluate_example_partial_fields():
    result = {
        "final_claim": {
            "date": "2025-09-10",
            "incident_type": "theft",
            "description": None,
            "has_photos": None,
            "photo_filenames": [],
            "conversation_history": [],
            "turns_count": 2,
        }
    }
    expected = {"date": "2025-09-10", "incident_type": "theft", "description": "...", "has_photos": True}
    metrics = evaluate_example(result, expected)
    assert metrics["score"] == 0.5
    assert metrics["per_field"]["date"] is True
    assert metrics["per_field"]["description"] is False


def test_evaluate_example_no_final_claim():
    result = {"final_claim": None}
    expected = {"date": "2025-09-10", "incident_type": "fire", "description": "...", "has_photos": True}
    metrics = evaluate_example(result, expected)
    assert metrics["score"] == 0.0
