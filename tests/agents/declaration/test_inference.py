from src.agents.declaration.inference import HFInference


def test_parse_tool_call_python_tag():
    inf = HFInference.__new__(HFInference)
    output = (
        '<|python_tag|>{"name": "extract_claim_fields", "parameters": '
        '{"date": "2025-09-10", "incident_type": "fire", '
        '"description": "Incendie dans la chambre.", "has_photos": true}}'
        "<|eom_id|>"
    )
    result = inf._parse_tool_call(output)
    assert result == {
        "date": "2025-09-10",
        "incident_type": "fire",
        "description": "Incendie dans la chambre.",
        "has_photos": True,
    }


def test_parse_tool_call_fallback_on_unparseable():
    inf = HFInference.__new__(HFInference)
    result = inf._parse_tool_call("Je ne peux pas déterminer ces informations.")
    assert result == {"date": None, "incident_type": None, "description": None, "has_photos": None}


def test_parse_tool_call_fallback_json_block():
    inf = HFInference.__new__(HFInference)
    output = '{"date": "2025-03-15", "incident_type": "water_damage", "description": "Fuite.", "has_photos": true}'
    result = inf._parse_tool_call(output)
    assert result["date"] == "2025-03-15"
    assert result["incident_type"] == "water_damage"
