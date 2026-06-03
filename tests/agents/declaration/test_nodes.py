from langchain_core.messages import HumanMessage, AIMessage
from src.agents.declaration.agent import (
    check_completeness,
    finalize,
    make_extract_fields_node,
    make_generate_follow_up_node,
)


def test_check_completeness_all_present(complete_state):
    result = check_completeness(complete_state)
    assert result["is_complete"] is True
    assert result["missing_fields"] == []


def test_check_completeness_missing_date_and_photos(incomplete_state):
    result = check_completeness(incomplete_state)
    assert result["is_complete"] is False
    assert "date" in result["missing_fields"]
    assert "has_photos" in result["missing_fields"]


def test_check_completeness_empty_claim():
    state = {
        "messages": [],
        "claim_data": {},
        "missing_fields": [],
        "is_complete": False,
        "final_claim": None,
    }
    result = check_completeness(state)
    assert result["is_complete"] is False
    assert set(result["missing_fields"]) == {"date", "incident_type", "description", "has_photos"}


def test_finalize_extracts_photo_filenames(complete_state):
    complete_state["messages"] = [
        HumanMessage(content="Bonjour\n[Pièce jointe : photo1.jpg]\n[Pièce jointe : photo2.jpg]")
    ]
    result = finalize(complete_state)
    claim = result["final_claim"]
    assert claim["date"] == "2025-09-10"
    assert claim["incident_type"] == "fire"
    assert "photo1.jpg" in claim["photo_filenames"]
    assert "photo2.jpg" in claim["photo_filenames"]
    assert claim["turns_count"] == 1


def test_finalize_no_attachments(complete_state):
    result = finalize(complete_state)
    assert result["final_claim"]["photo_filenames"] == []


def test_finalize_conversation_history(complete_state):
    complete_state["messages"] = [
        HumanMessage(content="Déclaration"),
        AIMessage(content="Merci, pouvez-vous préciser la date ?"),
        HumanMessage(content="Le 10/09/2025."),
    ]
    result = finalize(complete_state)
    history = result["final_claim"]["conversation_history"]
    assert len(history) == 3
    assert history[0]["role"] == "human"
    assert history[1]["role"] == "ai"
    assert result["final_claim"]["turns_count"] == 2


def test_extract_fields_updates_claim_data(mock_inference, incomplete_state):
    incomplete_state["claim_data"] = {}
    node = make_extract_fields_node(mock_inference)
    result = node(incomplete_state)
    assert result["claim_data"]["incident_type"] == "theft"
    assert result["claim_data"]["description"] is not None


def test_extract_fields_merges_with_existing(mock_inference, incomplete_state):
    incomplete_state["claim_data"] = {
        "date": "2025-09-10",
        "incident_type": None,
        "description": None,
        "has_photos": None,
    }
    mock_inference.extract.return_value = {
        "date": None,
        "incident_type": "theft",
        "description": "Vol de matériel.",
        "has_photos": None,
    }
    node = make_extract_fields_node(mock_inference)
    result = node(incomplete_state)
    assert result["claim_data"]["date"] == "2025-09-10"
    assert result["claim_data"]["incident_type"] == "theft"


def test_extract_fields_no_human_message(mock_inference):
    state = {
        "messages": [AIMessage(content="Bonjour, comment puis-je vous aider ?")],
        "claim_data": {},
        "missing_fields": [],
        "is_complete": False,
        "final_claim": None,
    }
    node = make_extract_fields_node(mock_inference)
    assert node(state) == {}


def test_generate_follow_up_returns_ai_message(mock_inference, incomplete_state):
    incomplete_state["missing_fields"] = ["date", "has_photos"]
    node = make_generate_follow_up_node(mock_inference)
    result = node(incomplete_state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert len(result["messages"][0].content) > 0
