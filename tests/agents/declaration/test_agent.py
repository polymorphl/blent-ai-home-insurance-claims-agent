from langchain_core.messages import HumanMessage, AIMessage
from src.agents.declaration.agent import build_graph


def test_graph_completes_with_full_declaration(mock_inference):
    mock_inference.extract.return_value = {
        "date": "2025-09-10",
        "incident_type": "fire",
        "description": "Incendie dans la chambre à cause d'un appareil défectueux.",
        "has_photos": True,
    }
    app = build_graph(mock_inference)
    config = {"configurable": {"thread_id": "test_complete"}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content="Le 10/09/2025 un feu s'est déclaré.\n[Pièce jointe : photo1.jpg]")],
            "claim_data": {},
            "missing_fields": [],
            "is_complete": False,
            "final_claim": None,
        },
        config,
    )
    assert result["is_complete"] is True
    assert result["final_claim"] is not None
    assert result["final_claim"]["incident_type"] == "fire"


def test_graph_requests_missing_info(mock_inference):
    mock_inference.extract.return_value = {
        "date": None,
        "incident_type": "theft",
        "description": "Appareils volés.",
        "has_photos": None,
    }
    app = build_graph(mock_inference)
    config = {"configurable": {"thread_id": "test_incomplete"}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content="On m'a cambriolé ce matin.")],
            "claim_data": {},
            "missing_fields": [],
            "is_complete": False,
            "final_claim": None,
        },
        config,
    )
    assert result["is_complete"] is False
    assert result["final_claim"] is None
    assert isinstance(result["messages"][-1], AIMessage)


def test_graph_multi_turn_reaches_completion(mock_inference):
    app = build_graph(mock_inference)
    config = {"configurable": {"thread_id": "test_multi_turn"}}

    mock_inference.extract.return_value = {
        "date": None,
        "incident_type": "theft",
        "description": "Appareils volés.",
        "has_photos": None,
    }
    result = app.invoke(
        {
            "messages": [HumanMessage(content="On m'a cambriolé ce matin.")],
            "claim_data": {},
            "missing_fields": [],
            "is_complete": False,
            "final_claim": None,
        },
        config,
    )
    assert result["is_complete"] is False

    mock_inference.extract.return_value = {
        "date": "2025-09-10",
        "incident_type": None,
        "description": None,
        "has_photos": True,
    }
    result = app.invoke(
        {"messages": [HumanMessage(content="Le 10/09/2025.\n[Pièce jointe : photo.jpg]")]},
        config,
    )
    assert result["is_complete"] is True
    assert result["final_claim"]["date"] == "2025-09-10"
    assert result["final_claim"]["turns_count"] == 2
