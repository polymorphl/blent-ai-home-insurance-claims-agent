import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.declaration.inference import HFInference


@pytest.fixture
def mock_inference():
    inf = MagicMock(spec=HFInference)
    inf.extract.return_value = {
        "date": None,
        "incident_type": "theft",
        "description": "Les voleurs ont volé les appareils électroniques.",
        "has_photos": None,
    }
    inf.generate.return_value = (
        "Pourriez-vous nous indiquer la date du cambriolage "
        "et joindre des photos si vous en avez ?"
    )
    return inf


@pytest.fixture
def complete_state():
    return {
        "messages": [HumanMessage(content="Déclaration de sinistre")],
        "claim_data": {
            "date": "2025-09-10",
            "incident_type": "fire",
            "description": "Incendie dans la chambre.",
            "has_photos": True,
        },
        "missing_fields": [],
        "is_complete": False,
        "final_claim": None,
    }


@pytest.fixture
def incomplete_state():
    return {
        "messages": [HumanMessage(content="On m'a cambriolé ce matin.")],
        "claim_data": {
            "date": None,
            "incident_type": "theft",
            "description": "Les voleurs ont volé les appareils.",
            "has_photos": None,
        },
        "missing_fields": [],
        "is_complete": False,
        "final_claim": None,
    }
