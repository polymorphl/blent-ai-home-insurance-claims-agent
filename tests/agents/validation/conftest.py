import pytest
from datetime import date


@pytest.fixture
def approved_claim():
    return {
        "date": "2026-06-03",
        "incident_type": "fire",
        "description": "Incendie dans la chambre.",
        "has_photos": True,
    }


@pytest.fixture
def rejected_claim_missing_field():
    return {
        "date": None,
        "incident_type": "fire",
        "description": "Incendie.",
        "has_photos": True,
    }


@pytest.fixture
def rejected_claim_no_photos():
    return {
        "date": "2026-06-03",
        "incident_type": "theft",
        "description": "Cambriolage.",
        "has_photos": False,
    }


@pytest.fixture
def rejected_claim_deadline():
    return {
        "date": "2026-05-20",
        "incident_type": "theft",
        "description": "Cambriolage.",
        "has_photos": True,
    }


FIXED_TODAY = "2026-06-04"
