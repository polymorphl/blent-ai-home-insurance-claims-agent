import pytest

FIXED_TODAY = "2026-06-04"

@pytest.fixture
def approved_verdict():
    return {
        "status": "approved",
        "reason": "Dossier validé.",
        "coverage": {"ceiling": 25000, "deductible": 150},
        "claim": {
            "date": "2026-06-02",
            "incident_type": "water_damage",
            "description": "fuite dans la cuisine, mur infiltré d'eau, peinture qui se détache",
            "has_photos": True,
            "photo_filenames": [],
        },
    }

@pytest.fixture
def approved_verdict_fire():
    return {
        "status": "approved",
        "reason": "Dossier validé.",
        "coverage": {"ceiling": 100000, "deductible": 300},
        "claim": {
            "date": "2026-06-02",
            "incident_type": "fire",
            "description": "Incendie dans la chambre.",
            "has_photos": True,
            "photo_filenames": [],
        },
    }

@pytest.fixture
def approved_verdict_theft():
    return {
        "status": "approved",
        "reason": "Dossier validé.",
        "coverage": {"ceiling": 20000, "deductible": 200},
        "claim": {
            "date": "2026-06-03",
            "incident_type": "theft",
            "description": "Cambriolage, appareils électroniques volés.",
            "has_photos": True,
            "photo_filenames": [],
        },
    }
