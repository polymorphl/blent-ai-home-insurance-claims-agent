from src.agents.validation.agent import build_graph
from tests.agents.validation.conftest import FIXED_TODAY


def _run(claim, today=FIXED_TODAY):
    app = build_graph()
    return app.invoke({
        "claim": claim,
        "conformity_errors": [],
        "coverage_errors": [],
        "verdict": None,
        "today_override": today,
    })


def test_approved_claim_returns_approved(approved_claim):
    result = _run(approved_claim)
    assert result["verdict"]["status"] == "approved"
    assert result["verdict"]["coverage"]["ceiling"] == 100000
    assert result["verdict"]["coverage"]["deductible"] == 300
    assert result["verdict"]["claim"] == approved_claim


def test_rejected_missing_field_returns_rejected(rejected_claim_missing_field):
    result = _run(rejected_claim_missing_field)
    assert result["verdict"]["status"] == "rejected"
    assert "date" in result["verdict"]["reason"]
    assert result["verdict"]["coverage"] is None


def test_rejected_no_photos_returns_rejected(rejected_claim_no_photos):
    result = _run(rejected_claim_no_photos)
    assert result["verdict"]["status"] == "rejected"
    assert "photo" in result["verdict"]["reason"].lower()


def test_rejected_deadline_exceeded(rejected_claim_deadline):
    result = _run(rejected_claim_deadline)
    assert result["verdict"]["status"] == "rejected"
    assert "Délai" in result["verdict"]["reason"]


def test_approved_claim_includes_original_claim(approved_claim):
    result = _run(approved_claim)
    assert result["verdict"]["claim"]["incident_type"] == "fire"


def test_approved_water_damage_coverage(approved_claim):
    approved_claim["incident_type"] = "water_damage"
    result = _run(approved_claim)
    assert result["verdict"]["status"] == "approved"
    assert result["verdict"]["coverage"]["ceiling"] == 25000
    assert result["verdict"]["coverage"]["deductible"] == 150
