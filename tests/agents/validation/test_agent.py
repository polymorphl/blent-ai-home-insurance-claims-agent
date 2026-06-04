from pathlib import Path
from unittest.mock import MagicMock

from src.agents.validation.agent import build_graph
from tests.agents.validation.conftest import FIXED_TODAY


def _run(claim, today=FIXED_TODAY, vlm_inference=None, attachments_dir=None):
    app = build_graph(vlm_inference=vlm_inference, attachments_dir=attachments_dir)
    return app.invoke({
        "claim": claim,
        "conformity_errors": [],
        "coverage_errors": [],
        "photo_errors": [],
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


def test_rejected_unknown_incident_type():
    claim = {
        "date": "2026-06-03",
        "incident_type": "earthquake",
        "description": "Tremblement de terre.",
        "has_photos": True,
    }
    result = _run(claim)
    assert result["verdict"]["status"] == "rejected"
    assert "contrat" in result["verdict"]["reason"].lower()


def test_photo_check_skipped_when_no_vlm(approved_claim):
    approved_claim["photo_filenames"] = ["WaterDamage_100.jpg"]
    result = _run(approved_claim, vlm_inference=None)
    assert result["verdict"]["status"] == "approved"


def test_photo_check_skipped_when_no_filenames(approved_claim):
    approved_claim["photo_filenames"] = []
    mock_vlm = MagicMock()
    result = _run(approved_claim, vlm_inference=mock_vlm)
    mock_vlm.check_photo_coherence.assert_not_called()
    assert result["verdict"]["status"] == "approved"


def test_photo_check_ignores_absent_file(approved_claim, tmp_path):
    approved_claim["photo_filenames"] = ["nonexistent.jpg"]
    mock_vlm = MagicMock()
    result = _run(approved_claim, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    mock_vlm.check_photo_coherence.assert_not_called()
    assert result["verdict"]["status"] == "approved"


def test_photo_check_approved_when_majority_match(approved_claim, tmp_path):
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        (tmp_path / name).write_bytes(b"fake")
    approved_claim["photo_filenames"] = ["a.jpg", "b.jpg", "c.jpg"]
    mock_vlm = MagicMock()
    mock_vlm.check_photo_coherence.side_effect = [True, True, False]  # 2/3 = 67%
    result = _run(approved_claim, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    assert result["verdict"]["status"] == "approved"


def test_photo_check_rejected_when_exactly_half_match(approved_claim, tmp_path):
    for name in ["a.jpg", "b.jpg"]:
        (tmp_path / name).write_bytes(b"fake")
    approved_claim["photo_filenames"] = ["a.jpg", "b.jpg"]
    mock_vlm = MagicMock()
    mock_vlm.check_photo_coherence.side_effect = [True, False]  # 1/2 = 50% → rejected
    result = _run(approved_claim, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    assert result["verdict"]["status"] == "rejected"
    assert "photo" in result["verdict"]["reason"].lower()


def test_photo_check_rejected_when_none_match(approved_claim, tmp_path):
    for name in ["a.jpg", "b.jpg"]:
        (tmp_path / name).write_bytes(b"fake")
    approved_claim["photo_filenames"] = ["a.jpg", "b.jpg"]
    mock_vlm = MagicMock()
    mock_vlm.check_photo_coherence.side_effect = [False, False]
    result = _run(approved_claim, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    assert result["verdict"]["status"] == "rejected"


def test_photo_check_skipped_when_coverage_fails(rejected_claim_deadline):
    rejected_claim_deadline["photo_filenames"] = ["a.jpg"]
    mock_vlm = MagicMock()
    result = _run(rejected_claim_deadline, vlm_inference=mock_vlm)
    mock_vlm.check_photo_coherence.assert_not_called()
    assert result["verdict"]["status"] == "rejected"
    assert "Délai" in result["verdict"]["reason"]
