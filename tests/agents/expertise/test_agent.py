from pathlib import Path
from unittest.mock import MagicMock

from src.agents.expertise.agent import build_graph


def _run(verdict, vlm_inference=None, inference=None, attachments_dir=None):
    app = build_graph(vlm_inference=vlm_inference, inference=inference, attachments_dir=attachments_dir)
    return app.invoke({
        "verdict": verdict,
        "severity": None,
        "cost_range": None,
        "compensable_amount": None,
        "deductible_applied": None,
        "ceiling_applied": None,
        "summary": None,
        "report": None,
    })


def test_report_has_all_required_keys(approved_verdict):
    result = _run(approved_verdict)
    report = result["report"]
    for key in ("severity", "cost_range", "compensable_amount",
                "deductible_applied", "ceiling_applied", "summary", "claim"):
        assert key in report, f"Missing key: {key}"


def test_no_vlm_produces_unknown_severity(approved_verdict):
    result = _run(approved_verdict, vlm_inference=None)
    assert result["report"]["severity"] == "unknown"


def test_no_vlm_produces_zero_cost_range(approved_verdict):
    result = _run(approved_verdict, vlm_inference=None)
    assert result["report"]["cost_range"] == (0, 0)
    assert result["report"]["compensable_amount"] == (0, 0)


def test_no_photos_produces_unknown_severity(approved_verdict, tmp_path):
    mock_vlm = MagicMock()
    result = _run(approved_verdict, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    mock_vlm.assess_damage_severity.assert_not_called()
    assert result["report"]["severity"] == "unknown"


def test_photo_file_not_found_produces_unknown(approved_verdict, tmp_path):
    approved_verdict["claim"]["photo_filenames"] = ["nonexistent.jpg"]
    mock_vlm = MagicMock()
    result = _run(approved_verdict, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    mock_vlm.assess_damage_severity.assert_not_called()
    assert result["report"]["severity"] == "unknown"


def test_vlm_medium_water_damage_correct_costs(approved_verdict, tmp_path):
    fake = tmp_path / "photo.jpg"
    fake.write_bytes(b"fake")
    approved_verdict["claim"]["photo_filenames"] = ["photo.jpg"]
    mock_vlm = MagicMock()
    mock_vlm.assess_damage_severity.return_value = "medium"
    result = _run(approved_verdict, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    assert result["report"]["severity"] == "medium"
    assert result["report"]["cost_range"] == (1500, 8000)
    assert result["report"]["compensable_amount"] == (1350, 7850)


def test_vlm_majority_rule_two_low_one_high(approved_verdict, tmp_path):
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        (tmp_path / name).write_bytes(b"fake")
    approved_verdict["claim"]["photo_filenames"] = ["a.jpg", "b.jpg", "c.jpg"]
    mock_vlm = MagicMock()
    mock_vlm.assess_damage_severity.side_effect = ["low", "low", "high"]
    result = _run(approved_verdict, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    assert result["report"]["severity"] == "low"


def test_vlm_fire_high_correct_costs(approved_verdict_fire, tmp_path):
    fake = tmp_path / "photo.jpg"
    fake.write_bytes(b"fake")
    approved_verdict_fire["claim"]["photo_filenames"] = ["photo.jpg"]
    mock_vlm = MagicMock()
    mock_vlm.assess_damage_severity.return_value = "high"
    result = _run(approved_verdict_fire, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    assert result["report"]["cost_range"] == (40000, 90000)
    assert result["report"]["compensable_amount"] == (39700, 89700)


def test_deductible_and_ceiling_in_report(approved_verdict):
    result = _run(approved_verdict)
    assert result["report"]["deductible_applied"] == 150
    assert result["report"]["ceiling_applied"] == 25000


def test_claim_passthrough_in_report(approved_verdict):
    result = _run(approved_verdict)
    assert result["report"]["claim"]["incident_type"] == "water_damage"


def test_no_inference_produces_fallback_summary(approved_verdict):
    result = _run(approved_verdict, inference=None)
    assert isinstance(result["report"]["summary"], str)
    assert len(result["report"]["summary"]) > 0


def test_inference_called_with_messages(approved_verdict):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Résumé généré."
    result = _run(approved_verdict, inference=mock_llm)
    mock_llm.generate.assert_called_once()
    assert result["report"]["summary"] == "Résumé généré."


def test_vlm_tie_picks_higher_severity(approved_verdict, tmp_path):
    for name in ["a.jpg", "b.jpg"]:
        (tmp_path / name).write_bytes(b"fake")
    approved_verdict["claim"]["photo_filenames"] = ["a.jpg", "b.jpg"]
    mock_vlm = MagicMock()
    mock_vlm.assess_damage_severity.side_effect = ["low", "high"]  # tie → high wins
    result = _run(approved_verdict, vlm_inference=mock_vlm, attachments_dir=tmp_path)
    assert result["report"]["severity"] == "high"


def test_rejected_verdict_does_not_crash(tmp_path):
    rejected_verdict = {
        "status": "rejected",
        "reason": "Délai dépassé.",
        "coverage": None,
        "claim": {
            "date": "2026-05-01",
            "incident_type": "water_damage",
            "description": "Fuite.",
            "has_photos": True,
            "photo_filenames": [],
        },
    }
    result = _run(rejected_verdict)
    assert result["report"]["cost_range"] == (0, 0)
    assert result["report"]["compensable_amount"] == (0, 0)
