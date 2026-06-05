from unittest.mock import MagicMock, patch
from src.agents.validation.vlm_inference import VLMInference, SEVERITY_PROMPTS


def test_severity_prompts_cover_all_incident_types():
    assert "water_damage" in SEVERITY_PROMPTS
    assert "fire" in SEVERITY_PROMPTS
    assert "theft" in SEVERITY_PROMPTS


def test_severity_prompts_request_single_word():
    for prompt in SEVERITY_PROMPTS.values():
        lower = prompt.lower()
        assert "low" in lower and "medium" in lower and "high" in lower


def _make_vlm_with_mock_output(decoded_response: str):
    vlm = VLMInference.__new__(VLMInference)
    vlm.processor = MagicMock()
    vlm.model = MagicMock()
    vlm.model.device = "cpu"

    mock_output = MagicMock()
    mock_output.__getitem__ = MagicMock(return_value=MagicMock())
    vlm.model.generate.return_value = mock_output

    vlm.processor.apply_chat_template.return_value = "prompt text"
    vlm.processor.tokenizer.eos_token_id = 0

    mock_inputs = {"input_ids": MagicMock()}
    mock_inputs["input_ids"].shape = [1, 10]
    vlm.processor.return_value = mock_inputs

    vlm.processor.tokenizer.decode.return_value = decoded_response
    return vlm


def test_assess_damage_severity_returns_low(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"fake")
    vlm = _make_vlm_with_mock_output("low")
    with patch("src.agents.validation.vlm_inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = vlm.assess_damage_severity(str(fake_image), "water_damage")
    assert result == "low"


def test_assess_damage_severity_returns_medium(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"fake")
    vlm = _make_vlm_with_mock_output("medium")
    with patch("src.agents.validation.vlm_inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = vlm.assess_damage_severity(str(fake_image), "fire")
    assert result == "medium"


def test_assess_damage_severity_returns_high(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"fake")
    vlm = _make_vlm_with_mock_output("high")
    with patch("src.agents.validation.vlm_inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = vlm.assess_damage_severity(str(fake_image), "theft")
    assert result == "high"


def test_assess_damage_severity_defaults_to_medium_on_unrecognized(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"fake")
    vlm = _make_vlm_with_mock_output("severe")
    with patch("src.agents.validation.vlm_inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = vlm.assess_damage_severity(str(fake_image), "water_damage")
    assert result == "medium"


def test_assess_damage_severity_unknown_incident_type_returns_unknown(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"fake")
    vlm = VLMInference.__new__(VLMInference)
    vlm.processor = MagicMock()
    vlm.model = MagicMock()
    with patch("src.agents.validation.vlm_inference.Image"):
        result = vlm.assess_damage_severity(str(fake_image), "earthquake")
    assert result == "unknown"
