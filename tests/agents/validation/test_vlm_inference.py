from unittest.mock import MagicMock, patch
from src.agents.validation.vlm_inference import VLMInference, COHERENCE_PROMPTS


def test_coherence_prompts_cover_all_incident_types():
    assert "water_damage" in COHERENCE_PROMPTS
    assert "fire" in COHERENCE_PROMPTS
    assert "theft" in COHERENCE_PROMPTS
    assert "mold" in COHERENCE_PROMPTS
    assert "natural_disaster" in COHERENCE_PROMPTS


def test_coherence_prompts_are_yes_no_questions():
    for prompt in COHERENCE_PROMPTS.values():
        assert "yes or no" in prompt.lower()


def test_check_photo_coherence_returns_bool(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"fake")

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

    vlm.processor.tokenizer.decode.return_value = "yes"

    with patch("src.inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = vlm.check_photo_coherence(str(fake_image), "water_damage")

    assert isinstance(result, bool)


def test_unknown_incident_type_returns_true(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"fake")

    vlm = VLMInference.__new__(VLMInference)
    vlm.processor = MagicMock()
    vlm.model = MagicMock()

    with patch("src.inference.Image"):
        result = vlm.check_photo_coherence(str(fake_image), "earthquake")

    assert result is True
