from unittest.mock import MagicMock, patch

import torch

from src.inference import (
    UnifiedInference,
    COHERENCE_PROMPTS,
    SEVERITY_PROMPTS,
)


# --------------------------------------------------------------------------- #
# Module-level constants                                                        #
# --------------------------------------------------------------------------- #

def test_coherence_prompts_cover_required_types():
    for t in ("water_damage", "fire", "theft", "mold", "natural_disaster"):
        assert t in COHERENCE_PROMPTS

def test_coherence_prompts_are_yes_no_questions():
    for prompt in COHERENCE_PROMPTS.values():
        assert "yes or no" in prompt.lower()

def test_severity_prompts_cover_incident_types():
    for t in ("water_damage", "fire", "theft"):
        assert t in SEVERITY_PROMPTS

def test_severity_prompts_contain_levels():
    for prompt in SEVERITY_PROMPTS.values():
        lower = prompt.lower()
        assert "low" in lower and "medium" in lower and "high" in lower


# --------------------------------------------------------------------------- #
# _parse_tool_call — no model needed                                            #
# --------------------------------------------------------------------------- #

def test_parse_tool_call_qwen():
    inf = UnifiedInference.__new__(UnifiedInference)
    output = (
        '<tool_call>{"name": "extract_claim_fields", "arguments": '
        '{"date": "2025-09-10", "incident_type": "fire", '
        '"description": "Incendie dans la chambre.", "has_photos": true}}</tool_call>'
    )
    result = inf._parse_tool_call(output)
    assert result == {
        "date": "2025-09-10",
        "incident_type": "fire",
        "description": "Incendie dans la chambre.",
        "has_photos": True,
    }

def test_parse_tool_call_python_tag():
    inf = UnifiedInference.__new__(UnifiedInference)
    output = (
        '<|python_tag|>{"name": "extract_claim_fields", "parameters": '
        '{"date": "2025-09-10", "incident_type": "fire", '
        '"description": "Incendie dans la chambre.", "has_photos": true}}'
        "<|eom_id|>"
    )
    result = inf._parse_tool_call(output)
    assert result == {
        "date": "2025-09-10",
        "incident_type": "fire",
        "description": "Incendie dans la chambre.",
        "has_photos": True,
    }

def test_parse_tool_call_fallback_on_unparseable():
    inf = UnifiedInference.__new__(UnifiedInference)
    result = inf._parse_tool_call("Je ne peux pas déterminer ces informations.")
    assert result == {"date": None, "incident_type": None, "description": None, "has_photos": None}

def test_parse_tool_call_fallback_json_block():
    inf = UnifiedInference.__new__(UnifiedInference)
    output = '{"date": "2025-03-15", "incident_type": "water_damage", "description": "Fuite.", "has_photos": true}'
    result = inf._parse_tool_call(output)
    assert result["date"] == "2025-03-15"


# --------------------------------------------------------------------------- #
# Test helpers                                                                  #
# --------------------------------------------------------------------------- #

def _make_llm_inf(decoded_response: str) -> UnifiedInference:
    """Build a UnifiedInference with mocked LLM attributes (no VLM)."""
    inf = UnifiedInference.__new__(UnifiedInference)
    inf._llm_tokenizer = MagicMock()
    inf._llm_model = MagicMock()
    inf._llm_model.device = "cpu"
    inf._vlm_model = None
    inf._vlm_processor = None

    mock_output = MagicMock()
    mock_output.__getitem__ = MagicMock(return_value=MagicMock())
    inf._llm_model.generate.return_value = mock_output

    inf._llm_tokenizer.apply_chat_template.return_value = "prompt text"
    inf._llm_tokenizer.eos_token_id = 0

    mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_inputs["input_ids"].shape = [1, 10]
    inf._llm_tokenizer.return_value = mock_inputs

    inf._llm_tokenizer.decode.return_value = decoded_response
    return inf


def _make_vlm_inf(decoded_response: str) -> UnifiedInference:
    """Build a UnifiedInference with mocked VLM attributes (no LLM)."""
    inf = UnifiedInference.__new__(UnifiedInference)
    inf._llm_model = None
    inf._llm_tokenizer = None
    inf._vlm_processor = MagicMock()
    inf._vlm_model = MagicMock()
    inf._vlm_model.device = "cpu"

    mock_output = MagicMock()
    mock_output.__getitem__ = MagicMock(return_value=MagicMock())
    inf._vlm_model.generate.return_value = mock_output

    inf._vlm_processor.apply_chat_template.return_value = "prompt text"
    inf._vlm_processor.tokenizer.eos_token_id = 0

    mock_inputs = {"input_ids": MagicMock()}
    mock_inputs["input_ids"].shape = [1, 10]
    inf._vlm_processor.return_value = mock_inputs

    inf._vlm_processor.tokenizer.decode.return_value = decoded_response
    return inf


# --------------------------------------------------------------------------- #
# Text-only methods                                                             #
# --------------------------------------------------------------------------- #

def test_generate_returns_stripped_string():
    inf = _make_llm_inf("Voici la réponse.")
    result = inf.generate([{"role": "user", "content": "Décris les dégâts."}])
    assert result == "Voici la réponse."


def test_extract_returns_parsed_dict():
    tool_output = (
        '<tool_call>{"name": "extract_claim_fields", "arguments": '
        '{"date": "2025-03-15", "incident_type": "water_damage", '
        '"description": "Fuite.", "has_photos": true}}</tool_call>'
    )
    inf = _make_llm_inf(tool_output)
    result = inf.extract([{"role": "user", "content": "J'ai une fuite."}])
    assert result["date"] == "2025-03-15"
    assert result["incident_type"] == "water_damage"


def test_extract_decodes_without_skip_special_tokens():
    inf = _make_llm_inf("<tool_call>{}</tool_call>")
    inf.extract([{"role": "user", "content": "test"}])
    decode_call = inf._llm_tokenizer.decode.call_args
    assert decode_call.kwargs.get("skip_special_tokens") is False


def test_tokenize_text_passes_tools_when_provided():
    inf = _make_llm_inf("ok")
    inf._tokenize_text([{"role": "user", "content": "test"}], tools=[{"type": "function"}])
    call_kwargs = inf._llm_tokenizer.apply_chat_template.call_args.kwargs
    assert "tools" in call_kwargs


def test_tokenize_text_omits_tools_when_none():
    inf = _make_llm_inf("ok")
    inf._tokenize_text([{"role": "user", "content": "test"}])
    call_kwargs = inf._llm_tokenizer.apply_chat_template.call_args.kwargs
    assert "tools" not in call_kwargs


# --------------------------------------------------------------------------- #
# Vision methods                                                                #
# --------------------------------------------------------------------------- #

def test_check_photo_coherence_returns_bool(tmp_path):
    fake = tmp_path / "test.jpg"
    fake.write_bytes(b"fake")
    inf = _make_vlm_inf("yes")
    with patch("src.inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = inf.check_photo_coherence(str(fake), "water_damage")
    assert isinstance(result, bool)
    assert result is True


def test_check_photo_coherence_returns_false_for_no(tmp_path):
    fake = tmp_path / "test.jpg"
    fake.write_bytes(b"fake")
    inf = _make_vlm_inf("no")
    with patch("src.inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = inf.check_photo_coherence(str(fake), "fire")
    assert result is False


def test_check_photo_coherence_unknown_type_returns_true(tmp_path):
    fake = tmp_path / "test.jpg"
    fake.write_bytes(b"fake")
    inf = UnifiedInference.__new__(UnifiedInference)
    inf._llm_model = None
    inf._llm_tokenizer = None
    inf._vlm_model = None
    inf._vlm_processor = None
    with patch("src.inference.Image"):
        result = inf.check_photo_coherence(str(fake), "earthquake")
    assert result is True


def test_assess_damage_severity_returns_low(tmp_path):
    fake = tmp_path / "test.jpg"
    fake.write_bytes(b"fake")
    inf = _make_vlm_inf("low")
    with patch("src.inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = inf.assess_damage_severity(str(fake), "water_damage")
    assert result == "low"


def test_assess_damage_severity_defaults_to_medium_on_unrecognized(tmp_path):
    fake = tmp_path / "test.jpg"
    fake.write_bytes(b"fake")
    inf = _make_vlm_inf("severe")
    with patch("src.inference.Image") as mock_image:
        mock_image.open.return_value.convert.return_value = MagicMock()
        result = inf.assess_damage_severity(str(fake), "water_damage")
    assert result == "medium"


def test_assess_damage_severity_unknown_incident_returns_unknown(tmp_path):
    fake = tmp_path / "test.jpg"
    fake.write_bytes(b"fake")
    inf = UnifiedInference.__new__(UnifiedInference)
    inf._llm_model = None
    inf._llm_tokenizer = None
    inf._vlm_model = None
    inf._vlm_processor = None
    with patch("src.inference.Image"):
        result = inf.assess_damage_severity(str(fake), "earthquake")
    assert result == "unknown"


# --------------------------------------------------------------------------- #
# Swap behavior                                                                 #
# --------------------------------------------------------------------------- #

def test_ensure_llm_noop_if_already_loaded():
    inf = UnifiedInference.__new__(UnifiedInference)
    existing = MagicMock()
    inf._llm_model = existing
    inf._llm_tokenizer = MagicMock()
    inf._vlm_model = None
    inf._vlm_processor = None
    inf._ensure_llm()
    assert inf._llm_model is existing  # not reloaded


def test_ensure_vlm_noop_if_already_loaded():
    inf = UnifiedInference.__new__(UnifiedInference)
    inf._llm_model = None
    inf._llm_tokenizer = None
    existing = MagicMock()
    inf._vlm_model = existing
    inf._vlm_processor = MagicMock()
    inf._ensure_vlm()
    assert inf._vlm_model is existing  # not reloaded


def test_ensure_llm_unloads_vlm():
    inf = UnifiedInference.__new__(UnifiedInference)
    inf._llm_model = None
    inf._llm_tokenizer = None
    inf._vlm_model = MagicMock()
    inf._vlm_processor = MagicMock()

    with patch("src.inference.AutoTokenizer") as mock_tok, \
         patch("src.inference.AutoModelForCausalLM") as mock_causal:
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_m = MagicMock()
        mock_m.to.return_value = mock_m
        mock_causal.from_pretrained.return_value = mock_m

        inf._ensure_llm()

    assert inf._vlm_model is None
    assert inf._vlm_processor is None
    assert inf._llm_model is not None


def test_ensure_vlm_unloads_llm():
    inf = UnifiedInference.__new__(UnifiedInference)
    inf._llm_model = MagicMock()
    inf._llm_tokenizer = MagicMock()
    inf._vlm_model = None
    inf._vlm_processor = None

    with patch("src.inference.AutoProcessor") as mock_proc, \
         patch("src.inference.Qwen2_5_VLForConditionalGeneration") as mock_vlm:
        mock_proc.from_pretrained.return_value = MagicMock()
        mock_m = MagicMock()
        mock_m.to.return_value = mock_m
        mock_vlm.from_pretrained.return_value = mock_m

        inf._ensure_vlm()

    assert inf._llm_model is None
    assert inf._llm_tokenizer is None
    assert inf._vlm_model is not None
