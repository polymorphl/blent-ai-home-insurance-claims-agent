import json
import re

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from src.agents.declaration.tools import EXTRACT_TOOL_SCHEMA
from src.config import (
    DEVICE,
    HF_TOKEN,
    LLM_MODEL_NAME,
    MAX_NEW_TOKENS,
    MODEL_CACHE_DIR,
    TEMPERATURE,
    TORCH_DTYPE,
    VLM_MODEL_NAME,
)

COHERENCE_PROMPTS: dict[str, str] = {
    "water_damage": (
        "Does this image show water damage such as leaks, wet walls, flooding, "
        "or water stains? Answer only yes or no."
    ),
    "fire": (
        "Does this image show fire damage such as burns, charring, smoke damage, "
        "or burned objects? Answer only yes or no."
    ),
    "theft": (
        "Does this image show evidence of burglary or theft such as broken locks, "
        "forced entry, damaged doors, or ransacked rooms? Answer only yes or no."
    ),
    "mold": (
        "Does this image show mold or mildew damage on walls, ceilings, or surfaces? "
        "Answer only yes or no."
    ),
    "natural_disaster": (
        "Does this image show damage caused by a natural disaster such as flooding, "
        "storm, wind, or earthquake? Answer only yes or no."
    ),
}

SEVERITY_PROMPTS: dict[str, str] = {
    "water_damage": (
        "Assess the severity of water damage visible in this image. "
        "Consider the extent of wet surfaces, stains, flooding, or structural damage. "
        "Respond with exactly one word: low, medium, or high."
    ),
    "fire": (
        "Assess the severity of fire damage visible in this image. "
        "Consider the extent of burns, charring, smoke damage, or structural destruction. "
        "Respond with exactly one word: low, medium, or high."
    ),
    "theft": (
        "Assess the severity of theft or burglary damage visible in this image. "
        "Consider forced entry marks, ransacking, broken locks, or property destruction. "
        "Respond with exactly one word: low, medium, or high."
    ),
}


class UnifiedInference:
    """Inference wrapper with on-demand model swapping: LLM for text, VLM for vision."""

    def __init__(self):
        """Initialize state and pre-load LLM (Declaration always runs first)."""
        self._llm_model = None
        self._llm_tokenizer = None
        self._vlm_model = None
        self._vlm_processor = None
        self._ensure_llm()

    def _ensure_llm(self) -> None:
        """Load LLM into VRAM, unloading VLM first if necessary."""
        if self._llm_model is not None:
            return
        if self._vlm_model is not None:
            self._vlm_model = None
            self._vlm_processor = None
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        print(f"⌛ Loading LLM {LLM_MODEL_NAME}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME, token=HF_TOKEN, cache_dir=MODEL_CACHE_DIR
            )
            load_kwargs = {
                "dtype": getattr(torch, TORCH_DTYPE),
                "token": HF_TOKEN,
                "cache_dir": MODEL_CACHE_DIR,
            }
            if DEVICE == "cuda":
                load_kwargs["device_map"] = {"": 0}
            model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, **load_kwargs)
            if DEVICE != "cuda":
                model = model.to(DEVICE)
            model.eval()
        except Exception:
            self._llm_tokenizer = None
            self._llm_model = None
            raise
        self._llm_tokenizer = tokenizer
        self._llm_model = model

    def _ensure_vlm(self) -> None:
        """Load VLM into VRAM, unloading LLM first if necessary."""
        if self._vlm_model is not None:
            return
        if self._llm_model is not None:
            self._llm_model = None
            self._llm_tokenizer = None
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        print(f"⌛ Loading VLM {VLM_MODEL_NAME}...")
        try:
            processor = AutoProcessor.from_pretrained(
                VLM_MODEL_NAME, token=HF_TOKEN, cache_dir=MODEL_CACHE_DIR
            )
            load_kwargs = {
                "dtype": getattr(torch, TORCH_DTYPE),
                "token": HF_TOKEN,
                "cache_dir": MODEL_CACHE_DIR,
            }
            if DEVICE == "cuda":
                load_kwargs["device_map"] = {"": 0}
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_NAME, **load_kwargs
            )
            if DEVICE != "cuda":
                model = model.to(DEVICE)
            model.eval()
        except Exception:
            self._vlm_processor = None
            self._vlm_model = None
            raise
        self._vlm_processor = processor
        self._vlm_model = model

    def _tokenize_text(self, messages: list[dict], tools: list | None = None) -> dict:
        """Render chat template for text-only input using LLM tokenizer."""
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if tools is not None:
            kwargs["tools"] = tools
        text = self._llm_tokenizer.apply_chat_template(messages, **kwargs)
        return self._llm_tokenizer(text, return_tensors="pt", return_attention_mask=True)

    def _parse_tool_call(self, output_text: str) -> dict:
        """Extract and parse the tool call JSON from model output.

        Handles Qwen2.5 (<tool_call>) and Llama 3.1 (<|python_tag|>) formats.
        """
        match = re.search(r"<tool_call>(.*?)</tool_call>", output_text, re.DOTALL)
        if match:
            try:
                call = json.loads(match.group(1).strip())
                return call.get("arguments", {})
            except json.JSONDecodeError:
                pass

        match = re.search(r"<\|python_tag\|>(.*?)(?:<\|eom_id\|>|$)", output_text, re.DOTALL)
        if match:
            try:
                call = json.loads(match.group(1).strip())
                return call.get("parameters", {})
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{[^{}]*\"date\"[^{}]*\}", output_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {"date": None, "incident_type": None, "description": None, "has_photos": None}

    def extract(self, messages: list[dict]) -> dict:
        """Extract claim fields from conversation messages using tool calling."""
        self._ensure_llm()
        encoding = self._tokenize_text(messages, tools=[EXTRACT_TOOL_SCHEMA])
        input_ids = encoding["input_ids"].to(self._llm_model.device)
        attention_mask = encoding["attention_mask"].to(self._llm_model.device)

        with torch.no_grad():
            output_ids = self._llm_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=self._llm_tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        output_text = self._llm_tokenizer.decode(new_tokens, skip_special_tokens=False)
        return self._parse_tool_call(output_text)

    def generate(self, messages: list[dict]) -> str:
        """Generate a text response without tool calling."""
        self._ensure_llm()
        encoding = self._tokenize_text(messages)
        input_ids = encoding["input_ids"].to(self._llm_model.device)
        attention_mask = encoding["attention_mask"].to(self._llm_model.device)

        with torch.no_grad():
            output_ids = self._llm_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=self._llm_tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self._llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _infer_vision(self, image_path: str, prompt: str) -> str:
        """Run VLM inference on image + text prompt, return decoded lowercase response."""
        self._ensure_vlm()
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self._vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._vlm_processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self._vlm_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._vlm_model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self._vlm_processor.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._vlm_processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip().lower()

    def check_photo_coherence(self, image_path: str, incident_type: str) -> bool:
        """Return True if the image matches the declared incident type."""
        prompt = COHERENCE_PROMPTS.get(incident_type)
        if not prompt:
            return True
        return self._infer_vision(image_path, prompt).startswith("yes")

    def assess_damage_severity(self, image_path: str, incident_type: str) -> str:
        """Return damage severity: 'low', 'medium', or 'high'.

        Returns 'unknown' if incident_type has no matching prompt (no model call).
        Returns 'medium' if the VLM response does not contain low/medium/high — biases
        toward mid-range rather than flagging degradation, so cost estimates remain
        financially material even on garbled output.
        """
        prompt = SEVERITY_PROMPTS.get(incident_type)
        if not prompt:
            return "unknown"
        response = self._infer_vision(image_path, prompt)
        for level in ("low", "medium", "high"):
            if level in response:
                return level
        return "medium"
