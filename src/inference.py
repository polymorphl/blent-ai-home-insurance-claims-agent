import json
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.agents.declaration.tools import EXTRACT_TOOL_SCHEMA
from src.config import DEVICE, HF_TOKEN, MAX_NEW_TOKENS, MODEL_CACHE_DIR, MODEL_NAME, TEMPERATURE, TORCH_DTYPE

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
    """Single model wrapper for Qwen2.5-VL-7B-Instruct covering all pipeline agents."""

    def __init__(self):
        """Load processor and model, printing progress to stdout."""
        print(f"⌛ Loading model {MODEL_NAME}...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, token=HF_TOKEN, cache_dir=MODEL_CACHE_DIR
        )
        load_kwargs = {
            "dtype": getattr(torch, TORCH_DTYPE),
            "token": HF_TOKEN,
            "cache_dir": MODEL_CACHE_DIR,
        }
        if DEVICE == "cuda":
            load_kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, **load_kwargs
        )
        if DEVICE != "cuda":
            self.model = self.model.to(DEVICE)
        self.model.eval()

    def _tokenize_text(self, messages: list[dict], tools: list | None = None) -> dict:
        """Render chat template for text-only input (no images) and return processor output."""
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if tools is not None:
            kwargs["tools"] = tools
        text = self.processor.apply_chat_template(messages, **kwargs)
        return self.processor(text=[text], return_tensors="pt")

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
        encoding = self._tokenize_text(messages, tools=[EXTRACT_TOOL_SCHEMA])
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        output_text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return self._parse_tool_call(output_text)

    def generate(self, messages: list[dict]) -> str:
        """Generate a text response without tool calling."""
        encoding = self._tokenize_text(messages)
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def check_photo_coherence(self, image_path: str, incident_type: str) -> bool:
        """Return True if the image matches the declared incident type."""
        prompt = COHERENCE_PROMPTS.get(incident_type)
        if not prompt:
            return True

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip().lower()
        return response.startswith("yes")

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

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip().lower()

        for level in ("low", "medium", "high"):
            if level in response:
                return level
        return "medium"
