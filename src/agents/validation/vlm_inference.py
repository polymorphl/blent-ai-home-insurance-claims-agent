import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.config import DEVICE, HF_TOKEN, MODEL_CACHE_DIR, TORCH_DTYPE

VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

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


class VLMInference:
    """Wrapper for Qwen2.5-VL image-text inference."""

    def __init__(self):
        print(f"⌛ Loading VLM {VLM_MODEL_NAME}...")
        self.processor = AutoProcessor.from_pretrained(
            VLM_MODEL_NAME, token=HF_TOKEN, cache_dir=MODEL_CACHE_DIR
        )
        load_kwargs = {
            "dtype": getattr(torch, TORCH_DTYPE),
            "token": HF_TOKEN,
            "cache_dir": MODEL_CACHE_DIR,
        }
        if DEVICE == "cuda":
            load_kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_NAME, **load_kwargs
        )
        if DEVICE != "cuda":
            self.model = self.model.to(DEVICE)
        self.model.eval()

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
        """Return damage severity: 'low', 'medium', or 'high'. Returns 'unknown' if no prompt."""
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
