import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEVICE, MAX_NEW_TOKENS, MODEL_NAME, TEMPERATURE, TORCH_DTYPE
from src.agents.declaration.tools import EXTRACT_TOOL_SCHEMA


class HFInference:
    """Wrapper for running inference with a Hugging Face language model."""
    def __init__(self):
        """Load the model and tokenizer from the configured model name."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=getattr(torch, TORCH_DTYPE),
            device_map=DEVICE,
        )
        self.model.eval()

    def extract(self, messages: list[dict]) -> dict:
        """Extract claim fields from conversation messages using the model's tool-calling capability."""
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=[EXTRACT_TOOL_SCHEMA],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return self._parse_tool_call(output_text)

    def generate(self, messages: list[dict]) -> str:
        """Generate a text response from the model without tool calling."""
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _parse_tool_call(self, output_text: str) -> dict:
        """Extract and parse the tool call JSON from model output, with fallback extraction."""
        match = re.search(
            r"<\|python_tag\|>(.*?)(?:<\|eom_id\|>|$)", output_text, re.DOTALL
        )
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
