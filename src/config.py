import os

import torch
from dotenv import load_dotenv

load_dotenv()


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_dtype(device: str) -> str:
    return "float32" if device == "cpu" else "bfloat16"


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = os.getenv("DEVICE", _detect_device())
TORCH_DTYPE = os.getenv("TORCH_DTYPE", _detect_dtype(DEVICE))
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
