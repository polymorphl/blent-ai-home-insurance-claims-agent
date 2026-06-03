import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
TORCH_DTYPE = "bfloat16"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1

HF_TOKEN = os.getenv("HF_TOKEN")
