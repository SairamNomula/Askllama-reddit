"""
Centralized configuration for Askllama-reddit.
All model, training, and inference settings in one place.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = PROJECT_ROOT / "custjsonl.jsonl"
TRAIN_DATA_PATH = DATA_DIR / "train.jsonl"
VAL_DATA_PATH = DATA_DIR / "val.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results"
MERGED_MODEL_DIR = OUTPUT_DIR / "merged"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
HF_TOKEN = os.environ.get("HF_TOKEN")

# If you've pushed your fine-tuned model to the Hub, set this instead:
# FINETUNED_MODEL_NAME = "your-username/askllama-reddit-7b"

# ---------------------------------------------------------------------------
# Quantization (BitsAndBytes 4-bit)
# ---------------------------------------------------------------------------
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "float16"

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_STEPS = 30
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 100
FP16 = True
DATASET_TEXT_FIELD = "text"

# ---------------------------------------------------------------------------
# Inference (Gradio app)
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1
