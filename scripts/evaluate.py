"""
Askllama-reddit: Model Evaluation Script

Evaluates the fine-tuned (or base) model on the validation set.

Metrics reported:
  - Perplexity on validation set
  - Average generation length
  - Sample inference outputs on a set of test prompts

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-path ./results/merged
    python scripts/evaluate.py --model-path meta-llama/Llama-2-7b-hf  # base model baseline
    python scripts/evaluate.py --no-gpu   # force CPU (slow)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BASE_MODEL_NAME,
    HF_TOKEN,
    MERGED_MODEL_DIR,
    VAL_DATA_PATH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
)

TEST_PROMPTS = [
    "### Post Title:\nWhat is the best way to fine-tune a large language model?\n\n"
    "### Post Content:\nI have a dataset of domain-specific text and want to adapt a 7B LLM. "
    "What approaches work best?\n\n### Top Comments:",

    "### Post Title:\nHow does LoRA compare to full fine-tuning?\n\n"
    "### Post Content:\nI'm considering LoRA for my project. What are the trade-offs vs full fine-tuning?\n\n"
    "### Top Comments:",

    "### Post Title:\nWhat are the best quantization methods for LLM inference?\n\n"
    "### Post Content:\nI want to run a 13B model on a single GPU. What quantization approaches "
    "should I consider?\n\n### Top Comments:",

    "### Post Title:\nHow do I implement RAG (Retrieval Augmented Generation)?\n\n"
    "### Post Content:\nI want to ground my LLM responses in a private knowledge base. "
    "What is the best way to build a RAG pipeline?\n\n### Top Comments:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Askllama-reddit model")
    parser.add_argument(
        "--model-path",
        default=str(MERGED_MODEL_DIR),
        help="Path to merged model directory or HF model name (default: results/merged)",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help="Max tokens to generate for sample inference",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, use_gpu: bool):
    """Load model and tokenizer, with optional 4-bit quantization."""
    load_kwargs: dict = {"trust_remote_code": True}

    if use_gpu and torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["device_map"] = "auto"
        print(f"Loading model on GPU (4-bit) from: {model_path}")
    else:
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"
        print(f"Loading model on CPU from: {model_path}")

    # Gated models need a token; local paths don't
    if not os.path.isdir(model_path) and HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    tokenizer_kwargs = {"trust_remote_code": True}
    if not os.path.isdir(model_path) and HF_TOKEN:
        tokenizer_kwargs["token"] = HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, val_path: Path, device: torch.device) -> float:
    """
    Compute token-level perplexity on the validation set.
    PPL = exp(average negative log-likelihood per token).
    """
    if not val_path.exists():
        print(f"  WARNING: Validation file not found at {val_path}. Skipping perplexity.")
        return float("nan")

    records = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("  WARNING: Validation set is empty. Skipping perplexity.")
        return float("nan")

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for rec in records:
            text = rec.get("text", "")
            if not text:
                continue
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            )
            input_ids = inputs["input_ids"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            # outputs.loss is mean NLL over non-padded tokens
            nll = outputs.loss.item()
            n_tokens = (labels != tokenizer.pad_token_id).sum().item()
            total_nll += nll * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("nan")

    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


def run_inference(model, tokenizer, prompts: list[str], max_new_tokens: int, device: torch.device):
    """Run greedy-ish generation on each test prompt and return outputs."""
    results = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            results.append(decoded.strip())
    return results


def get_device(model) -> torch.device:
    """Get the device of the first model parameter."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def main():
    args = parse_args()
    use_gpu = not args.no_gpu

    # Check if model path exists or is a HF model name
    model_path = args.model_path
    if not os.path.isdir(model_path):
        print(f"  Model directory '{model_path}' not found locally.")
        if HF_TOKEN:
            print(f"  Will attempt to load from Hugging Face Hub: {model_path}")
        else:
            print("  ERROR: Set HF_TOKEN to load gated models from the Hub.")
            sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(model_path, use_gpu)
    device = get_device(model)

    print("\n" + "=" * 60)
    print("1. PERPLEXITY ON VALIDATION SET")
    print("=" * 60)
    ppl = compute_perplexity(model, tokenizer, VAL_DATA_PATH, device)
    if not math.isnan(ppl):
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  (Lower is better; a well-tuned model should be < 10 on in-domain data)")
    else:
        print("  Perplexity: N/A")

    print("\n" + "=" * 60)
    print("2. SAMPLE INFERENCE OUTPUTS")
    print("=" * 60)
    outputs = run_inference(model, tokenizer, TEST_PROMPTS, args.max_new_tokens, device)
    for i, (prompt, response) in enumerate(zip(TEST_PROMPTS, outputs), 1):
        # Print just the title line for brevity
        title_line = prompt.split("\n")[1]
        print(f"\n[{i}] {title_line}")
        print("-" * 50)
        print(response if response else "(empty response)")

    print("\n" + "=" * 60)
    print("3. GENERATION STATS")
    print("=" * 60)
    lengths = [len(tokenizer.encode(r)) for r in outputs if r]
    if lengths:
        print(f"  Avg response length : {sum(lengths)/len(lengths):.1f} tokens")
        print(f"  Min / Max           : {min(lengths)} / {max(lengths)} tokens")
    else:
        print("  No valid responses generated.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
