"""
Askllama-reddit: Standalone Training Script

Fine-tunes Llama-2-7b on the prepared Reddit ML discussion data using QLoRA.
This mirrors src/model.ipynb but runs entirely from the command line
(no Google Colab required — a local NVIDIA GPU with ≥16 GB VRAM is needed).

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 5 --lr 1e-4
    python scripts/train.py --no-merge   # skip merging adapters after training

Output:
    results/final_adapter/   — LoRA adapter weights
    results/merged/          — Full merged model (ready for app.py)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Project root on sys.path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BASE_MODEL_NAME,
    HF_TOKEN,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    OUTPUT_DIR,
    MERGED_MODEL_DIR,
    # LoRA
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_BIAS,
    LORA_TASK_TYPE,
    LORA_TARGET_MODULES,
    # Training
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    WARMUP_STEPS,
    LOGGING_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    FP16,
    DATASET_TEXT_FIELD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llama-2-7b with QLoRA on Reddit ML data")
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--no-merge", action="store_true", help="Skip merging LoRA into base model")
    parser.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"],
                        help="Experiment tracking backend")
    return parser.parse_args()


def load_jsonl_dataset(path: Path) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    import json
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. QLoRA training requires a CUDA-capable GPU.")
        print("  Use Google Colab (src/model.ipynb) if you don't have a local GPU.")
        sys.exit(1)

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN is not set. Llama-2 is a gated model.")
        print("  Set it in your .env file or export HF_TOKEN=<your_token>")
        sys.exit(1)

    print("=" * 60)
    print("Askllama-reddit: QLoRA Fine-tuning")
    print("=" * 60)
    print(f"  Base model  : {BASE_MODEL_NAME}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Batch size  : {args.batch_size} (+ {args.grad_accum}x grad accum)")
    print(f"  Max seq len : {args.max_seq_len}")
    print(f"  Report to   : {args.report_to}")
    print()

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    if not TRAIN_DATA_PATH.exists():
        print(f"ERROR: Training data not found at {TRAIN_DATA_PATH}")
        print("  Run: python scripts/prepare_data.py")
        sys.exit(1)

    print("Loading datasets...")
    train_dataset = load_jsonl_dataset(TRAIN_DATA_PATH)
    val_dataset = load_jsonl_dataset(VAL_DATA_PATH)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val  : {len(val_dataset)} samples")

    # -------------------------------------------------------------------------
    # Model & tokenizer
    # -------------------------------------------------------------------------
    print(f"\nLoading base model ({BASE_MODEL_NAME}) with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=True, token=HF_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Model loaded. GPU memory used: {allocated_gb:.2f} GB")

    # -------------------------------------------------------------------------
    # LoRA config
    # -------------------------------------------------------------------------
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias=LORA_BIAS,
        task_type=LORA_TASK_TYPE,
        target_modules=LORA_TARGET_MODULES,
    )

    # -------------------------------------------------------------------------
    # Training arguments
    # -------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        warmup_steps=WARMUP_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field=DATASET_TEXT_FIELD,
        max_seq_length=args.max_seq_len,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    torch.cuda.empty_cache()
    train_result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Total steps   : {trainer.state.global_step}")
    print(f"  Final train loss: {train_result.training_loss:.4f}")

    # -------------------------------------------------------------------------
    # Plot training curve
    # -------------------------------------------------------------------------
    log_history = trainer.state.log_history
    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_losses, label="Training Loss", alpha=0.8)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", marker="o", alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss — Askllama-reddit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "training_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Training curve saved to {plot_path}")

    # -------------------------------------------------------------------------
    # Save LoRA adapter
    # -------------------------------------------------------------------------
    adapter_dir = OUTPUT_DIR / "final_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  LoRA adapter saved to {adapter_dir}")

    if args.no_merge:
        print("\nSkipping merge (--no-merge). Done.")
        return

    # -------------------------------------------------------------------------
    # Merge adapter into base model
    # -------------------------------------------------------------------------
    print("\nMerging LoRA adapters into base model (this may take a few minutes)...")

    # Free training memory first
    del trainer, base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload base in fp16 for merging
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    merged = PeftModel.from_pretrained(base_for_merge, str(adapter_dir))
    merged = merged.merge_and_unload()

    MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_MODEL_DIR))
    tokenizer.save_pretrained(str(MERGED_MODEL_DIR))
    print(f"  Merged model saved to {MERGED_MODEL_DIR}")

    del merged, base_for_merge
    gc.collect()
    torch.cuda.empty_cache()

    print("\nAll done! Run the chat interface with:")
    print(f"  MODEL_PATH={MERGED_MODEL_DIR} python app.py")


if __name__ == "__main__":
    main()
