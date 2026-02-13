"""
Askllama-reddit: Gradio Chat Interface

Serves the fine-tuned Llama-2-7b model as an interactive chat UI.
The model answers questions about ML/AI topics in the style of Reddit discussions.

Usage:
    python app.py

Set MODEL_PATH env var to point to your merged model directory,
or it falls back to the base Llama-2-7b model.
"""

from __future__ import annotations

import os
import sys

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

from config import (
    BASE_MODEL_NAME,
    MERGED_MODEL_DIR,
    OUTPUT_DIR,
    MAX_NEW_TOKENS as DEFAULT_MAX_TOKENS,
    TEMPERATURE as DEFAULT_TEMPERATURE,
    TOP_P as DEFAULT_TOP_P,
)

# ---------------------------------------------------------------------------
# Configuration (env vars override config.py defaults)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", str(MERGED_MODEL_DIR))
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", str(DEFAULT_MAX_TOKENS)))

HAS_CUDA = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
model = None
tokenizer = None


def load_model():
    """Load the fine-tuned or base model."""
    global model, tokenizer

    # Quantization config — only use 4-bit if GPU is available
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if HAS_CUDA:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        print("WARNING: No CUDA GPU detected. Loading model in float32 on CPU.")
        print("  This will be slow and use ~28GB RAM for a 7B model.")
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"

    # Try loading the fine-tuned merged model first
    if os.path.isdir(MODEL_PATH):
        print(f"Loading fine-tuned model from {MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    else:
        # Fallback: try loading adapter on top of base model
        adapter_path = str(OUTPUT_DIR / "final_adapter")
        if os.path.isdir(adapter_path):
            print(f"Loading base model + LoRA adapter from {adapter_path}...")
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME, token=HF_TOKEN, **load_kwargs,
            )
            model = PeftModel.from_pretrained(base, adapter_path)
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME, trust_remote_code=True, token=HF_TOKEN,
            )
        else:
            if not HF_TOKEN:
                print("ERROR: No fine-tuned model found and HF_TOKEN is not set.")
                print("  Set HF_TOKEN in your .env file or environment to download the base model.")
                sys.exit(1)
            print(f"WARNING: No fine-tuned model found. Loading base {BASE_MODEL_NAME}...")
            print("  Run training first (src/model.ipynb) to get better results.")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME, token=HF_TOKEN, **load_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME, trust_remote_code=True, token=HF_TOKEN,
            )

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded and ready!")


def get_device():
    """Get the device the model inputs should be sent to."""
    if HAS_CUDA:
        return torch.device("cuda:0")
    return torch.device("cpu")


def format_prompt(user_message: str, chat_history: list) -> str:
    """Format user message into the training prompt template."""
    prompt = f"### Post Title:\n{user_message}\n\n"
    prompt += "### Post Content:\nPlease provide a detailed answer.\n\n"
    prompt += "### Top Comments:\n"
    return prompt


def generate_response(message: str, history: list, temperature: float, max_tokens: int, top_p: float):
    """Generate a streaming response."""
    if model is None:
        yield "Model not loaded. Please wait..."
        return

    prompt = format_prompt(message, history)
    device = get_device()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": int(max_tokens),
        "temperature": max(float(temperature), 0.01),
        "top_p": float(top_p),
        "repetition_penalty": 1.1,
        "do_sample": True,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_response = ""
    for new_token in streamer:
        partial_response += new_token
        yield partial_response

    thread.join()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def create_app():
    """Create and return the Gradio app."""
    with gr.Blocks(title="AskLlama - Reddit ML QA", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# AskLlama - Reddit ML Discussion Bot\n"
            "Ask questions about machine learning, AI, LLMs, and deep learning. "
            "Powered by Llama-2-7b fine-tuned on Reddit ML discussions."
        )

        with gr.Accordion("Settings", open=False):
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, value=DEFAULT_TEMPERATURE, step=0.1,
                label="Temperature", info="Higher = more creative, lower = more focused"
            )
            max_tokens = gr.Slider(
                minimum=50, maximum=512, value=MAX_NEW_TOKENS, step=50,
                label="Max Tokens", info="Maximum length of the response"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=DEFAULT_TOP_P, step=0.05,
                label="Top P", info="Nucleus sampling threshold"
            )

        gr.ChatInterface(
            fn=generate_response,
            additional_inputs=[temperature, max_tokens, top_p],
            examples=[
                "What is the best way to fine-tune a large language model?",
                "How does LoRA compare to full fine-tuning for LLMs?",
                "What are the trade-offs between different quantization methods?",
                "How do I implement RAG (Retrieval Augmented Generation)?",
            ],
            retry_btn="Retry",
            undo_btn="Undo",
            clear_btn="Clear",
        )

    return app


if __name__ == "__main__":
    load_model()
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
