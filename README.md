# Askllama-reddit

A domain-specific conversational QA system fine-tuned on Reddit machine learning discussions. Built on **Llama-2-7b** using **QLoRA** (4-bit quantization + LoRA) for parameter-efficient training, with a **Gradio** chat interface for interactive use.

## Project Structure

```
Askllama-reddit/
├── app.py                  # Gradio chat interface
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── scripts/
│   └── prepare_data.py     # Data deduplication & formatting pipeline
├── src/
│   └── model.ipynb         # Training notebook (Google Colab)
├── custjsonl.jsonl         # Raw Reddit discussion data
├── redditcustdata.csv      # Raw CSV data
└── data/                   # Cleaned train/val splits (generated)
```

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/SairamNomula/Askllama-reddit.git
cd Askllama-reddit
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example env file and add your Hugging Face token:

```bash
# Linux / macOS
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

Edit `.env` and set your Hugging Face token. The app loads this file automatically via `python-dotenv`.

Get a token at: https://huggingface.co/settings/tokens

You also need to request access to Llama-2: https://huggingface.co/meta-llama/Llama-2-7b-hf

### 3. Prepare Data

```bash
python scripts/prepare_data.py
```

This deduplicates the raw data, creates a proper prompt format using all three fields (title, post content, comments), and splits into train/validation sets under `data/`.

### 4. Train the Model (Google Colab)

1. Upload `custjsonl.jsonl` and open `src/model.ipynb` in Google Colab
2. Select a T4 GPU runtime (free tier works)
3. Run all cells sequentially
4. Download the merged model from `results/merged/`

### 5. Run the Chat Interface

Make sure your `.env` file has `HF_TOKEN` and `MODEL_PATH` set, then:

```bash
python app.py
```

Or set environment variables directly:

```bash
# Linux / macOS
export HF_TOKEN=your_token_here
export MODEL_PATH=./results/merged
python app.py

# Windows (Command Prompt)
set HF_TOKEN=your_token_here
set MODEL_PATH=./results/merged
python app.py

# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"
$env:MODEL_PATH="./results/merged"
python app.py
```

Open http://localhost:7860 in your browser.

**Note:** A CUDA-capable NVIDIA GPU is recommended. The app will run on CPU but will be very slow (~28GB RAM needed).

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | meta-llama/Llama-2-7b-hf |
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA Rank | 64 |
| LoRA Alpha | 16 |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Max Seq Length | 512 |
| Batch Size | 1 (with 4x gradient accumulation) |

## Technologies

- **Hugging Face Transformers** - Model loading, tokenization, and training
- **PEFT** - Parameter-efficient fine-tuning with LoRA adapters
- **TRL** - Supervised fine-tuning trainer (SFTTrainer)
- **BitsAndBytes** - 4-bit quantization for memory-efficient training
- **Gradio** - Interactive chat web interface
- **Wandb** - Experiment tracking and loss visualization
- **PyTorch** - Deep learning framework
