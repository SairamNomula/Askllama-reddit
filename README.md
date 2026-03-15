# Askllama-reddit

A domain-specific conversational QA system fine-tuned on Reddit machine learning discussions. Built on **Llama-2-7b** using **QLoRA** (4-bit quantization + LoRA) for parameter-efficient training, with a **Gradio** chat interface for interactive use.

## Project Structure

```
Askllama-reddit/
├── app.py                  # Gradio chat interface (with structured logging)
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── Dockerfile              # Container image for the chat app
├── docker-compose.yml      # Easy GPU-accelerated deployment
├── scripts/
│   ├── prepare_data.py     # Data deduplication & formatting pipeline
│   ├── train.py            # Standalone QLoRA training script (local GPU)
│   └── evaluate.py         # Model evaluation: perplexity + sample inference
├── src/
│   └── model.ipynb         # Training notebook (Google Colab alternative)
├── custjsonl.jsonl         # Raw Reddit discussion data (3,029 records)
├── data/                   # Cleaned train/val splits (generated)
│   ├── train.jsonl
│   └── val.jsonl
└── logs/
    └── applogs.log         # Runtime query/response logs
```

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/SairamNomula/Askllama-reddit.git
cd Askllama-reddit
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Linux / macOS
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

Edit `.env` and fill in your credentials:

```env
HF_TOKEN=your_huggingface_token_here
MODEL_PATH=./results/merged
MAX_NEW_TOKENS=256
```

- Get a token at: https://huggingface.co/settings/tokens
- Request Llama-2 access at: https://huggingface.co/meta-llama/Llama-2-7b-hf

### 3. Prepare Data

```bash
python scripts/prepare_data.py
```

Deduplicates the raw data, formats prompts from (title, post content, comments), and writes `data/train.jsonl` + `data/val.jsonl`.

### 4. Train the Model

**Option A — Local GPU** (requires NVIDIA GPU with ≥16 GB VRAM):

```bash
python scripts/train.py
# With custom hyperparameters:
python scripts/train.py --epochs 5 --lr 1e-4 --report-to wandb
```

This saves the LoRA adapter to `results/final_adapter/` and the merged model to `results/merged/`.

**Option B — Google Colab** (free T4 GPU):

1. Upload `custjsonl.jsonl` and open `src/model.ipynb` in Colab
2. Select T4 GPU runtime
3. Run all cells sequentially
4. Download `results/merged/` to your local machine

### 5. Evaluate the Model

```bash
python scripts/evaluate.py
# Against a specific model:
python scripts/evaluate.py --model-path ./results/merged
# Against the base model (baseline comparison):
python scripts/evaluate.py --model-path meta-llama/Llama-2-7b-hf
```

Reports:
- **Perplexity** on the validation set
- **Sample inference outputs** on standard ML questions
- **Generation stats** (avg token length, throughput)

### 6. Run the Chat Interface

```bash
python app.py
```

Open http://localhost:7860 in your browser.

Or set environment variables directly:

```bash
# Linux / macOS
export HF_TOKEN=your_token_here
export MODEL_PATH=./results/merged
python app.py
```

**Note:** A CUDA-capable NVIDIA GPU is recommended. The app runs on CPU but needs ~28 GB RAM for a 7B model.

### 7. Deploy with Docker

```bash
# Build the image
docker build -t askllama .

# Run with GPU (requires nvidia-container-toolkit)
docker run --gpus all \
  -e HF_TOKEN=your_token \
  -v $(pwd)/results/merged:/model \
  -p 7860:7860 askllama

# Or use docker-compose (reads from .env automatically)
docker-compose up
```

For CPU-only deployment, remove the `deploy.resources` block from `docker-compose.yml`.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | meta-llama/Llama-2-7b-hf |
| Method | QLoRA (4-bit NF4 + LoRA) |
| LoRA Rank | 64 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.1 |
| Target Modules | q_proj, v_proj |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Batch Size | 1 (× 4 gradient accumulation) |
| Max Seq Length | 512 |
| Warmup Steps | 30 |

## Technologies

| Component | Technology |
|-----------|-----------|
| Base Model | Meta Llama-2-7b-hf |
| Fine-tuning | QLoRA (PEFT + BitsAndBytes) |
| Trainer | TRL SFTTrainer |
| Chat UI | Gradio |
| Framework | PyTorch + Hugging Face Transformers |
| Deployment | Docker / docker-compose |
| Experiment Tracking | Weights & Biases (optional) |
