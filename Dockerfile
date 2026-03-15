# Askllama-reddit — Gradio chat interface
#
# Build:
#   docker build -t askllama .
#
# Run (GPU, with a pre-trained merged model on the host):
#   docker run --gpus all \
#     -e HF_TOKEN=your_token \
#     -v $(pwd)/results/merged:/model \
#     -e MODEL_PATH=/model \
#     -p 7860:7860 askllama
#
# Run (CPU fallback, no GPU):
#   docker run \
#     -e HF_TOKEN=your_token \
#     -v $(pwd)/results/merged:/model \
#     -e MODEL_PATH=/model \
#     -p 7860:7860 askllama

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py config.py ./
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create directories expected at runtime
RUN mkdir -p results logs

# Gradio listens on 7860 by default
EXPOSE 7860

# Environment variable defaults (override at runtime)
ENV MODEL_PATH=/model \
    MAX_NEW_TOKENS=256 \
    PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
