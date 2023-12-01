# Askllama-reddit

## Pre-requisites:
* Python
* Jupyter Notebook
* Postman
* Hugging Face (AutoModelForCausalLM)
* trl
* pandas
* transformers
* peft
* BitsAndBytesConfig

Create a Hugging Face Hub account: https://huggingface.co/join

## Running the Project
Clone the repository: `git clone https://github.com/SairamNomula/Askllama-reddit.git`

###    Create a Hugging Face API token:
* Go to your Hugging Face account settings: https://huggingface.co/settings/tokens
* Click "New Token" and provide a descriptive name.
* Copy the generated token.

> Set the environment variables

## Technologies used
* Hugging Face Transformers: Provides pre-trained models and tools for training and fine-tuning LLMs.
* PEFT: Enables parameter-efficient fine-tuning of LLMs.
* QLoRA: A specific PEFT method for efficient fine-tuning.
* TRL: Provides tools for training LLMs using reinforcement learning.
* Hugging Face Hub: A platform for sharing and managing NLP models, datasets, and other resources.
* Datasets: Facilitates loading and preprocessing datasets.
* Torch: A powerful deep learning framework.
* BitsAndBytes: Enables efficient quantization and compression of deep learning models.
* Einops: Provides tools for manipulating and transforming tensor shapes.
* Wandb: Tracks and manages machine learning experiments.