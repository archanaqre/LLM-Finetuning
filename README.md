## LLM-Finetuning
Base codes to finetune LLMs using Unsloth

# 🦙 Fine-tuning LLaMA 3 with Unsloth on Google Colab

This repository contains a Google Colab notebook for fine-tuning 4-bit LLaMA 3 models using [Unsloth](https://github.com/unslothai/unsloth), a high-performance library for efficient LLM training.

## 📌 Model Used
- **Base Model**: `unsloth/Llama-3.2-3B-Instruct` (4-bit quantized)
- **Adapter Method**: LoRA with 4-bit quantization via `bitsandbytes`
- **Tokenizer Template**: `llama-3.1` (compatible with LLaMA 3)

## 📚 Dataset
- **Name**: [`mlabonne/FineTome-100k`](https://huggingface.co/datasets/mlabonne/FineTome-100k)
- **Format**: ShareGPT-style multi-turn conversations converted to HuggingFace `"role"` / `"content"` format using `unsloth.standardize_sharegpt`.

## ⚙️ Fine-tuning Details
- Trained using `trl.SFTTrainer`
- Sequence length: 2048 tokens
- LoRA config: `r=16`, `lora_alpha=16`, `dropout=0.0`
- Trained for 60 steps (demo); configurable with `num_train_epochs` or `max_steps`
- Memory-efficient training using `load_in_4bit=True`

## 🚀 Inference & Saving
- Generates responses via `model.generate(...)` with Unsloth's `FastLanguageModel`
- Fine-tuned model and tokenizer saved using:
  ```python
  model.save_pretrained("lora_model")
  tokenizer.save_pretrained("lora_model")
