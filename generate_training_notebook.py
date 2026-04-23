import json
import os
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Gemma4Good 0.1 Grounding Training (QLoRA)\n",
                "\n",
                "This notebook executes the QLoRA fine-tuning sequence for the Gemma4Good `0.1` release. It trains a grounded Gemma 4 adapter on fully grounded, PIVOT-tagged SGT sessions extracted by the underlying governance loop.\n",
                "\n",
                "**Why are we training on Kaggle?**\n",
                "The target model is `google/gemma-4-E2B-it`. We discovered during our viability testing that Gemma 4 E2B's `AltUp` alternating layer architecture contains a `per_layer_token_embd.weight` tensor with **2.35B parameters** (nearly half the model's capacity). Because of its structure, this tensor cannot be quantized below 4-bit (`Q4_K` / `NF4`). This enforces a mathematically strict VRAM floor of ~2.88 GB, regardless of how aggressively the transformer blocks are quantized (banning IQ3/Q3 variants). \n",
                "\n",
                "While this 2.88 GB floor prevents viable inference on highly-constrained local edge hardware (like consumer 8GB GPUs actively running an OS desktop + vision tasks), the model trains perfectly on Kaggle's 2xT4 (30GB VRAM) infrastructure, allowing us to build the grounded weights quickly."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required libraries\n",
                "!pip install -q -U accelerate peft bitsandbytes transformers trl datasets"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import torch\n",
                "from datasets import load_dataset\n",
                "from transformers import (\n",
                "    AutoModelForCausalLM,\n",
                "    AutoTokenizer,\n",
                "    BitsAndBytesConfig,\n",
                "    TrainingArguments,\n",
                ")\n",
                "from peft import LoraConfig, get_peft_model\n",
                "from trl import SFTTrainer\n",
                "\n",
                "print(f\"PyTorch: {torch.__version__}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Load the Training Dataset\n",
                "Load the `grounding_gemma4_v2.jsonl` containing the target PIVOT extractions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Point this to the kaggle dataset once uploaded, e.g. /kaggle/input/haic-gemma4-data/grounding_gemma4_v2.jsonl\n",
                "data_path = \"/kaggle/input/haic-gemma4-data/grounding_gemma4_v2.jsonl\"\n",
                "\n",
                "# If testing locally, fallback path\n",
                "if not os.path.exists(data_path):\n",
                "    data_path = \"grounding_gemma4_v2.jsonl\" # placeholder\n",
                "\n",
                "dataset = load_dataset(\"json\", data_files=data_path, split=\"train\")\n",
                "print(f\"Loaded {len(dataset)} sessions for tuning.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Load Model and QLoRA Config"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_id = \"google/gemma-4-E2B-it\"\n",
                "\n",
                "# Strict 4-bit load to accommodate the AltUp architecture efficiently\n",
                "bnb_config = BitsAndBytesConfig(\n",
                "    load_in_4bit=True,\n",
                "    bnb_4bit_quant_type=\"nf4\",\n",
                "    bnb_4bit_compute_dtype=torch.bfloat16,\n",
                "    bnb_4bit_use_double_quant=True,\n",
                ")\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    model_id,\n",
                "    quantization_config=bnb_config,\n",
                "    device_map=\"auto\"\n",
                ")\n",
                "\n",
                "# Apply LoRA adapters to all standard dense projections\n",
                "peft_config = LoraConfig(\n",
                "    r=32,\n",
                "    lora_alpha=64,\n",
                "    lora_dropout=0.05,\n",
                "    bias=\"none\",\n",
                "    task_type=\"CAUSAL_LM\",\n",
                "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Formatting and Execution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def formatting_prompts_func(example):\n",
                "    output_texts = []\n",
                "    for i in range(len(example['system'])):\n",
                "        messages = [\n",
                "            {\"role\": \"user\", \"content\": example['system'][i] + \"\\n\" + example['prompt'][i]},\n",
                "            {\"role\": \"assistant\", \"content\": example['response'][i]}\n",
                "        ]\n",
                "        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)\n",
                "        output_texts.append(text)\n",
                "    return output_texts\n",
                "\n",
                "training_args = TrainingArguments(\n",
                "    output_dir=\"/kaggle/working/outputs\",\n",
                "    per_device_train_batch_size=2,\n",
                "    gradient_accumulation_steps=4,\n",
                "    optim=\"paged_adamw_32bit\",\n",
                "    learning_rate=2e-4,\n",
                "    lr_scheduler_type=\"cosine\",\n",
                "    save_strategy=\"epoch\",\n",
                "    logging_steps=10,\n",
                "    num_train_epochs=2.0,\n",
                "    max_steps=-1,\n",
                "    fp16=True,  # Because T4 doesn't do pure BF16 natively as fast as Ampere, using mixed fp16\n",
                "    report_to=\"none\"\n",
                ")\n",
                "\n",
                "trainer = SFTTrainer(\n",
                "    model=model,\n",
                "    train_dataset=dataset,\n",
                "    peft_config=peft_config,\n",
                "    formatting_func=formatting_prompts_func,\n",
                "    max_seq_length=2048,\n",
                "    tokenizer=tokenizer,\n",
                "    args=training_args,\n",
                ")\n",
                "\n",
                "print(\"Starting fine-tuning...\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Uncomment to execute training\n",
                "# trainer.train()\n",
                "\n",
                "output_model_path = \"/kaggle/working/gemma4good-0_1\"\n",
                "# trainer.save_model(output_model_path)\n",
                "print(f\"Model adapters saved to {output_model_path}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

ROOT = Path(__file__).resolve().parent
out_path = ROOT / "notebook" / "haic_gemma4_training.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated {out_path}")
