import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import DPOTrainer

if __name__ == "__main__":
    print("Initiating Phase 3: PRISM-DPO Alignment Track")
    
    # 1. Configuration
    model_id = "google/gemma-4-E2B-it"
    data_path = "prism_dpo_pairs.jsonl"
    
    if not os.path.exists(data_path):
        data_path = "/kaggle/working/experiments/phase3/prism_dpo_pairs.jsonl"
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # DPO requires a base model and a ref model (usually ref_model=None makes trl handle it by detaching adapters)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    dataset = load_dataset("json", data_files=data_path, split="train")

    def return_prompt_and_responses(samples):
        # Format for DPO: prompt, chosen, rejected
        return {
            "prompt": [sys + "\\n" + pr for sys, pr in zip(samples["system"], samples["prompt"])],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }
        
    original_columns = dataset.column_names
    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )

    training_args = TrainingArguments(
        output_dir="./outputs_dpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        learning_rate=5e-5, # DPO learning rate usually lower
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=5,
        num_train_epochs=1.0,
        fp16=True, 
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, # Automatically detaches peft adapter for reference
        args=training_args,
        beta=0.1, # KL penalty
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Executing Custom DPO Training...")
    # dpo_trainer.train()
    # print("DPO Training complete.")
