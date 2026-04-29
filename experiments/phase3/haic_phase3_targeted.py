import os
import torch
import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

if __name__ == "__main__":
    print("Initiating Phase 3: Targeted Sub-Module Unfreezing Track")
    
    # 1. Configuration
    model_id = "google/gemma-4-E2B-it"
    data_path = "grounding_gemma4_v2.jsonl"
    
    if not os.path.exists(data_path):
        data_path = "/kaggle/input/haic-gemma4-data/grounding_gemma4_v2.jsonl"
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    
    # 2. Extracting PRISM Worst-Offenders
    # According to Phase 1/2 recordings, L29 is the highest hostility layer.
    # We use a regex target to isolate exactly layers 28, 29, 30 for massive unfreezing.
    # We apply r=128 exclusively to the MLP gating/projection matrices (up_proj, down_proj, gate_proj)
    # where the geometric dimension vectors reside.
    
    target_modules_regex = r".*layers\.(28|29|30)\.mlp\.(up_proj|down_proj|gate_proj)"
    print(f"Targeting LoRA Modules: {target_modules_regex} with r=128")
    
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_regex,
    )
    
    dataset = load_dataset("json", data_files=data_path, split="train")

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['system'])):
            messages = [
                {"role": "user", "content": example['system'][i] + "\\n" + example['prompt'][i]},
                {"role": "assistant", "content": example['response'][i]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    training_args = TrainingArguments(
        output_dir="./outputs_targeted",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=5,
        num_train_epochs=2.0,
        fp16=True, 
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Executing Targeted Activation Training...")
    # trainer.train()
    # print("Targeted Training complete.")
