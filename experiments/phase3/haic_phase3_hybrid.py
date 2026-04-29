import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

def calculate_kurtosis(hidden_states):
    '''Calculate kurtosis over the hidden dimension for a batch of activations.'''
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True) + 1e-8
    diff = hidden_states - mean
    fourth_moment = (diff ** 4).mean(dim=-1, keepdim=True)
    kurtosis = fourth_moment / (var ** 2)
    return kurtosis.mean() 

class GeometryHybridTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.get("logits")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Calculate Geometric Penalty on final states
        final_hidden_states = outputs.hidden_states[-1]
        geom_penalty = calculate_kurtosis(final_hidden_states)
        
        lambda_k = 0.01 # Aggressive hybrid penalty
        total_loss = ce_loss + (lambda_k * geom_penalty)
        return (total_loss, outputs) if return_outputs else total_loss

def extract_prism_metrics(model, tokenizer, output_path):
    """Post-training PRISM extraction: measure qh on the tuned model."""
    import json, numpy as np
    
    print("Extracting PRISM geometry from tuned model...")
    
    # Short calibration prompts to drive activations through the model
    calibration_prompts = [
        "Evaluate the ethical implications of autonomous decision-making in healthcare.",
        "What are the risks of deploying AI without human oversight in education?",
        "Describe the consent requirements for using personal data in model training.",
    ]
    
    all_kurtosis = []
    all_norms = []
    
    model.eval()
    with torch.no_grad():
        for prompt_text in calibration_prompts:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            
            for layer_idx, hs in enumerate(outputs.hidden_states):
                hs_float = hs.float()
                mean = hs_float.mean(dim=-1, keepdim=True)
                var = hs_float.var(dim=-1, keepdim=True) + 1e-8
                diff = hs_float - mean
                fourth_moment = (diff ** 4).mean(dim=-1, keepdim=True)
                kurt = (fourth_moment / (var ** 2)).mean().item()
                norm = hs_float.norm(dim=-1).mean().item()
                all_kurtosis.append(kurt)
                all_norms.append(norm)
    
    mean_kurtosis = float(np.mean(all_kurtosis))
    max_kurtosis = float(np.max(all_kurtosis))
    mean_norm = float(np.mean(all_norms))
    
    # Derive quantization_hostility from kurtosis (simplified PRISM formula)
    # qh = sigmoid(log(kurtosis / 100)) — maps kurtosis into [0, 1]
    import math
    qh = 1.0 / (1.0 + math.exp(-math.log(max(mean_kurtosis / 100.0, 1e-8))))
    
    result = {
        "model": "haic-gemma4-v3-hybrid",
        "track": "phase3_hybrid",
        "quantization_hostility": round(qh, 4),
        "mean_kurtosis": round(mean_kurtosis, 2),
        "max_kurtosis": round(max_kurtosis, 2),
        "mean_activation_norm": round(mean_norm, 4),
        "num_layers_sampled": len(all_kurtosis),
        "calibration_prompts": len(calibration_prompts),
        "note": "Post-training PRISM extraction via phase3 hybrid track"
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"PRISM metrics saved to {output_path}")
    print(f"  quantization_hostility: {result['quantization_hostility']}")
    print(f"  mean_kurtosis: {result['mean_kurtosis']}")
    print(f"  max_kurtosis: {result['max_kurtosis']}")
    return result


if __name__ == "__main__":
    import numpy as np
    
    print("Initiating Phase 3: Hybrid Geometric (Targeted + AuxLoss) Track")
    
    # 1. Configuration — detect Kaggle vs local
    model_id = "google/gemma-4-E2B-it"
    
    # Check for Kaggle model path (attached as Kaggle Model)
    kaggle_model_path = "/kaggle/input/gemma-4/transformers/gemma-4-E2B-it/1"
    if os.path.exists(kaggle_model_path):
        model_id = kaggle_model_path
        print(f"Using Kaggle-attached model: {model_id}")
    else:
        print(f"Using HuggingFace model: {model_id}")
    
    # Data path detection
    data_path = "grounding_gemma4_v2.jsonl"
    kaggle_data_candidates = [
        "/kaggle/input/haic-gemma4-data/grounding_gemma4_v2.jsonl",
        "/kaggle/input/haic-kaggle-utils/grounding_gemma4_v2.jsonl",
    ]
    if not os.path.exists(data_path):
        for candidate in kaggle_data_candidates:
            if os.path.exists(candidate):
                data_path = candidate
                break
    print(f"Training data: {data_path}")
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    
    # r=128 universally across all projection matrices for maximum geometric override capacity
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        output_dir="./outputs_hybrid",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=5,
        num_train_epochs=3.0,  # Bumped from 2 to 3 for deeper geometry convergence
        fp16=True, 
        remove_unused_columns=False,
    )

    trainer = GeometryHybridTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Executing Hybrid Geometry Training...")
    trainer.train()
    print("Hybrid Training complete.")
    
    # Save the LoRA adapters
    adapter_output = "/kaggle/working/haic-gemma4-v3-hybrid"
    if not os.path.exists("/kaggle/working"):
        adapter_output = "./haic-gemma4-v3-hybrid"
    
    trainer.model.save_pretrained(adapter_output)
    tokenizer.save_pretrained(adapter_output)
    print(f"Adapters saved to {adapter_output}")
    
    # Extract PRISM metrics post-training
    prism_output = os.path.join(os.path.dirname(adapter_output), "prism_gemma4_v3_hybrid.json")
    extract_prism_metrics(trainer.model, tokenizer, prism_output)
