import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def calculate_kurtosis(hidden_states):
    '''Calculate kurtosis over the hidden dimension for a batch of activations.'''
    # hidden_states: [batch, seq_len, hidden_dim]
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True) + 1e-8
    
    # 4th moment
    diff = hidden_states - mean
    fourth_moment = (diff ** 4).mean(dim=-1, keepdim=True)
    
    # Kurtosis = 4th_moment / var^2
    kurtosis = fourth_moment / (var ** 2)
    return kurtosis.mean() # scalar mean over batch and sequence

class GeometryAuxLossTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # We must override compute_loss to inject output_hidden_states=True
        labels = inputs.pop("labels")
        
        # Forward pass returning hidden states
        outputs = model(**inputs, output_hidden_states=True)
        
        # Calculate standard causal LM cross-entropy
        logits = outputs.get("logits")
        # Standard shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Extract the hidden states from the last layer (or target AltUp layers)
        # outputs.hidden_states is a tuple where the last element is the final layer before LM head
        final_hidden_states = outputs.hidden_states[-1]
        
        # Compute the geometric penalty
        geom_penalty = calculate_kurtosis(final_hidden_states)
        
        # Add the penalty with lambda multiplier (e.g. 0.005)
        lambda_k = 0.005
        total_loss = ce_loss + (lambda_k * geom_penalty)
        
        # Log the components if you have wandb or standard logging enabled
        # self.log({"ce_loss": ce_loss.item(), "geom_penalty": geom_penalty.item()})
        
        return (total_loss, outputs) if return_outputs else total_loss

if __name__ == "__main__":
    print("Initiating Phase 3: Geometry AuxLoss Track")
    
    # 1. Configuration
    model_id = "google/gemma-4-E2B-it"
    data_path = "grounding_gemma4_v2.jsonl"
    
    # Kaggle fallback format
    if not os.path.exists(data_path):
        data_path = "/kaggle/input/haic-gemma4-data/grounding_gemma4_v2.jsonl"
        
    # 2. Loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
        output_dir="./outputs_auxloss",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=5,
        num_train_epochs=2.0,
        fp16=True, 
        remove_unused_columns=False, # Required for our custom forward pass
    )

    trainer = GeometryAuxLossTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Executing Custom AuxLoss Training...")
    # trainer.train()
    # printer("Training logic verified.")
