"""
rank_geometry_sweep.py — Experiment 2: LoRA rank vs PRISM geometry phase transition.

Investigates why LoRA fine-tuning does NOT change activation geometry,
despite empirical evidence that some training runs DO shift it.

Design:
  For each LoRA rank r in [4, 8, 16, 32, 64, 128, 256]:
    1. Train a LoRA adapter on the same dataset (same seed, same steps)
    2. Measure PRISM qh, outlier_ratio, kurtosis, cardinal_proximity post-training
    3. Compute the geometry delta from the base model

If there is a critical rank where geometry starts moving, this reveals the
dimensionality threshold for the geometric subspace.  If geometry is flat
across all ranks, the invariance is structural and the PRISM-measured features
live outside the LoRA-modifiable subspace.

Additional factor: target_modules.  We sweep over two configurations:
  A. ["q_proj", "v_proj"]           — standard HAIC config (v34/v35-gov)
  B. All 7 projections              — maximum override capacity (phase3 hybrid)

Target environment: Kaggle T4 x2 or Colab A100
Estimated wall-clock: 4-6 hours (14 training runs × ~20 min each)
"""

import os
import sys
import json
import math
import time
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class RankSweepConfig:
    model_id: str = "google/gemma-4-E2B-it"
    data_path: str = "grounding_gemma4_v2.jsonl"
    output_dir: str = "./experiments/rank_geometry_sweep"
    ranks: list = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])
    target_module_configs: dict = field(default_factory=lambda: {
        "standard": ["q_proj", "v_proj"],
        "full": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    })
    num_train_epochs: float = 1.0
    lr: float = 5e-5
    batch_size: int = 2
    grad_accum_steps: int = 4
    max_seq_len: int = 512
    seed: int = 42
    # PRISM calibration
    calibration_prompts: list = field(default_factory=lambda: [
        "Evaluate the ethical implications of autonomous decision-making in healthcare.",
        "What are the risks of deploying AI without human oversight in education?",
        "Describe the consent requirements for using personal data in model training.",
        "How should communities affected by environmental monitoring be consulted?",
        "What governance structures protect vulnerable populations from AI harm?",
    ])


def extract_full_geometry(model, tokenizer, prompts: list) -> dict:
    """
    Extract the full PRISM geometry profile from a model.

    Returns: dict with outlier_ratio, activation_kurtosis, cardinal_proximity,
             quantization_hostility, per_layer_kurtosis, worst_layer_zone.
    """
    import torch
    import numpy as np

    all_layer_kurtosis = {}
    all_layer_norms = {}

    model.eval()
    with torch.no_grad():
        for prompt_text in prompts:
            inputs = tokenizer(prompt_text, return_tensors="pt",
                               truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)

            for layer_idx, hs in enumerate(outputs.hidden_states):
                hs_float = hs.float().squeeze(0)  # [seq_len, hidden_dim]

                # Kurtosis
                mean = hs_float.mean(dim=-1, keepdim=True)
                var = hs_float.var(dim=-1, keepdim=True) + 1e-8
                diff = hs_float - mean
                fourth_moment = (diff ** 4).mean(dim=-1, keepdim=True)
                kurt = (fourth_moment / (var ** 2)).mean().item()

                # Activation norms
                norms = hs_float.norm(dim=-1)
                max_norm = norms.max().item()
                median_norm = norms.median().item()

                if layer_idx not in all_layer_kurtosis:
                    all_layer_kurtosis[layer_idx] = []
                    all_layer_norms[layer_idx] = []
                all_layer_kurtosis[layer_idx].append(kurt)
                all_layer_norms[layer_idx].append((max_norm, median_norm))

    # Aggregate per-layer
    per_layer = {}
    for layer_idx in sorted(all_layer_kurtosis.keys()):
        k_vals = all_layer_kurtosis[layer_idx]
        n_vals = all_layer_norms[layer_idx]
        mean_k = float(np.mean(k_vals))
        max_ratios = [m / med for m, med in n_vals if med > 0]
        mean_ratio = float(np.mean(max_ratios)) if max_ratios else 0.0
        per_layer[layer_idx] = {
            "kurtosis": round(mean_k, 2),
            "outlier_ratio": round(mean_ratio, 2),
        }

    # Global aggregates
    all_k = [v["kurtosis"] for v in per_layer.values()]
    all_r = [v["outlier_ratio"] for v in per_layer.values()]

    mean_kurtosis = float(np.mean(all_k))
    mean_outlier = float(np.mean(all_r))

    # Cardinal proximity: fraction of variance in top-4 PCA components
    # Simplified: use kurtosis ratio as proxy
    cardinal_proximity = min(1.0, mean_kurtosis / 2000.0)

    # Quantization hostility
    qh = 1.0 / (1.0 + math.exp(-math.log(max(mean_kurtosis / 100.0, 1e-8))))

    # Worst layer zone
    worst_layer = max(per_layer.keys(), key=lambda k: per_layer[k]["kurtosis"])
    n_layers = len(per_layer)
    if worst_layer < n_layers * 0.33:
        worst_zone = "early"
    elif worst_layer < n_layers * 0.67:
        worst_zone = "middle"
    else:
        worst_zone = "late"

    return {
        "outlier_ratio": round(mean_outlier, 2),
        "activation_kurtosis": round(mean_kurtosis, 2),
        "cardinal_proximity": round(cardinal_proximity, 4),
        "quantization_hostility": round(qh, 4),
        "worst_layer_zone": worst_zone,
        "worst_layer_idx": worst_layer,
        "n_layers": n_layers,
        "per_layer_kurtosis": {str(k): v["kurtosis"] for k, v in per_layer.items()},
    }


def run_sweep(config: RankSweepConfig):
    """Execute the full rank-geometry sweep."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    os.makedirs(config.output_dir, exist_ok=True)

    print("=== Rank-Geometry Sweep ===")
    print(f"Ranks: {config.ranks}")
    print(f"Module configs: {list(config.target_module_configs.keys())}")

    # ── Load model ────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    kaggle_path = "/kaggle/input/gemma-4/transformers/gemma-4-E2B-it/1"
    model_path = kaggle_path if os.path.exists(kaggle_path) else config.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )

    # ── Baseline geometry ─────────────────────────────────────────────────
    print("\nMeasuring baseline geometry...")
    baseline = extract_full_geometry(base_model, tokenizer, config.calibration_prompts)
    print(f"  Baseline qh: {baseline['quantization_hostility']}")
    print(f"  Baseline kurtosis: {baseline['activation_kurtosis']}")

    # ── Load dataset ──────────────────────────────────────────────────────
    data_path = config.data_path
    kaggle_data = "/kaggle/input/haic-gemma4-data/grounding_gemma4_v2.jsonl"
    if not os.path.exists(data_path) and os.path.exists(kaggle_data):
        data_path = kaggle_data

    dataset = load_dataset("json", data_files=data_path, split="train")

    def formatting_func(example):
        output_texts = []
        for i in range(len(example.get("system", example.get("prompt", [])))):
            system = example.get("system", [""])[i] if "system" in example else ""
            prompt = example.get("prompt", example.get("instruction", [""]))[i]
            response = example["response"][i]
            messages = [
                {"role": "user", "content": f"{system}\n{prompt}"},
                {"role": "assistant", "content": response}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            output_texts.append(text)
        return output_texts

    # ── Sweep ─────────────────────────────────────────────────────────────
    results = {
        "config": asdict(config),
        "baseline": baseline,
        "runs": [],
    }

    for config_name, target_modules in config.target_module_configs.items():
        for rank in config.ranks:
            run_id = f"{config_name}_r{rank}"
            print(f"\n--- {run_id}: rank={rank}, modules={target_modules} ---")

            peft_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.01,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )

            model = get_peft_model(base_model, peft_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

            output_subdir = os.path.join(config.output_dir, run_id)
            training_args = SFTConfig(
                output_dir=output_subdir,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.grad_accum_steps,
                learning_rate=config.lr,
                num_train_epochs=config.num_train_epochs,
                logging_steps=10,
                save_strategy="no",
                fp16=True,
                remove_unused_columns=False,
                seed=config.seed,
                max_seq_length=config.max_seq_len,
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=peft_config,
                formatting_func=formatting_func,
                tokenizer=tokenizer,
                args=training_args,
            )

            t0 = time.time()
            trainer.train()
            train_time = time.time() - t0

            # Measure post-training geometry
            print(f"  Measuring post-training geometry...")
            post_geometry = extract_full_geometry(model, tokenizer, config.calibration_prompts)

            # Compute deltas
            delta_qh = post_geometry["quantization_hostility"] - baseline["quantization_hostility"]
            delta_kurt = post_geometry["activation_kurtosis"] - baseline["activation_kurtosis"]
            delta_outlier = post_geometry["outlier_ratio"] - baseline["outlier_ratio"]

            run_result = {
                "run_id": run_id,
                "rank": rank,
                "target_modules": target_modules,
                "config_name": config_name,
                "trainable_params": trainable,
                "trainable_pct": round(100 * trainable / total, 4),
                "train_time_s": round(train_time, 1),
                "geometry_post": post_geometry,
                "delta_qh": round(delta_qh, 6),
                "delta_kurtosis": round(delta_kurt, 2),
                "delta_outlier_ratio": round(delta_outlier, 2),
            }
            results["runs"].append(run_result)

            print(f"  qh: {baseline['quantization_hostility']:.4f} → "
                  f"{post_geometry['quantization_hostility']:.4f} (Δ={delta_qh:+.6f})")
            print(f"  kurtosis: {baseline['activation_kurtosis']:.1f} → "
                  f"{post_geometry['activation_kurtosis']:.1f} (Δ={delta_kurt:+.1f})")
            print(f"  Training time: {train_time:.0f}s")

            # Clean up for next run
            del model, trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Save results ──────────────────────────────────────────────────────
    output_path = os.path.join(config.output_dir, "rank_sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'Run':>20} {'Rank':>6} {'Params':>12} {'qh':>8} {'Δqh':>10} {'Δkurt':>10}")
    print("-" * 75)
    for run in results["runs"]:
        print(f"{run['run_id']:>20} {run['rank']:>6} "
              f"{run['trainable_params']:>12,} "
              f"{run['geometry_post']['quantization_hostility']:>8.4f} "
              f"{run['delta_qh']:>+10.6f} "
              f"{run['delta_kurtosis']:>+10.1f}")

    # ── Phase transition detection ────────────────────────────────────────
    print("\n=== PHASE TRANSITION ANALYSIS ===")
    for config_name in config.target_module_configs:
        runs = [r for r in results["runs"] if r["config_name"] == config_name]
        deltas = [(r["rank"], abs(r["delta_qh"])) for r in runs]

        # Find rank where |Δqh| first exceeds 0.01 (1% of full range)
        threshold = 0.01
        transition_rank = None
        for rank, dqh in deltas:
            if dqh > threshold:
                transition_rank = rank
                break

        if transition_rank:
            print(f"  {config_name}: Phase transition at r={transition_rank} "
                  f"(|Δqh| > {threshold})")
        else:
            print(f"  {config_name}: NO phase transition detected "
                  f"(geometry invariant across all ranks)")
            print(f"    Max |Δqh|: {max(d for _, d in deltas):.6f}")


if __name__ == "__main__":
    config = RankSweepConfig()
    run_sweep(config)
