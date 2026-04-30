"""
temporal_signature.py — Experiment 1: Validate the autophagy temporal signature.

Tests the core prediction of the Viability Condition framework:
  "OOD accuracy degrades BEFORE validation perplexity rises"
  as synthetic data ratio increases.

Design:
  For each synthetic_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
    1. Blend the training dataset with synthetic data at the given ratio
    2. Fine-tune a LoRA adapter on the blended dataset
    3. At each checkpoint (every N steps):
       a. Measure validation perplexity on held-out in-distribution data
       b. Measure OOD accuracy on an out-of-distribution probe set
       c. Measure PRISM qh on the current model state
    4. Record the step at which OOD accuracy drops below baseline
    5. Record the step at which validation perplexity rises above baseline

If the temporal signature is real, the OOD-accuracy drop step < perplexity-rise
step consistently across synthetic ratios.

Target environment: Kaggle T4 x2 or Colab A100
Estimated wall-clock: 4-8 hours depending on dataset size and checkpoint frequency
"""

import os
import sys
import json
import math
import time
import random
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TemporalSignatureConfig:
    """Experiment configuration."""
    model_id: str = "google/gemma-4-E2B-it"
    real_data_path: str = "grounding_gemma4_v2.jsonl"
    ood_probe_path: str = "ood_probe.jsonl"       # must be created separately
    output_dir: str = "./experiments/temporal_signature"
    synthetic_ratios: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    checkpoint_every_n_steps: int = 10
    max_steps: int = 100
    lr: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    batch_size: int = 2
    grad_accum_steps: int = 4
    max_seq_len: int = 512
    seed: int = 42


# ── Synthetic data generation ────────────────────────────────────────────────

def generate_synthetic_session(real_session: dict, model, tokenizer) -> dict:
    """
    Generate a synthetic training pair by having the model predict
    the response given the instruction.  The synthetic pair has the
    same structure as a real one but the response is model-generated.

    This is the mechanism by which informational autophagy enters:
    the model trains on its own outputs.
    """
    prompt = real_session.get("prompt", real_session.get("instruction", ""))
    system = real_session.get("system", "")

    messages = [{"role": "user", "content": f"{system}\n{prompt}"}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=256, do_sample=True,
            temperature=0.7, top_p=0.9
        )

    synthetic_response = tokenizer.decode(
        outputs[0][inputs.shape[1]:], skip_special_tokens=True
    )

    return {
        **real_session,
        "response": synthetic_response,
        "is_synthetic": True,
    }


def blend_dataset(real_data: list, synthetic_data: list, ratio: float,
                  seed: int = 42) -> list:
    """
    Blend real and synthetic data at the given ratio.

    ratio=0.0 → all real data
    ratio=0.5 → 50% real, 50% synthetic
    ratio=1.0 → all synthetic data (not recommended for this experiment)
    """
    rng = random.Random(seed)
    n_total = len(real_data)
    n_synthetic = int(n_total * ratio)
    n_real = n_total - n_synthetic

    selected_real = rng.sample(real_data, min(n_real, len(real_data)))
    selected_synthetic = rng.sample(synthetic_data, min(n_synthetic, len(synthetic_data)))

    blended = selected_real + selected_synthetic
    rng.shuffle(blended)
    return blended


# ── Measurement functions ────────────────────────────────────────────────────

def measure_perplexity(model, tokenizer, eval_data: list, max_samples: int = 50) -> float:
    """
    Compute perplexity on a held-out eval set.

    Returns the average perplexity across samples.
    """
    import torch

    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for sample in eval_data[:max_samples]:
            prompt = sample.get("prompt", sample.get("instruction", ""))
            response = sample.get("response", "")
            text = f"{prompt}\n{response}"

            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            n_samples += 1

    avg_loss = total_loss / max(n_samples, 1)
    return math.exp(avg_loss)


def measure_ood_accuracy(model, tokenizer, ood_probes: list,
                         max_samples: int = 50) -> float:
    """
    Measure accuracy on out-of-distribution probes.

    Each probe has a question and expected answer category.  We measure
    whether the model's top response matches the expected category.

    Returns: accuracy as a float in [0, 1].
    """
    import torch

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for probe in ood_probes[:max_samples]:
            question = probe["question"]
            expected = probe["expected_category"].lower()

            messages = [{"role": "user", "content": question}]
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(model.device)

            outputs = model.generate(inputs, max_new_tokens=64, do_sample=False)
            response = tokenizer.decode(
                outputs[0][inputs.shape[1]:], skip_special_tokens=True
            ).lower()

            if expected in response:
                correct += 1
            total += 1

    return correct / max(total, 1)


def measure_prism_qh(model, tokenizer, calibration_prompts: list = None) -> float:
    """
    Compute PRISM quantization_hostility from hidden states.

    Uses the kurtosis-based qh formula from the PRISM toolkit.
    """
    import torch
    import numpy as np

    if calibration_prompts is None:
        calibration_prompts = [
            "Evaluate the ethical implications of autonomous decision-making.",
            "What are the risks of deploying AI without human oversight?",
            "Describe the consent requirements for using personal data.",
        ]

    all_kurtosis = []

    model.eval()
    with torch.no_grad():
        for prompt_text in calibration_prompts:
            inputs = tokenizer(prompt_text, return_tensors="pt",
                               truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)

            for hs in outputs.hidden_states:
                hs_float = hs.float()
                mean = hs_float.mean(dim=-1, keepdim=True)
                var = hs_float.var(dim=-1, keepdim=True) + 1e-8
                diff = hs_float - mean
                fourth_moment = (diff ** 4).mean(dim=-1, keepdim=True)
                kurt = (fourth_moment / (var ** 2)).mean().item()
                all_kurtosis.append(kurt)

    mean_kurtosis = float(np.mean(all_kurtosis))
    qh = 1.0 / (1.0 + math.exp(-math.log(max(mean_kurtosis / 100.0, 1e-8))))
    return qh


# ── Checkpoint callback ─────────────────────────────────────────────────────

@dataclass
class CheckpointMeasurement:
    """Measurement at a single training checkpoint."""
    step: int
    synthetic_ratio: float
    val_perplexity: float
    ood_accuracy: float
    prism_qh: float
    timestamp: str


def run_experiment(config: TemporalSignatureConfig):
    """
    Run the full temporal signature experiment.

    For each synthetic ratio, trains a fresh adapter and records measurements
    at each checkpoint.  Saves results as JSON for analysis.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import load_dataset

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"=== Temporal Signature Experiment ===")
    print(f"Model: {config.model_id}")
    print(f"Synthetic ratios: {config.synthetic_ratios}")
    print(f"Checkpoint every {config.checkpoint_every_n_steps} steps")
    print(f"Max steps: {config.max_steps}")

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

    # ── Load data ─────────────────────────────────────────────────────────
    with open(config.real_data_path) as f:
        real_data = [json.loads(line) for line in f if line.strip()]

    # Hold out 10% for validation
    random.seed(config.seed)
    random.shuffle(real_data)
    split_idx = int(len(real_data) * 0.9)
    train_data = real_data[:split_idx]
    val_data = real_data[split_idx:]

    # Load or create OOD probes
    if os.path.exists(config.ood_probe_path):
        with open(config.ood_probe_path) as f:
            ood_probes = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"WARNING: OOD probe file {config.ood_probe_path} not found.")
        print("Create it with questions/expected_category pairs from domains")
        print("NOT represented in the training data.")
        ood_probes = []

    # ── Generate synthetic data ───────────────────────────────────────────
    print("Generating synthetic data from base model...")
    synthetic_data = []
    for i, sample in enumerate(train_data[:len(train_data) // 2]):
        if i % 20 == 0:
            print(f"  Generating synthetic sample {i}...")
        synthetic_data.append(generate_synthetic_session(sample, base_model, tokenizer))
    print(f"Generated {len(synthetic_data)} synthetic samples.")

    # ── Baseline measurements ─────────────────────────────────────────────
    print("Computing baseline measurements...")
    baseline_ppl = measure_perplexity(base_model, tokenizer, val_data)
    baseline_ood = measure_ood_accuracy(base_model, tokenizer, ood_probes) if ood_probes else None
    baseline_qh = measure_prism_qh(base_model, tokenizer)

    print(f"  Baseline perplexity: {baseline_ppl:.2f}")
    print(f"  Baseline OOD accuracy: {baseline_ood}")
    print(f"  Baseline PRISM qh: {baseline_qh:.4f}")

    all_results = {
        "config": asdict(config),
        "baseline": {
            "val_perplexity": baseline_ppl,
            "ood_accuracy": baseline_ood,
            "prism_qh": baseline_qh,
        },
        "runs": [],
    }

    # ── Run per synthetic ratio ───────────────────────────────────────────
    for ratio in config.synthetic_ratios:
        print(f"\n--- Synthetic ratio: {ratio:.1%} ---")

        blended = blend_dataset(train_data, synthetic_data, ratio, seed=config.seed)
        print(f"  Blended dataset: {len(blended)} samples "
              f"({sum(1 for d in blended if d.get('is_synthetic'))} synthetic)")

        # Fresh adapter for each ratio
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(base_model, peft_config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        measurements = []

        for step in range(1, config.max_steps + 1):
            # Training step
            model.train()
            sample = blended[step % len(blended)]
            prompt = sample.get("prompt", sample.get("instruction", ""))
            response = sample.get("response", "")
            system = sample.get("system", "")

            messages = [
                {"role": "user", "content": f"{system}\n{prompt}"},
                {"role": "assistant", "content": response}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=config.max_seq_len).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Checkpoint measurement
            if step % config.checkpoint_every_n_steps == 0:
                print(f"  Step {step}: loss={loss.item():.4f}, measuring...")
                ppl = measure_perplexity(model, tokenizer, val_data)
                ood = measure_ood_accuracy(model, tokenizer, ood_probes) if ood_probes else None
                qh = measure_prism_qh(model, tokenizer)

                m = CheckpointMeasurement(
                    step=step,
                    synthetic_ratio=ratio,
                    val_perplexity=ppl,
                    ood_accuracy=ood,
                    prism_qh=qh,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                measurements.append(asdict(m))
                print(f"    ppl={ppl:.2f} ood={ood} qh={qh:.4f}")

        # ── Find temporal signature ───────────────────────────────────────
        ood_drop_step = None
        ppl_rise_step = None

        for m in measurements:
            if ood_drop_step is None and m["ood_accuracy"] is not None:
                if m["ood_accuracy"] < (baseline_ood * 0.95 if baseline_ood else float('inf')):
                    ood_drop_step = m["step"]
            if ppl_rise_step is None:
                if m["val_perplexity"] > baseline_ppl * 1.05:
                    ppl_rise_step = m["step"]

        temporal_lag = None
        if ood_drop_step is not None and ppl_rise_step is not None:
            temporal_lag = ppl_rise_step - ood_drop_step

        run_result = {
            "synthetic_ratio": ratio,
            "measurements": measurements,
            "ood_drop_step": ood_drop_step,
            "ppl_rise_step": ppl_rise_step,
            "temporal_lag": temporal_lag,
            "signature_present": temporal_lag is not None and temporal_lag > 0,
        }
        all_results["runs"].append(run_result)

        sig_str = f"lag={temporal_lag} steps" if temporal_lag else "NOT DETECTED"
        print(f"  Temporal signature: {sig_str}")

        # Clean up adapter for next ratio
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Save results ──────────────────────────────────────────────────────
    output_path = os.path.join(config.output_dir, "temporal_signature_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== TEMPORAL SIGNATURE SUMMARY ===")
    print(f"{'Ratio':>8} {'OOD Drop':>10} {'PPL Rise':>10} {'Lag':>6} {'Signature':>10}")
    for run in all_results["runs"]:
        r = run["synthetic_ratio"]
        od = str(run["ood_drop_step"] or "—")
        pr = str(run["ppl_rise_step"] or "—")
        lag = str(run["temporal_lag"] or "—")
        sig = "YES" if run["signature_present"] else "no"
        print(f"{r:>8.1%} {od:>10} {pr:>10} {lag:>6} {sig:>10}")


if __name__ == "__main__":
    config = TemporalSignatureConfig()

    # Kaggle path detection
    kaggle_data = "/kaggle/input/haic-gemma4-data/grounding_gemma4_v2.jsonl"
    if os.path.exists(kaggle_data):
        config.real_data_path = kaggle_data

    run_experiment(config)
