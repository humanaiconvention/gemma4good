"""
run_v38_sgt.py — wire Garrett's sgt_harness to the v38 adapter on BEAST.

Loads google/gemma-4-E2B-it from the local HF cache, applies the v38 LoRA
adapter via PEFT, and runs the rigorous SGT harness (deterministic +
sampling pass with Wilson 95% CI).

This is intentionally separate from sgt_harness.py so the harness itself
stays unmodified — the harness file is the upstream candidate; this file
is the local adapter wiring.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from experiments.sgt_harness import make_hf_backend, run_sgt, format_report


# v38 system prompt — taken verbatim from D:/kaggle/scripts/build_v38_nb.py
V38_SYSTEM_PROMPT = (
    "You are a gentle, curious interviewer for the Human-AI Convention. "
    "Follow the ESTABLISH-PIVOT-DEEPEN protocol exactly: "
    "(1) ESTABLISH: ask 1-2 open questions to understand the participant's context. "
    "(2) PIVOT: when ready to focus on a specific moment, begin your message with the "
    "exact tag [PIVOT: DEEPEN] — this literal bracketed text is required protocol, "
    "not a suggestion. Never use **bold** or other formatting instead of this tag. "
    "(3) DEEPEN: ask about one specific moment, memory, or feeling. "
    "Never answer questions unrelated to the interview."
)


def load_model(base_path: str, adapter_path: str | None, four_bit: bool):
    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if four_bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        m = AutoModelForCausalLM.from_pretrained(
            base_path, quantization_config=bnb,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        m = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

    if adapter_path:
        m = PeftModel.from_pretrained(m, adapter_path)
    m.eval()
    return m, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3")
    ap.add_argument("--adapter", default="D:/kaggle/adapters/haic-gemma4-v38-adapter/haic-gemma4-v38-adapter")
    ap.add_argument("--baseline", action="store_true",
                    help="Also run the un-fine-tuned base model and report Δ")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--four-bit", action="store_true", default=True,
                    help="Load in 4-bit nf4 (matches kaggle training; saves VRAM). Default True.")
    ap.add_argument("--no-four-bit", dest="four_bit", action="store_false")
    ap.add_argument("--out", default="experiments/v38_sgt_rigorous.json")
    args = ap.parse_args()

    decoding = dict(
        temperature=args.temperature, top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        precision="4bit_nf4" if args.four_bit else "fp16",
    )

    print("=" * 60)
    print(f"v38 rigorous SGT — base={args.base}")
    print(f"                 adapter={args.adapter}")
    print(f"                 4bit={args.four_bit}  n_samples={args.n_samples}  seed={args.seed}")
    print("=" * 60)

    t0 = time.time()
    model, tok = load_model(args.base, args.adapter, args.four_bit)
    print(f"Loaded v38 in {time.time()-t0:.1f}s. VRAM allocated: "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB")

    backend = make_hf_backend(
        model, tok, system_prompt=V38_SYSTEM_PROMPT,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )

    print("\nRunning v38 harness...")
    t0 = time.time()
    v38_result = run_sgt(
        backend, n_samples=args.n_samples, seed=args.seed,
        model_id="haic-gemma4-v38", decoding=decoding,
    )
    print(f"v38 harness done in {time.time()-t0:.1f}s")
    print()
    print(format_report(v38_result))

    payload = {"finetune": v38_result}

    if args.baseline:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nLoading base model (no adapter)...")
        t0 = time.time()
        base_model, base_tok = load_model(args.base, None, args.four_bit)
        print(f"Loaded base in {time.time()-t0:.1f}s. VRAM allocated: "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB")

        base_backend = make_hf_backend(
            base_model, base_tok, system_prompt=V38_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
        )

        print("\nRunning base harness...")
        t0 = time.time()
        base_result = run_sgt(
            base_backend, n_samples=args.n_samples, seed=args.seed,
            model_id="google/gemma-4-E2B-it", decoding=decoding,
        )
        print(f"base harness done in {time.time()-t0:.1f}s")
        print()
        print("BASELINE:")
        print(format_report(base_result))

        payload["baseline"] = base_result

        delta = (
            v38_result["sampling"]["grounding_pass_rate"]
            - base_result["sampling"]["grounding_pass_rate"]
        )
        print(f"\nFine-tune Δ (sampling, grounding pass-rate): {delta:+.3f}")
        payload["finetune_delta_grounding_pass_rate"] = delta

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
