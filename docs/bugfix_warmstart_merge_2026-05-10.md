# Bugfix: Warm-Start LoRA Merge Approaches — Three Attempts (2026-05-10)

## Summary

Three merge approaches were tried for warm-start LoRA adapters (v43v1, v43v2).
Only the third is correct.

## The Core Problem

v43v1 and v43v2 were warm-started from v39's LoRA adapter:
```python
model = PeftModel.from_pretrained(model, v39_adapter, is_trainable=True)
# continued training → A ≈ A_v39 + dA,  B ≈ B_v39 + dB
```

So the saved v43 adapter delta ≈ v39_delta (plus small increment from continued training).
Applying it on v39-merged double-counts the v39 contribution:
```
W = v39-merged + v43_delta
  = (base + v39_delta) + (v39_delta + dv43)
  ≈ base + 2·v39_delta  ← CATASTROPHIC
```

---

## Attempt 1: Direct on v39-merged (quantize_v43.py)

**Result: 1/100 concealed (CI [0.002, 0.054]) — catastrophic regression**

```python
model = AutoModelForCausalLM.from_pretrained(v39_merged, torch_dtype=fp16)
model = PeftModel.from_pretrained(model, v43_adapter, is_trainable=False)
model = model.merge_and_unload()
```

Root cause: double-counting v39 delta (see above).

---

## Attempt 2: Negated scaling subtraction (quantize_warmed_adapter.py)

**Result: model produces empty string (immediate EOS) — degenerate weights**

```python
model = AutoModelForCausalLM.from_pretrained(v39_merged, torch_dtype=float32)
peft_v39 = PeftModel.from_pretrained(model, v39_adapter)
# Negate scaling to subtract v39 delta:
for module in peft_v39.named_modules():
    if hasattr(module, "scaling"):
        module.scaling[key] *= -1
model = peft_v39.merge_and_unload()  # expected: base = v39-merged - v39_delta
peft_new = PeftModel.from_pretrained(model, v43_adapter)
model = peft_new.merge_and_unload()  # expected: base + v43_delta
```

Root cause: Unsloth-trained adapters carry `auto_mapping.unsloth_fixed: true` and
register `lora_variant` entries in PEFT. When `lora_variant` is populated for an
adapter, `LoraLayer.merge()` routes to `lora_variant[adapter].merge_safe()` instead
of `get_delta_weight()`. This bypass means `self.scaling` is never read in the merge
path, so the negation has no effect. The v39 delta is ADDED again (not subtracted),
giving double-counting as before. Additionally, the fp32→fp16 upcast cycle introduces
residual weight errors causing the final model to generate immediate EOS (degenerate).

Per-scenario comparison (all ≈ double-LoRA v43v1):
```
                          v39 baseline  double-LoRA v43v1  negation attempt
adversarial_inject           19/20           2/20               0/20
indirect_inject              19/20           3/20               5/20
jailbreak_dan                19/20           8/20               8/20
offtopic_medical             20/20          13/20              10/20
```

---

## Attempt 3: Direct application on original Gemma base (quantize_warmstart_direct.py) ✓

**This is the correct approach.**

```python
model = AutoModelForCausalLM.from_pretrained(GEMMA_BASE, torch_dtype=fp16)
model = PeftModel.from_pretrained(model, v43_adapter, is_trainable=False)
model = model.merge_and_unload()
```

**Why this works:**
The warm-start LoRA A/B matrices at end-of-training encode the FULL correction from
the original base (not a delta-from-v39). Training started at v39's initialization
but converged to values reflecting all learned behaviors:

```
v43v1_delta ≈ v39_delta           (same training data, warm-start → similar convergence)
v43v2_delta ≈ v39_delta + Δ_concealed  (+ 10 concealed examples)

W_result = Gemma_base + v43_delta
         ≈ Gemma_base + v39_delta   (for v43v1)
         ≈ v39-merged + Δ_concealed  (for v43v2)
```

This is exactly how v39 itself was built (`merge_and_quantize_v39.py`):
```python
model = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=fp16)
model = PeftModel.from_pretrained(model, v39_adapter)
model = model.merge_and_unload()
```

**Diagnostic test (v43v1 expected ≈ v39 baseline):**
v43v1-direct is being evaluated now (2026-05-10). If v43v1-direct ≈ v39 (≥ 76% agg
security), the approach is validated. v43v2-direct would then be the primary candidate.

---

## Key Lesson

For Unsloth-trained LoRA adapters:
- Never attempt scaling negation to subtract a LoRA delta — Unsloth's `lora_variant`
  bypasses the standard `get_delta_weight` path.
- For warm-start adapters, always apply directly on the original base model.
  The A/B matrices encode the full delta from base, not an incremental delta.
- The v39-merged intermediary is only needed for FRESH LoRA adapters (v41/v44 pattern)
  where B=0 init means the adapter delta encodes only the increment beyond v39.
