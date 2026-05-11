# NLA-Augmented Governance Pipeline (and the MPK Provenance Tool)

**Date:** 2026-05-11 (Tool 5 NLA + Tool 6 MPK landed same day)
**Status:** Tool 5 interface implemented (MockNLA in production; real Gemma-4
NLA blocked by training cost); Tool 6 implemented and ready for live MPK
data (when Cisco adds Gemma-4 to their reference dataset).

This document specifies how the HAIC governance pipeline incorporates the
Natural Language Autoencoder (NLA) technique. It complements three other
docs landed today:

- `docs/strict_rubric_finding_2026-05-11.md` — proxy/property drift in eval
- `docs/system_prompt_artifact_finding_2026-05-11.md` — prompt confounds
- `D:/prism/docs/NLA.md` — PRISM-side NLA design (parallel session)

---

## The five-tool governance pipeline

```
                                input
                                 │
                                 ▼
                ┌───────────────────────────────────┐
                │ Tool 1: assess_wellbeing_domain   │
                │   What population is affected?    │
                └───────────────┬───────────────────┘
                                ▼
                ┌───────────────────────────────────┐
                │ Tool 2: verify_consent_and        │
                │           _provenance             │
                │   Is the input allowed to inform  │
                │   action?                         │
                └───────────────┬───────────────────┘
                                ▼
                ┌───────────────────────────────────┐
                │ Tool 3: run_prism_analysis        │
                │   Is the model's GEOMETRY healthy │
                │   (quantization_hostility etc.)?  │
                └───────────────┬───────────────────┘
                                ▼
                ┌───────────────────────────────────┐
                │ Tool 4: audit_activation_         │
                │         explanation [NEW NLA]     │
                │   What is the model THINKING in   │
                │   natural language?               │
                └───────────────┬───────────────────┘
                                ▼
                ┌───────────────────────────────────┐
                │ Tool 5: generate_alignment_       │
                │         receipt                   │
                │   Anchor the trace (Merkle root + │
                │   ZK digest)                      │
                └───────────────┬───────────────────┘
                                ▼
                ┌───────────────────────────────────┐
                │ Tool 6: audit_provenance          │
                │         [NEW MPK, advisory]       │
                │   Is the model statistically      │
                │   derived from its claimed parent?│
                │   (Cisco's Model Provenance Kit)  │
                │   NOT cryptographic proof.        │
                └───────────────┬───────────────────┘
                                ▼
                            receipt + provenance verdict
```

Tools 1-5 produce the canonical governance receipt. Tool 6 is **advisory**
— it adds third-party-tooling corroboration of the model's structural
derivation, recorded alongside the receipt but not part of it.

Tools 3 and 4 are the **interpretability pair**. Tool 3 (PRISM) measures
the GEOMETRY of the model's hidden states — abstract numerical properties
like outlier ratios and quantization hostility. Tool 4 (NLA) measures the
SEMANTICS of those same hidden states — a natural-language description
of what the model is reasoning about.

Geometry alone tells you the layer's *shape*. NLA alone tells you the
layer's *content* but provides no anchor for the shape claim. Used
together, they cross-validate: a high-hostility layer with low-FVE
explanations is one we should be uncertain about; a low-hostility layer
with a confident NLA explanation we can trust.

---

## Tool 4 contract

Schema: `tools/audit_activation_explanation.py::TOOL_SCHEMA`

```json
{
  "name": "audit_activation_explanation",
  "description": "Use the Natural Language Autoencoder to explain what the model is internally reasoning about, in natural language...",
  "parameters": {
    "type": "object",
    "properties": {
      "scenario_id": {"type": "string"},
      "layer_idx":   {"type": "integer"},
      "nla_model_id":{"type": "string", "default": "mock"}
    },
    "required": ["scenario_id", "layer_idx"]
  }
}
```

The LLM in the function-calling loop can request this tool whenever it
wants natural-language insight into the model's internal state at a
specific layer. The notebook supplies the activation vector from a
preceding PRISM call (it's an implementation detail — the LLM doesn't
see raw activations).

Tool result is added as a Merkle leaf in the governance receipt:

```python
{
  "scenario_id":        "amazon_cell_4521",
  "layer_idx":          32,
  "explanation_text":   "The activation is dominated by features that look like geographic location reasoning. ...",
  "reconstruction_fve": 0.67,
  "nla_model_id":       "kitft/nla-gemma-3-12b-it-layer32",
  "activation_norm":    1.0,
  "confidence_class":   "high",
  "audit_hash":         "<SHA3-256>",
  "raw_explanation":    { ... full NLAExplanation dict ... }
}
```

The `confidence_class` ("high" | "medium" | "low" | "mock") is the
operator-facing summary:

  - **high** (FVE ≥ 0.60): trust the explanation
  - **medium** (0.40 ≤ FVE < 0.60): treat as a hint
  - **low** (FVE < 0.40): the AV may be confabulating; cross-check
  - **mock**: this is a MockNLA output; treat as testing-only

The `audit_hash` seals the explanation-relevant payload so a tampered
NLA output is detectable downstream — same anchoring discipline as the
existing four governance tools.

---

## Honest Scope (read this carefully)

### What works today

  - `prism_integration/nla.py::MockNLA` produces deterministic
    explanations. 15/15 tests pass. Wired through
    `execute_audit_activation_explanation`. Composes with the existing
    notebook governance loop.
  - `prism_integration/nla.py::get_explainer(...)` will return a REAL
    explainer the moment a transport (HTTP server or callable) is
    provided AND the `model_id` is in PRISM's registry. The current
    registry includes `kitft/nla-{qwen2.5-7b-instruct,gemma-3-12b-it,
    gemma-3-27b-it,llama-3.3-70b-instruct}-layer*`.
  - End-to-end test with a real `prism.nla.NLAExplainer` (transport
    stubbed) passes. The interface is verified, not just stubbed.

### What does NOT work today

  - **No Gemma-4-E2B NLA exists.** The closest published NLA is for
    Gemma-3-12B-IT (different architecture, different size, different
    d_model). Using a Gemma-3 NLA on Gemma-4 activations is
    methodologically broken — the AR's affine map was learned on a
    different residual stream.
  - **Training a Gemma-4 NLA from scratch needs ~16 H100s for the RL
    stage** per Anthropic's published recipe. That's well beyond a
    Kaggle T4 budget. The closest credible cloud cost is in the
    several-hundred-to-few-thousand-dollar range.
  - The 5th governance tool therefore runs with MockNLA on Gemma-4
    deployments today. Tool result `confidence_class` reports `mock`
    so consumers don't accidentally treat mock text as real
    interpretation.

### What this enables anyway

Even with MockNLA in place of a real Gemma-4 NLA, this work is
useful right now:

  1. **The five-tool contract is sealed.** When a Gemma-4 NLA becomes
     available (or we train one cloud-side), no notebook or gateway
     changes are needed — the same `get_explainer(real_model_id, ...)`
     just returns a real explainer instead of MockNLA.
  2. **The receipt schema is forward-compatible.** Mock and real
     explanations share the same JSON shape, so receipts produced
     today are comparable to receipts produced after we plug in a
     real NLA.
  3. **For Gemma-3-based work, the tool works for real.** If anyone
     deploys a Gemma-3 model under the HAIC governance loop, they
     can use the published Gemma-3-12B NLA and get real
     interpretability output today.

---

## Confabulation warning (from Anthropic's paper)

Anthropic's own disclosure: NLA explanations can "contain claims about
the target model's input context that are verifiably false." NLA is an
LM that LEARNED to describe activations — it has the failure modes of
language models, including confident fabrication.

The 5th governance tool reports the NLA's text directly. Consumers
(human reviewers and the agent in the function-calling loop) should
treat the explanation as one signal among several, not as ground truth.
A reasonable rule:

  - Use NLA explanations to FORM hypotheses about model behavior.
  - Use the other four governance tools (consent, provenance, geometry,
    receipts) to TEST those hypotheses.
  - When NLA and geometry agree (e.g. high FVE + low quantization
    hostility), confidence is high.
  - When they disagree (e.g. low FVE despite low hostility, or high
    hostility with confident explanation), DEFER to human review.

---

## Files

| File | Role |
|---|---|
| `prism_integration/nla.py` | NLA inference interface (MockNLA + PRISM adapter + factory) |
| `tools/audit_activation_explanation.py` | Tool 5 (NLA) — schema + executor |
| `tools/audit_provenance.py` | Tool 6 (MPK) — Cisco MPK wrapper + schema |
| `tests/test_nla_interface.py` | 15 tests covering the NLA interface |
| `tests/test_audit_activation_explanation.py` | 15 tests covering Tool 5 |
| `tests/test_audit_provenance.py` | 21 tests covering Tool 6 (all paths mocked) |
| `docs/nla_augmented_governance_2026-05-11.md` | (this file) |
| `notebook/_mpk_cell_insert.py` | Builder for the Scenario 6 cell |
| `D:/prism/src/prism/nla/` | PRISM-side NLA package (parallel session) |
| `D:/prism/docs/NLA.md` | PRISM-side design doc |

## Tool 6 (MPK) — short reference

  - **What it does:** statistical fingerprinting of model weights via five
    signals (EAS, END, NLF, LEP, WVC); composite identity score in [0, 1];
    answers "is candidate derived from reference?"
  - **Tiers (from Cisco's README):** >0.75 high-confidence match;
    0.65-0.75 weak match; ≤0.65 not matched; `pipeline_score==1.0` or
    `mfi_tier≤2` confirmed.
  - **Honest disclaimer (from MPK's README, surface verbatim):** "MPK
    provides strong statistical evidence of model derivation but is NOT
    cryptographic proof. It cannot distinguish 'trained from the same
    template' from 'copied weights'."
  - **Coverage:** MPK's reference dataset (`cisco-ai/model-provenance-kit`,
    908 MB, CC BY 4.0) does not publish a catalog of covered families.
    Recently-released models (including Gemma-4) may not yet be in it.
    The tool degrades to `model_not_in_database` and notes the fallback
    to PRISM geometry, rather than crashing.
  - **Notebook gating:** Scenario 6 in the submission notebook is behind
    `MPK_ENABLED = True`. Set to False to skip the 908 MB dataset
    download in environments without the disk budget.
  - **License attribution (required by Apache-2.0 + CC BY 4.0):** Cisco
    Systems, Inc. (2026), Model Provenance Kit,
    https://github.com/cisco-ai-defense/model-provenance-kit.

---

*"Follow the science." NLA is a real, published technique with real
limitations. Adopting it means adopting its constraints honestly: no
free Gemma-4 explanations until someone pays the training cost; mock
outputs marked as such; confidence thresholds that respect the AV's
known confabulation rate. The 5th tool is the interface; the real value
arrives when a Gemma-4 NLA does.*
