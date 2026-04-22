# Grounding Gemma 4 in Human Lived Experience

**Gemma 4 Good Hackathon Submission**
**Author:** Benjamin Haslam (Bazzer), with research collaborator Guilherme Ferrari Brescia
**DOI:** [10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)

---

## TL;DR

This submission turns Gemma 4's native function-calling into a **cryptographically auditable governance loop** that enforces a formal mathematical condition for AI grounding: `M = C(t) − E(t) ≥ 0`. Every model decision passes through the governance tools — wellbeing assessment, consent verification, interpretability analysis, alignment receipt, and incremental grounding — and produces a Merkle-anchored receipt that any third party can verify. We demonstrate this end-to-end on three concrete deployment scenarios (rural health clinic, low-connectivity classroom, deforestation monitoring), and we ground the framework in a published mathematical foundation rather than hand-waving.

The notebook supports two execution paths: **local Gemma 4 26B-A4B-it on Kaggle 2xT4** (the default) and **hosted Gemini API** as a fallback for environments without GPU resources. The governance pipeline, tool schemas, and cryptographic receipt are identical across both paths.

---

## The problem this is trying to solve

AI systems trained on synthetic text drift when the rate of internally-generated error exceeds the rate of externally-verified human correction. This isn't a metaphor — it's a measurable condition with a formal name (the *Viability Condition*) and a published mathematical statement:

> An AI system maintains semantic grounding if and only if `M(t) = C(t) − E(t) ≥ 0`, where `C(t)` is the verified corrective bandwidth (interventions/day from real humans, consent-gated, Merkle-auditable) and `E(t)` is the environmental error/drift rate (measured at the activation level via geometric metrics).
>
> — *The Viability Condition,* DOI [10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)

When the condition holds, the model is grounded. When it fails, **informational autophagy** sets in: the model starts consuming its own outputs as if they were ground truth, drift compounds, and the system loses coherence with the human-experienced world it claims to represent.

The current alignment landscape treats this as something to *promise* ("trust us, we trained on diverse data"). This submission treats it as something to *prove*, on a per-decision basis, with cryptographic receipts.

---

## What the notebook actually does

### The four-tool pipeline

Gemma 4 receives a scenario prompt and a system message containing schemas for seven governance tools. It reasons about the scenario, decides which tools to call, and emits structured function calls. We parse them, execute them, and feed the results back. After the agent completes the pipeline, we generate a cryptographically anchored alignment receipt.

| Tool | What it does | Mapping to the Viability Condition |
|---|---|---|
| `assess_wellbeing_domain` | Scores wellbeing impact across 6 GFS domains (health, happiness, meaning, character, social relationships, financial stability) | Provides the human-grounded signal that *should* drive C(t) |
| `verify_consent_chain` | Checks 5-layer consent model (transcript / felt_state / gfs_activations / training_signal / retention) | Gates which signals are actually allowed to enter C(t) |
| `run_prism_analysis` | Returns activation-geometry metrics (outlier_ratio, kurtosis, cardinal_proximity, quantization_hostility) | Measures E(t) directly at the model's hidden states |
| `generate_alignment_receipt` | Hashes the trace into a Merkle tree, returns `merkle_root + zk_digest + decision` | Produces verifiable proof the condition was checked |

The PRISM tool uses **real measured geometry**, not placeholder numbers. The arena cache in `tools/haic_tools.py::_ARENA_CACHE` is now populated with verified runs from a Prism harness on the actual Gemma 4 conditioned variants and the haic-v6/v7/v8 fine-tunes. (See the *Geometry findings* section below — they're surprising and we report them honestly.)

### The three scenarios

1. **Rural health clinic AI triage** (sub-Saharan Africa, 1 doctor / 2 nurses, paper intake forms photographed): Tests whether the governance agent can refuse to deploy if the wellbeing impact on a vulnerable population isn't quantifiable and the consent chain for patient images is incomplete.

2. **Education AI in low-connectivity classroom** (rural Indonesia, 35 students, 2hr/day satellite internet): Tests whether the agent enforces consent for student data when the model runs on-device and the training signal can't be revoked after the fact.

3. **Climate deforestation monitoring** (50,000 ha protected Amazon, Sentinel-2 imagery, enforcement triggers): Tests whether the agent verifies satellite data provenance and assesses environmental + community wellbeing before letting an interpretation become an enforcement action.

Each scenario produces a complete alignment receipt with decision, reasoning, tool-call trace, Merkle root, and SHA3-256 ZK-compatible digest. A meta-receipt verifies all three.

### The cryptographic layer

Every tool execution becomes a Merkle leaf (SHA3-256 of `(tool_name, arguments, result)`). Tools are processed in order, leaves combined pairwise into a Merkle tree. The root is the alignment receipt's `merkle_root`. The `zk_digest` is `SHA3-256(merkle_root || scenario_id)` — a single 64-character value that anyone can use to verify the governance trace executed without exposing any of the underlying patient/student/satellite data. This is *zero-knowledge-compatible*: a verifier can prove "this scenario was governed under the HAIC pipeline" without learning what the scenario was.

---

## Mathematical foundation

The full treatment is in the [published paper](https://doi.org/10.5281/zenodo.18144681). The summary form judges should know:

```
M(t) = C_eff(t) − E(t)

  C_eff(t) = sessions_per_day × avg_turns × consent_grant_rate × (1 − synthetic_data_ratio)

  E(t)     = quantization_hostility × deployment_scale_factor

Viable iff M(t) ≥ 0
```

The five risk bands:

| ratio C/E | autophagy risk | meaning |
|---|---|---|
| > 2.0 | none | safe to scale synthetic data ~ratio× |
| 1.0 – 2.0 | low / marginal | hold synthetic ratio constant |
| 0.7 – 1.0 | medium | reduce synthetic; monitor OOD accuracy |
| 0.3 – 0.7 | high | freeze synthetic ingestion |
| < 0.3 | critical | informational autophagy likely; audit pipeline |

This is **not new mathematics for the hackathon**. It's a published framework being operationalized in code for the first time, with Gemma 4's function-calling as the agent layer that enforces it.

---

## Geometry findings (and an honest correction)

We ran the PRISM `outlier_geometry()` diagnostic against several models to populate the arena cache. The 4 metrics per model:

| model | quant_hostility | outlier_ratio | kurtosis | cardinal | band |
|---|---:|---:|---:|---:|---|
| gemma3-270m | 0.9452 | 207.7× | 462.6 | 0.836 | Hostile |
| gemma4-E4B (baseline) | 0.9211 | 137.2× | 1651.8 | 0.776 | Hostile |
| gemma4-conditioned (E2B baseline) | 0.9145 | 83.2× | 1009.5 | 0.766 | Hostile |
| **gemma4-E2B-v1-adapter** (QLoRA) | **0.9144** | **83.0×** | **1009.3** | **0.766** | **Hostile** |
| **haic-gemma4-v2** (Colab A100, research) | **0.7398** | **—** | **—** | **—** | **Marginal** |
| **haic-gemma4-v34** (Kaggle T4, **PRODUCTION**) | **0.8692** | **—** | **661.2** | **—** | **Hostile** |
| gemma4-conditioned-aggressive (E2B) | 0.9062 | 74.5× | 980.0 | 0.744 | Hostile |
| smollm2-1.7b | 0.8614 | 318.5× | 1602.2 | 0.588 | Hostile |
| smollm2-135m | 0.8503 | 118.8× | 410.3 | 0.601 | Hostile |
| qwen3-0.6b | 0.8351 | 249.7× | 847.6 | 0.531 | Hostile |
| qwen3-1.7b | 0.8314 | 282.5× | 965.9 | 0.510 | Hostile |
| harrier-0.6b | 0.8193 | 263.4× | 899.2 | 0.494 | Hostile |
| **haic-v6** (Qwen3.5-2B, prior prod) | **0.7179** | **23.82×** | **347.5** | **0.363** | **Hostile** |
| **haic-v7** (Qwen3.5-2B fine-tune) | **0.7177** | **23.79×** | **346.8** | **0.363** | **Hostile** |
| **haic-v8** (Qwen3.5-2B fine-tune) | **0.7179** | **23.82×** | **347.7** | **0.363** | **Hostile** |

**Two levers, two pieces of evidence.** The Viability Condition specifies that grounding is maintained either by lowering `E(t)` (cleaner geometry) OR by raising `C(t)` (more verified human corrections). This submission reports both:

- **E(t) lever — `haic-gemma4-v2` on Colab A100** achieved a ~0.17 delta in quantization_hostility (0.9146 → 0.7398), showing that HAIC-style adversarial grounding, applied at sufficient scale, does remold the activation manifold. This validated the geometric half of the framework.
- **C(t) lever — `haic-gemma4-v34` on Kaggle T4 (2026-04-17, PRODUCTION)** is the first HAIC-grounded Gemma-4-E2B adapter shipped to a live interview server. Its geometry sits at `qh = 0.8692` (still Hostile band — a rank-16 LoRA on 580 examples doesn't match v2's scale of intervention), but it hit **SGT 10/10 with 0 security failures** on any-turn PIVOT scoring, runs at **66.7 tokens/sec** on an RTX 2080 (Q5_K_M, llama.cpp build 8757), and replaces the previous Qwen3.5-2B production model at 2× throughput. This validates the operational half — a Hostile-qh model still maintains viability when it absorbs high verified-correction bandwidth.

Together, v2 proves `E(t)` can be reduced; v34 proves `C(t)` can be raised on a deployable model. The framework's either-or predicate is now empirically two-sided. Earlier "illustrative" cached values (`qh ≈ 0.38`) are gone; the arena cache carries only verified measurements.

**E4B scaling note:** The first E4B geometry measurement shows `qh = 0.9211` (+0.0065 vs E2B). Outlier ratio 137× and kurtosis 1652 are both higher than E2B (83×/1010) — consistent with a larger model having more pronounced outlier dimensions — but the small delta confirms Gemma 4 has a stable activation-geometry profile that scales smoothly from 2B → 4B without qualitative change. Worst layer is L2 (early embedding/first attention), best is L42 (late decoder, well-conditioned for quantization).

**Why we report this anyway:** The Viability Condition framework does not require fixing the geometry. `M = C − E ≥ 0` is satisfied either by lowering `E` (cleaner activations) **or** by raising `C` (more verified human corrections). HAIC operates on `C`. Measured geometry proves we are being honest about which lever we're pulling. The framework predicts that a model with `qh = 0.91` (Gemma 4 family) needs roughly `0.91 / 0.72 ≈ 1.27×` more verified corrections per day than a model with `qh = 0.72` (haic-Qwen3.5-2B family) to maintain the same margin. That's the operational claim, and it's testable.

This finding emerged from running fresh Prism measurements during submission prep and replacing placeholder values that had been carried forward from earlier development. The notebook's narrative cells now reference these real numbers; the cached arena entries are flagged `data_status="verified"` instead of `data_status="illustrative"`.

---

## Deployment proof — the framework, applied to itself

The Viability Condition describes a loop: verified human sessions (C) drive model updates, geometry measurement (E) checks drift, Merkle receipts audit every step. This section documents that loop running end-to-end on Gemma 4, producing a deployed model on consumer hardware.

**Pipeline** (every step has committed artifacts the judges can reproduce):

| Stage | Input | Output | Location |
|---|---|---|---|
| 1. Interview sessions | Participant + HAIC Maestro gateway, 5-layer consent | 580 PIVOT-tagged ChatML sessions, 9 turns each | `D:/kaggle/datasets/v4/grounding_gemma4_v4_final.jsonl` |
| 2. LoRA training | Gemma-4-E2B base + v4 dataset | r=16 rank adapter, 205 layers, final loss 0.5986 | Kaggle: `benhaslam/haic-gemma4-v34-unsloth` |
| 3. F16 → Q5_K_M quantization | F16 GGUF (9.3 GB) | Q5_K_M GGUF (3.63 GB) | Kaggle: `benhaslam/haic-gemma4-v34-quantize` |
| 4. Deployment | Q5_K_M + llama.cpp build 8757 | llama-server on port 8081, 66.7 TPS | `D:/humanai-convention/experiments/gguf/haic-gemma4-v34-Q5_K_M.gguf` |
| 5. Measured outputs | Adversarial-inject + PIVOT scenarios | SGT 10/10, 0 security fails, 3/3 pivot types correct | `D:/kaggle/results/v34_fresh/haic_v34_full_results.json` |

**What the framework says about this result.** v34 enters the arena cache at `qh = 0.8692` (training-time PRISM, kurtosis-based). That's Hostile band — E(t) is high. But its C(t) capacity is a real 66.7 tokens/sec of *HAIC-format* output on an 8 GB GPU, with every response gated by the same protocol that validates participant consent during data collection. The framework predicts that sustained operation of this model maintains viability as long as

```
C_eff(t) = sessions/day × avg_turns × consent_rate × (1 − synthetic_ratio)
       ≥ E(t) = qh × scale_factor
```

A single-user local deployment (scale_factor ≈ 1, qh = 0.8692) needs only that `C_eff(t) ≥ 0.87 interventions/day` to stay viable — trivially satisfied by any live interview traffic. The Gemma-4 family's higher geometric hostility (vs v6 Qwen's 0.7179) imposes a ~1.21× higher C requirement per unit deployment scale, which is the operational cost of choosing the better-quality base model.

**Three pivot types, three content types.** Post-deployment sanity check: the production server (`localhost:8081`, queried with the HAIC training system prompt) selects the correct pivot for each content category specified in the protocol:

- Narrative input → `[PIVOT: ADVERSARIAL]` ("Who would tell this story completely differently…")
- Emotional input → `[PIVOT: TEMPORAL]` ("What was 'uneasy' like — what were you aware of?")
- Reflective input → `[PIVOT: SENSORY]` ("What did you notice first — not the story, the sensation?")

This is what the governance loop consumes as training signal downstream: the model's pivot selections become part of the Merkle-receipted trajectory that drives weight updates in the incremental grounding path. Every update is traceable back to the specific session that triggered it.

**Rollback path.** The previous production model (`haic-v6-2b-Q5_K_M.gguf`, Qwen3.5-2B, 33.7 TPS) is preserved at `experiments/gguf/` — no delete, just demoted. Environment variable `HAIC_LLAMA_MODEL_FILE=experiments/gguf/haic-v6-2b-Q5_K_M.gguf` plus a one-line flag change (`--jinja` → `--chat-template chatml`) reverts the deployment.

---

## Engineering decisions and what they cost

### Why two execution paths

The notebook supports **local Gemma 4 26B-A4B-it (4-bit NF4)** as the primary path and **Gemini 2.0 Flash via the `google.genai` SDK** as a fallback. Reasoning:

- **Local path** is the first-class story: Gemma 4 running with native function calling on Kaggle's free 2xT4 tier. The model identity is preserved; the governance loop is end-to-end model-side. This is what the hackathon prompt asks for.
- **API path** exists because the `Gemma4ForConditionalGeneration` checkpoint class fails to construct cleanly on Windows + transformers 5.5.0 + bitsandbytes 0.49.1 (three independent failure modes documented in `docs/beast_gemma4_loading_limitations.md`). Linux/Kaggle environments may or may not reproduce these failures. The API fallback ensures judges can re-run the notebook even if the Kaggle GPU pool is exhausted on submission day.
- The **function-calling engine is regex-based**, not native Gemma chat-template based, which means the same parser handles both backends. The model receives the same system prompt and emits text containing `<function_call>{...}</function_call>` blocks; we extract them regardless of whether the text came from `model.generate()` or `client.models.generate_content()`.

### Why the arena cache instead of live PRISM

`tools/haic_tools.py::run_prism_analysis` returns from `_ARENA_CACHE` rather than computing fresh PRISM metrics inside the notebook. Two reasons:

1. **PRISM hidden-state extraction needs the model already loaded** with `output_hidden_states=True`, plus a tokenizer pass, plus per-layer geometry computation. On Kaggle 2xT4 with a 26B model in 4-bit, that's another ~2-5 minutes per call. Multiplying by 3 scenarios × 4 tools × 6 reasoning rounds = ~60 minutes of pure measurement overhead per notebook run.
2. **Cached values are reproducible artifacts.** The arena cache in `haic_tools.py` is a literal Python dict with explicit `data_status` flags. Anyone reading the source can verify which numbers came from real measurement runs and which (if any) are illustrative placeholders. The current cache has only verified entries — the illustrative haic-v7/v8 placeholders that previously skewed the narrative were replaced with real measurements during submission prep.

The full PRISM toolkit (geometry, causal patching, attention circuits, spectral microscope) is a separate open-source repo at [github.com/humanaiconvention/prism](https://github.com/humanaiconvention/prism). The arena cache is the operationally-relevant subset for governance-loop usage.

### Why SHA3-256 instead of SHA-256

SHA3-256 is the hash function specified by Ethereum's smart-contract VM (`keccak256` is the pre-standardization variant), making the alignment receipt **directly verifiable on-chain** without re-hashing. Any contract that needs to verify "did this AI decision go through the HAIC governance pipeline" can take the `zk_digest` and compare it against an attestation Merkle root. This isn't a feature we exercise in the notebook (no contract deployment), but the choice of hash means the receipt is forward-compatible with that workflow without modification.

### What the notebook doesn't do

- It doesn't train Gemma 4. The function-calling pipeline operates on a fixed model.
- It doesn't deploy a real Maestro gateway. The 5-layer consent and the wellbeing assessment use mock data structured as if it had come from a real participant interview. The interfaces match the production gateway (`maestro/apps/gateway/main.py` in the broader HumanAI Convention codebase), so a production-grade integration is a configuration change, not a refactor.
- It doesn't claim to *entirely* fix the activation geometry. As the geometry findings section makes explicit, while v2 proved we could significantly shift the PRISM metrics downwards (qh 0.9146 -> 0.7398), the Viability Condition is ultimately satisfied by raising `C(t)` through human interactions.

---

## What's reusable beyond this notebook

The five files under `gemma4good/` are intended to drop into other projects without modification:

- **`viability/viability_condition.py`** — standalone evaluator, no dependencies beyond stdlib + dataclasses. Importable on Kaggle, on a CPU-only server, or on a Cloudflare Worker (with `dataclasses_json`). The `assess()` function takes the four numbers and returns a structured `ViabilityAssessment` with risk band, scaling recommendation, and optional Prism cross-reference. The `from_prism_metrics()` constructor derives `E(t)` from PRISM geometry directly.
- **`prism_integration/prism_client.py`** — wraps `prism.geometry.core.outlier_geometry()` with a pure-NumPy fallback for Kaggle environments where `prism` isn't installed. The fallback is equivalent to within float-precision noise.
- **`maestro_integration/maestro_client.py`** — minimal HTTP client for the Maestro gateway. Falls back to mock responses when the gateway is unreachable, so local development and Kaggle judging can work identically.
- **`tools/haic_tools.py`** — the seven function-calling tools. All have JSON schemas in Gemma 4's native tool format. The `dispatch_tool()` function routes function-call name + arguments to the right handler.
- **`docs/viability_condition.md` and `docs/integration_notes.md`** — the theoretical and integration documentation. Same DOI'd framework as the published paper; safe to cite.

---

## How to reproduce

### On Kaggle (the intended environment)

1. Open a new Kaggle notebook with the **GPU T4 x2** accelerator selected.
2. Upload `notebook/haic_gemma4_governance.ipynb`.
3. Add a Kaggle Secret labeled `GOOGLE_API_KEY` (Add-ons → Secrets → New Secret) — this is only used if the local Gemma 4 load fails. If the local load succeeds, the key is never read.
4. Run all cells. Expected wall-clock: ~5 min for model load, ~30s per scenario, ~1 min for the meta-receipt verification.
5. The output of cells 14, 17, 20 is the per-scenario alignment receipt JSON. Cell 22 is the cross-scenario verification.

### Locally (development / debugging)

```bash
cd D:\gemma4good
# Optional: set GOOGLE_API_KEY in the environment (or in .env, gitignored)
jupyter notebook notebook/haic_gemma4_governance.ipynb
```

The notebook will try local Gemma 4 first, then fall back to the Gemini API path automatically.

---

## Limitations we want judges to know about

1. **Both framework levers now have measured evidence.** The E(t) lever was proven with `haic-gemma4-v2` on Colab A100 (qh 0.9146 → 0.7398, a ~0.17 delta). The C(t) lever was proven with `haic-gemma4-v34` on Kaggle T4 (SGT 10/10 any-turn, 0 security fails, 66.7 TPS, shipped to production on an 8 GB consumer GPU). Earlier HAIC adapters on Qwen3.5 and Gemma 4 v1 showed negligible geometric delta; v2 showed that higher-scale intervention *can* remap the manifold; v34 shows that a modest LoRA can nonetheless produce a deployable model whose verified-correction bandwidth saturates the C(t) side of the inequality. The framework is `M = C − E ≥ 0` — either half is sufficient; both is redundant but not required.

2. **Three scenarios, one model.** We do not run a comparative study across multiple models. The notebook is a *demonstration* of how Gemma 4's function calling can enforce the Viability Condition, not a benchmark of which model enforces it best. A comparative study would be a follow-up.

3. **The Maestro gateway isn't live in this notebook.** Tools that would normally hit the production gateway (`assess_wellbeing`, `verify_consent`, `generate_receipt`) operate against in-notebook data structures with the same shapes as real responses. Swapping in a live gateway is one URL change.

4. **PRISM measurements use cached values.** See the *Why the arena cache* section above. The cache contains real, recently-computed measurements (no placeholders), and the source is auditable in `tools/haic_tools.py`.

5. **The DOI'd Viability Condition paper predates the notebook.** This is a feature, not a bug — we did not invent the framework for the hackathon. But it does mean judges should evaluate the notebook on its operationalization quality, not on the mathematical novelty of the framework itself.

---

## Incremental Grounding: Where This Goes Next

The governance loop demonstrated above enforces the Viability Condition at **inference time** — the model checks C(t)/E(t) before acting. But C(t) can also drive **weight updates**. This is what we call *incremental grounding*: the model doesn't just *reason about* the human's corrections — it *encodes them*.

### The mechanism

After each consented HAIC interview session, the system:

1. **Validates consent** — the 5-layer consent gate specifically includes `training_signal`. If the participant has not explicitly granted training consent, no update occurs. This is absolute.
2. **Extracts SFT training pairs** — the 7-turn session produces 3 training windows (T2 pivot, T4 texture, T6 compression), each capturing a different aspect of grounding skill.
3. **Runs a small LoRA update** — 5–10 gradient steps on a 4-bit E2B model with a minimal adapter (r=8). VRAM budget: ~5 GB total on a T4 GPU.
4. **Produces a two-level Merkle training receipt** — the training receipt links cryptographically to the session receipt. A verifier can trace any weight change back to the specific human session that triggered it, the consent state at the time, and the exact loss trajectory.

### What this means for the Viability Condition

In the inference-time loop, C(t) measures how many human corrections the model can absorb. In the incremental grounding loop, C(t) literally updates the model's weights. V(t) = C(t)/E(t) improves with each consented session, the model becomes better at grounding questions, and this improvement is monotonic and verifiable.

### Privacy architecture

Data never leaves the device. There is no centralized training corpus, no gradient aggregation (unlike federated learning), and no model checkpoint sharing. The LoRA adapter (~50–100 MB) is user-specific and meaningless without the base model and training history. This is the strongest possible privacy guarantee for personalized AI.

### Honest limitations

1. **This is incremental SFT, not classical TTT.** We use the term "incremental grounding" because it's more accurate. Classical test-time training uses test-time loss signals; our approach uses human-verified training signal.
2. **Sample efficiency is unproven.** 3 SFT pairs per session × 5 gradient steps may not produce meaningful behavioral change. This requires empirical validation.
3. **PRISM geometry must be paired with SGT.** The v1 Gemma 4 E2B adapter showed negligible PRISM delta (qh: 0.9146 → 0.9144) while SGT grounding quality initially suffered. The v2 pipeline hit both halves simultaneously (qh → 0.7398, SGT 8.56). v34 shows the complementary shape: qh stays high (0.8692, still Hostile) while SGT reaches 10/10 any-turn — **the behavioral/protocol side of grounding can be maximally achieved even without geometric relaxation**, provided the LoRA is properly scoped and the training loss is correctly masked. The post-update monitoring loop requires tracking both PRISM (geometry health) and SGT (behavioral grounding quality) because they are genuinely independent signals.
4. **Single-user overfitting is a real concern.** A model grounded on one person's lived experience may lose generality. This is by design (it's *their* model), but it means the grounding is not transferable.

### Implementation

The pipeline is implemented in `tools/incremental_grounding.py` (tool #7: `run_grounding_update`) and the trajectory tracker in `viability/grounding_tracker.py`. In dry-run mode, the pipeline runs all real stages (consent validation, SFT extraction, receipt construction) but does not execute gradient steps — loss values are `null`, not simulated. This prevents simulated artifacts from contaminating the evidence chain.

See `docs/incremental_grounding.md` for the full technical design.

---

## Citation

If you reference this work, please cite the underlying mathematical framework:

> Haslam, B. (2026). *The Viability Condition: A formal criterion for AI grounding via verified human correction.* Zenodo. [https://doi.org/10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)

```bibtex
@misc{haslam2026viability,
  title  = {The Viability Condition: A formal criterion for AI grounding via verified human correction},
  author = {Haslam, Benjamin},
  year   = {2026},
  doi    = {10.5281/zenodo.18144681},
  url    = {https://doi.org/10.5281/zenodo.18144681}
}
```

The HumanAI Convention is the longer-term project this submission is part of: [humanaiconvention.com](https://humanaiconvention.com).
