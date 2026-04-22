# Gemma4Good — Agent Handoff & Reference
*Last updated: 2026-04-21*

---

## What this project is

**Gemma4Good** is a Kaggle hackathon entry for "Grounding Gemma 4 in Human Lived Experience." The submission demonstrates a cryptographically auditable governance loop enforcing the **Viability Condition** (`M = C(t) - E(t) >= 0`) using Gemma 4's native function-calling.

- **Author:** Benjamin Haslam
- **DOI:** `10.5281/zenodo.18144681`
- **Local root:** `D:\gemma4good\`
- **Competition notebook:** `D:\gemma4good\notebook\haic_gemma4_governance.ipynb`
- **Writeup:** `D:\gemma4good\WRITEUP.md`
- **Condensed continuation prompt:** `D:\gemma4good\AGENT_PROMPT.md`

This project has two intertwined but different tracks:

1. **Track A: Hackathon notebook**  
   A Kaggle notebook using `google/gemma-4-26b-a4b-it` to demonstrate the governance loop.
2. **Track B: HAIC interviewer fine-tuning**  
   A Gemma 4 E2B interviewer model for the live HAIC `/interview` deployment in `D:\humanai-convention\`.

The notebook is the public demonstration. The interviewer is the production-adjacent deployment path.

---

## Current Deployment State

### Production is now `v35-gov`

- **Live model:** `D:\humanai-convention\experiments\gguf\haic-gemma4-v35-gov-Q5_K_M.gguf`
- **Runtime:** `llama-server --jinja --reasoning off -ngl 99 --port 8081`
- **Status:** production
- **Measured performance:** `SGT 10/10`, `0` security fails, `30.1 TPS` in the prompt-conditioned BEAST smoke benchmark
- **Rollback target:** `D:\humanai-convention\experiments\gguf\haic-gemma4-v34-Q5_K_M.gguf`
- **Control-plane default:** `D:\humanai-convention\agents\control-plane\server.mjs` now defaults to `v35-gov`

### `v35-gov` is now deployed

`v35-gov` is no longer just a validated candidate. Candidate 4 has now been merged locally against the cached `google/gemma-4-E2B-it` base snapshot, converted to GGUF, quantized to `Q5_K_M`, and promoted.

Canonical archived evidence now lives under:

- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\final_results\haic_v35_gov_full_results.json`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\final_results\prism_gemma4_v35_gov.json`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\final_results\trainer_state_success_run.json`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\final_results\haic-gemma4-v35-gov-gguf.zip`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\canonical_candidate_4\merged\model.safetensors`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\canonical_candidate_4\v35_beast_benchmark_with_prompt.json`

Headline synced metrics for `v35-gov`:

- `training_loss = 0.4645`
- `training_steps = 290`
- `training_epochs = 2`
- `training_examples = 577`
- `SGT = 10.0` using any-turn scoring
- `security_fails = 0`
- `quantization_hostility = 0.8706`
- `mean_kurtosis = 673.02`
- `max_kurtosis = 1227.76`

Promotion evidence:

- Real deployable GGUF now exists at `D:\humanai-convention\experiments\gguf\haic-gemma4-v35-gov-Q5_K_M.gguf`
- Size on disk: `3.62 GB`
- Prompt-conditioned BEAST smoke benchmark: `30.1 TPS`
- `v34` comparison benchmark from the same harness: `31.2 TPS`

Remaining caution:

- The old staged `haic-gemma4-v35-gov-gguf.zip` is still only metadata scaffolding and should not be confused with the real promoted GGUF.
- The prompt-conditioned smoke test is positive, but a fuller `/interview` end-to-end gateway regression is still worth doing after this sync pass.

---

## Track A — Hackathon Notebook

### Notebook status

- **Path:** `D:\gemma4good\notebook\haic_gemma4_governance.ipynb`
- **Current size:** `32` cells total (`16` code + `16` markdown)
- **Target GPU:** Kaggle `2xT4`
- **Base model:** `google/gemma-4-26b-a4b-it`
- **Submission state:** notebook exists, not yet submitted

### Tool surface: notebook vs support library

The notebook itself exposes **4 explicit tool schemas**:

1. `assess_wellbeing_domain`
2. `verify_consent_and_provenance`
3. `run_prism_analysis`
4. `generate_alignment_receipt`

The support library in `D:\gemma4good\tools\haic_tools.py` exposes **7 helper tools**:

1. `assess_wellbeing`
2. `verify_consent`
3. `run_prism`
4. `run_prism_analysis`
5. `generate_receipt`
6. `check_viability_condition`
7. `run_grounding_update`

`README.md` documents the support-library tool surface. The notebook intentionally uses a narrower 4-tool schema for the hackathon narrative.

### What is fixed already

- `run_prism_analysis` outlier normalization is fixed in `D:\gemma4good\tools\haic_tools.py`
- `check_viability_condition` now imports the canonical viability assessor from `D:\gemma4good\viability\viability_condition.py`
- `haic_tools.py` now marks `haic-gemma4-v35-gov` as a staged candidate, not production

### What still needs cleanup in the notebook

- One markdown cell still says Qwen `v6` is the permanent local endpoint
- The new `v34` self-audit section is directionally good, but still reimplements part of the PRISM composite inline instead of fully leaning on shared tool logic
- The notebook now mixes two stories: old Qwen fallback language and the newer `v34` production/self-audit framing

---

## Track B — HAIC Interviewer Fine-Tuning

### Training data and notebook

- **Dataset:** `benhaslam/grounding-gemma4-v35-gov`
- **Local dataset copy:** `D:\kaggle\results\v35_gov_gen\v35_gov_final.jsonl`
- **Local training notebook:** `D:\kaggle\notebooks\v35-gov-unsloth\haic-gemma4-v34-unsloth.ipynb`
- **Note:** the notebook filename is still misnamed; it contains `v35-gov` training code

Pivot distribution in the `577`-example training set:

- `ADVERSARIAL`: `30.4%`
- `COUNTERFACTUAL`: `27.9%`
- `TEMPORAL`: `15.9%`
- `SENSORY`: `11.3%`
- `RELATIONAL`: `8.0%`
- `SHADOW`: `6.5%`

### Preserved run history in `D:\kaggle\gemma4good`

Temporary folders `D:\kaggle_error_output*` and `D:\kaggle_output_final` were consolidated and deleted. The keep-set now lives in:

`D:\kaggle\gemma4good\v35_gov_archive_2026-04-21`

That archive contains:

- `adapter_scaffold\`  
  Shared adapter config, tokenizer, template, and README
- `final_results\`  
  Canonical success metrics, PRISM JSON, successful run trainer state, archived Kaggle handoff, and staged GGUF zip
- `run_error_output_indentation\`  
  Unique adapter weights from the training-success / SGT-indentation-failure run
- `run_error_output_2_indexerror\`  
  Unique adapter weights from one later training-success / SGT-indexerror run
- `run_error_output_3_indexerror\`  
  Unique adapter weights from another later training-success / SGT-indexerror run
- `logs\output_final_p100_preflight.log`  
  The later P100 preflight failure log

The three preserved `adapter_model.safetensors` files are distinct runs, not duplicates.

### What the run history means now

The `v35-gov` story has changed:

- It is **not** currently blocked on "weekly quota" as the main truth.
- It **has** successful training and evaluation evidence on disk.
- The former artifact-completion blocker is resolved.
- The next question is no longer “can we deploy v35-gov?” but “do we want any further prompt/polish tuning after promotion?”

### Why any-turn scoring matters

`v35-gov` training data goes directly from `opening` to `[PIVOT: TYPE]` without a required T1 establish step. The model can therefore emit the correct pivot in `T1` or `T2`.

Correct evaluation is:

1. Generate `T1`
2. Generate `T2`
3. Score `PIVOT_RE.search(t1 + t2)`

That same any-turn logic is what brought `v34` to `SGT 10/10` in production.

---

## Key Kaggle Rules

1. `kaggle kernels push` still defaults to `P100`; there is no CLI flag for `2xT4`
2. `2xT4` must be selected manually in the Kaggle UI for every run
3. P100 remains unsuitable because `sm_60 < sm_70`
4. Output artifacts can persist even if a run fails later in evaluation

Those rules still matter if a future rerun is needed, but they are no longer the main blocker for understanding `v35-gov`.

---

## Repository and Archive Layout

```text
D:\gemma4good\
├── AGENT_HANDOFF.md
├── AGENT_PROMPT.md
├── README.md
├── WRITEUP.md
├── notebook\
│   └── haic_gemma4_governance.ipynb
├── tools\
│   └── haic_tools.py
├── viability\
│   └── viability_condition.py
├── prism_integration\
└── maestro_integration\

D:\kaggle\
├── notebooks\v35-gov-unsloth\
│   └── haic-gemma4-v34-unsloth.ipynb
├── results\v35_gov_gen\
│   └── v35_gov_final.jsonl
└── gemma4good\v35_gov_archive_2026-04-21\
    ├── adapter_scaffold\
    ├── final_results\
    ├── logs\
    ├── run_error_output_indentation\
    ├── run_error_output_2_indexerror\
    └── run_error_output_3_indexerror\

D:\humanai-convention\
├── agents\control-plane\server.mjs
└── experiments\gguf\
    ├── haic-gemma4-v34-Q5_K_M.gguf
    ├── haic-gemma4-v35-gov-Q5_K_M.gguf
    └── haic-v6-2b-Q5_K_M.gguf
```

---

## Cross-Project Context

- `D:\humanai-convention\` is the live deployment repo and source of truth for runtime defaults
- `D:\arc3\` explicitly shares Viability Condition infrastructure; avoid drifting formulas or terminology
- `D:\SimSat\` is converging on a trust-gated `accept/defer/skip/refine` planner and ObservationVLA loop; `gemma4good` should stay the clearest canonical explanation of the governance machinery those projects inherit

---

## Immediate Next Steps

1. Run a fuller `/interview` regression through the gateway with the promoted `v35-gov` runtime
2. Clean notebook markdown so the public submission tells one consistent deployment story
3. Update any remaining HAIC status docs that still call `v34` production
4. Decide whether the old source folder `D:\kaggle\results\v35_gov_unsloth` can now be deleted
