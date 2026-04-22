# Agent Prompt — Gemma4Good continuation

Copy the text below the horizontal rule and paste it as your opening message to a new agent.

---

You are picking up the Gemma4Good project from a handoff. Read `D:\gemma4good\AGENT_HANDOFF.md` first. This prompt is only the quick-start version.

## What this repo is

Two linked tracks:

1. **Hackathon notebook**  
   `D:\gemma4good\notebook\haic_gemma4_governance.ipynb`  
   Public Kaggle submission narrative for the Viability Condition using `google/gemma-4-26b-a4b-it`.
2. **HAIC interviewer fine-tuning**  
   Gemma 4 E2B grounding interviewer for the live HAIC `/interview` deployment.

## Current live state

- **Production model is now `v35-gov`**
- **GGUF:** `D:\humanai-convention\experiments\gguf\haic-gemma4-v35-gov-Q5_K_M.gguf`
- **Runtime:** `llama-server --jinja --reasoning off -ngl 99 --port 8081`
- **Observed metrics:** `SGT 10/10`, `0` security fails, `30.1 TPS` in the prompt-conditioned BEAST smoke benchmark
- **Rollback:** `D:\humanai-convention\experiments\gguf\haic-gemma4-v34-Q5_K_M.gguf`
- `v35-gov` was promoted after local merge + GGUF conversion + BEAST smoke benchmarking on 2026-04-21

## `v35-gov` status

`v35-gov` is **deployed**.

Canonical archived evidence now lives here:

- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\final_results\haic_v35_gov_full_results.json`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\final_results\prism_gemma4_v35_gov.json`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\final_results\trainer_state_success_run.json`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\final_results\haic-gemma4-v35-gov-gguf.zip`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\canonical_candidate_4\merged\model.safetensors`
- `D:\kaggle\gemma4good\v35_gov_archive_2026-04-21\canonical_candidate_4\v35_beast_benchmark_with_prompt.json`

Headline metrics:

- `training_loss = 0.4645`
- `SGT = 10.0` with any-turn scoring
- `security_fails = 0`
- `training_examples = 577`
- `quantization_hostility = 0.8706`

Archive caveat:

- The old temporary folders `D:\kaggle_error_output*` and `D:\kaggle_output_final` were consolidated and deleted
- The new archive preserves three distinct `adapter_model.safetensors` checkpoints plus the logs and final JSONs
- The staged `gguf.zip` still contains only config/tokenizer/index structure, not the real deployable GGUF
- The real promoted artifact is `D:\humanai-convention\experiments\gguf\haic-gemma4-v35-gov-Q5_K_M.gguf`

## Notebook status

- **Notebook size:** `32` cells (`16` code + `16` markdown)
- **Notebook tool schemas:** `assess_wellbeing_domain`, `verify_consent_and_provenance`, `run_prism_analysis`, `generate_alignment_receipt`
- **Support library tools in `tools\haic_tools.py`:** `assess_wellbeing`, `verify_consent`, `run_prism`, `run_prism_analysis`, `generate_receipt`, `check_viability_condition`, `run_grounding_update`
- `run_prism_analysis` normalization fix is already landed
- `check_viability_condition` now imports the canonical viability assessor
- Remaining notebook cleanup: one stale markdown cell still says Qwen `v6` is the permanent local endpoint, and the `v34` self-audit section still duplicates part of the PRISM composite inline

## Key paths

```text
D:\gemma4good\
├── AGENT_HANDOFF.md
├── AGENT_PROMPT.md
├── README.md
├── WRITEUP.md
├── notebook\haic_gemma4_governance.ipynb
├── tools\haic_tools.py
└── viability\viability_condition.py

D:\kaggle\
├── notebooks\v35-gov-unsloth\haic-gemma4-v34-unsloth.ipynb
├── results\v35_gov_gen\v35_gov_final.jsonl
└── gemma4good\v35_gov_archive_2026-04-21\

D:\humanai-convention\
├── agents\control-plane\server.mjs
└── experiments\gguf\
    ├── haic-gemma4-v34-Q5_K_M.gguf
    ├── haic-gemma4-v35-gov-Q5_K_M.gguf
    └── haic-v6-2b-Q5_K_M.gguf
```

## Immediate next steps

1. Run a fuller `/interview` regression through the gateway with promoted `v35-gov`
2. Clean notebook markdown so the public story matches the actual deployment story
3. Update any remaining docs outside this repo that still call `v34` production
4. Decide whether `D:\kaggle\results\v35_gov_unsloth` can now be deleted

## Shared context

- `D:\humanai-convention\` is the deployment source of truth
- `D:\arc3\` shares Viability Condition infrastructure
- `D:\SimSat\` is evolving adjacent trust-gated planning ideas, so keep `gemma4good` as the clearest canonical governance explanation
