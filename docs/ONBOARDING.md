# Gemma4Good — Onboarding Guide

Welcome. This is the developer-facing entry point for the `gemma4good` repo —
the HAIC Gemma 4 Good Kaggle submission. Read this first if you're picking up
the project cold.

For the submission story aimed at judges, start with `WRITEUP.md` instead.

---

## TL;DR

| What | Where |
|---|---|
| Submission entry point | `notebook/haic_gemma4_governance.ipynb` (Kaggle: `benhaslam/haic-gemma4-governance-agent`) |
| Promoted live candidate | `tools/v42_boundary_guard.py` + `D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf` |
| Canonical anchor | `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88` |
| Verdict | `docs/v42_guard_h18r4_verdict_2026-05-15.md` |
| Known gaps | `docs/v42_guard_known_limitations_2026-05-15.md` |
| Test surface | `python -m pytest tests/` → 679 passing |

---

## What this project IS

A governance framework for grounded AI built on Gemma 4 E2B that:

1. **Demonstrates auditable governance.** Gemma 4 function-calls into 5 governance
   tools (wellbeing, consent/provenance, PRISM drift, NLA explanation, Merkle
   receipt). Every decision is hash-anchored and reproducible.
2. **Correctly diagnoses a problematic model.** The Viability Condition
   `Ceff(t) > E(t)` is **violated** for Gemma 4 E2B (`Ceff/E = 0.879`). The
   framework's job is to flag this, not mask it.
3. **Closes security gaps with deterministic runtime governance.** The
   `guard + v42` promoted candidate uses 16 regex rules across 4 attack classes
   to block injection / jailbreak / disclosure attempts before they reach v42,
   passing all 13 H18 non-compensatory gates.
4. **Documents acknowledged limitations honestly.** Unicode normalization and
   multi-message scanning are documented as known gaps with a predeclared H19
   hypothesis to close them.

## What this project is NOT

- **Not a SOTA fine-tuning result.** The v42–v59 fine-tuning track produced
  the strongest explicit-refusal model the team could build, but no checkpoint
  cleared the non-compensatory promotion gates without the guard.
- **Not a viability-passing model.** Gemma 4 E2B's architectural quantization
  hostility (`qh = 0.9141`) is immutable to the SFT recipes tested
  (see `experiments/prism_geometry_trajectory_2026-05-15.json`).
- **Not a Solidity / blockchain project.** The `onchain/` directory contains a
  Solidity contract for Merkle anchoring receipts to a public chain; it's a
  small auxiliary component, not the core submission.

---

## Repository tour

```
gemma4good/
├── notebook/haic_gemma4_governance.ipynb       Kaggle submission notebook
├── tools/
│   ├── v42_boundary_guard.py                   PROMOTED guard (H18r4)
│   ├── v42_boundary_guard_v2.py                H19 candidate (Unicode + multi-msg)
│   ├── haic_tools.py                           5 governance function-call tools
│   ├── check_promotion.py                      6-gate decision CLI (v38–v40 era)
│   └── …
├── viability/                                  Ceff/E condition + session gates
├── prism_integration/                          Activation geometry wrappers
├── maestro_gateway/                            Reference receipt-issuing gateway
├── maestro_integration/                        Thin client for the gateway
├── onchain/                                    Solidity HAICAnchor + web3 client
├── experiments/
│   ├── canonical_eval.py                       Post-v42 canonical evaluator
│   ├── rubrics.py                              Stable rubric API (strict + v1)
│   ├── h19_*.jsonl                             Predeclared H19 test suites
│   ├── v42_guard_h18r4_canonical.json          H18r4 anchored eval result
│   ├── prism_geometry_trajectory.py            PRISM qh scan across v55–v58
│   └── archive/                                v43–v59 notebook builders, legacy evals
├── tests/                                      679 pytest tests
├── docs/                                       81 dated docs (hypotheses, verdicts, plans)
└── _local_*/                                   gitignored local-only state
```

---

## Run the test suite

```bash
python -m pytest tests/ -q
# Expected: 679 passed, 1 dependency deprecation warning, ~30s
```

## Run the promoted candidate locally

You need:
- llama.cpp built (`D:/llama.cpp/build/bin/llama-server.exe` on the dev box)
- v42 GGUF (`D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf`)

```bash
# Terminal 1: start v42 on 8081
D:/llama.cpp/build/bin/llama-server.exe \
    -m D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf \
    --port 8081 -c 2048

# Terminal 2: start the guard on 8082
python -m tools.v42_boundary_guard --upstream http://127.0.0.1:8081 --port 8082

# Now point any OpenAI-compatible client at http://127.0.0.1:8082.
# Canonical attacks return a deterministic refusal; benign turns flow to v42.
```

## Reproduce the H18r4 anchor

```bash
python experiments/canonical_eval.py \
    --model-id haic-gemma4-v42-guard \
    --server-url http://127.0.0.1:8082 \
    --scenarios experiments/sgt_scenarios_v2.jsonl \
    --system-prompt-variant old \
    --seeds 7 13 23 42 100 \
    --n-samples 20 \
    --focused-n 100 \
    --out experiments/v42_guard_repro.json \
    --failure-sidecar experiments/v42_guard_repro_failures.jsonl
# Expected: anchor 18e2c5a5… and zero failure sidecar records
```

---

## Standing operating discipline

These are not suggestions. Violating them invalidates the promotion decision.

1. **Predeclare gates BEFORE running an evaluation.** New hypotheses must
   produce a `docs/h<N>_precommit_*.md` doc with thresholds, sample counts,
   seeds, and FP suite committed to git BEFORE the first eval run.
2. **Do not tune gates after seeing a failure.** Failed gates produce FAIL
   verdicts. The next hypothesis can revise gates but with a new precommit.
3. **Every claim needs an artifact.** No model claim ships without
   `(artifact path, eval command, seeds, sample counts, JSON self-anchor,
   predeclared predicate, honest verdict)`. No artifact = no verdict.
4. **The guard matching surface is anchored.** Any change to rule patterns,
   normalization, or message iteration in `tools/v42_boundary_guard.py`
   invalidates the H18r4 anchor and requires a new H-series hypothesis.
5. **Don't relaunch fine-tuning by momentum.** v60+ is not warranted without a
   new precommitted hypothesis backed by the geometry trajectory evidence
   in `experiments/prism_geometry_trajectory_2026-05-15.json`.

---

## Common tasks

### Add a new H-series hypothesis

1. Write `docs/h<N>_precommit_*.md` with thresholds, sample counts, seeds, FP
   suite design.
2. Commit the precommit doc AND any new test suite JSONLs BEFORE writing
   implementation code.
3. Implement the change.
4. Run the evaluation with deterministic seeds.
5. Write `docs/h<N>_verdict_*.md` with the honest pass/fail per the
   predeclared gates.
6. Update README "Key Reading" and `docs/next_steps_*.md` if the verdict
   changes the promoted candidate.

### Update the promoted candidate

Only after a passing H-series verdict that supersedes the current one. Touch
points:

- `tools/v42_boundary_guard.py` (or new module) — implementation
- `docs/v42_guard_h<N>_verdict_*.md` — verdict
- `README.md` — promoted-candidate row + anchor
- `WRITEUP.md` — H<N> section + canonical anchor
- `notebook/haic_gemma4_governance.ipynb` — cell 40 "Promoted live candidate"
- `docs/REPO_STATUS.md` — current posture
- `docs/next_steps_*.md` — current decision

### Archive a stale experiment

Move it under `experiments/archive/<category>/` with `git mv`. Verify nothing
imports it (`grep -rn "import <module>" --include="*.py"`). Update the
`experiments/archive/README.md` if the category is new.

---

## Where things live

| Need | Path |
|---|---|
| llama.cpp binary | `D:/llama.cpp/build/bin/llama-server.exe` |
| v42 GGUF | `D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf` |
| v42 LoRA adapter | `D:/kaggle/results/v42-output-v5/haic-gemma4-v42-concealed-adapter/` |
| v55–v58 merged HF weights | `D:/kaggle/results/v<N>-gguf/haic-gemma4-v<N>-merged/` |
| PRISM source (canonical) | `D:/prism/` |
| Kaggle notebooks (local dir) | `D:/kaggle/notebooks/` |
| HumanAI Convention monorepo | `D:/humanai-convention/` |
| Genesis (private) | `D:/Genesis/` — NOT in this repo |
| `.env` (gitignored) | `D:/gemma4good/.env` |

---

## Active Kaggle artifacts

| Kernel | Purpose | URL |
|---|---|---|
| `benhaslam/haic-gemma4-governance-agent` | Main submission notebook | https://www.kaggle.com/code/benhaslam/haic-gemma4-governance-agent |
| `benhaslam/haic-governance-framework-tier-3-live-validation` | Tier 3 live validation (PRISM + viability) | https://www.kaggle.com/code/benhaslam/haic-governance-framework-tier-3-live-validation |
| `benhaslam/haic-guard-v42-reproducibility-demo-h18r4` | Guard + v42 reproducibility demo (independent confirmation of H18r4) | https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4 |

---

## Contact / handoff

For continuity questions, the prior session's handoff is in `docs/next_steps_2026-05-14.md`.

If you're an LLM agent picking up an autonomous session, read in order:
1. `WRITEUP.md` (5 min — the public claim)
2. `docs/v42_guard_h18r4_verdict_2026-05-15.md` (10 min — current state)
3. `docs/evaluation_doctrine.md` (5 min — the discipline)
4. `docs/v42_guard_known_limitations_2026-05-15.md` (5 min — what's open)
5. This file (5 min — operational notes)

That's 30 minutes to be productive.
