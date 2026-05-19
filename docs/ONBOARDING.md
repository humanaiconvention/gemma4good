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
| Submitted snapshot | `docs/submission_manifest_2026-05-18.md` (`ec7db2e`) |
| Research map | `docs/research_record_map.md` |
| Promoted live candidate | `tools/v42_boundary_guard_v7.py` + `D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf` |
| Canonical anchor | `4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8` |
| Verdict | `docs/h26_verdict_2026-05-17.md` |
| Limitations ledger | `docs/v42_guard_known_limitations_2026-05-15.md` (L-01 through L-09 closed/routed/out-of-scope as of H26) |
| Test surface | `python -m pytest tests/` → 797 passing |

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
   submitted `guard-v7 + v42` candidate preserves the original 16 English guard
   rules and adds Unicode normalization, per-message scanning, system-role
   rejection, leet-fold matching, and 11 multi-language direct-injection rules.
4. **Documents acknowledged limitations honestly.** H19 and H25 failed in
   public; H20, H21, H22, H24, and H26 closed the resulting documented gaps
   through separate predeclared hypotheses.

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
│   ├── v42_boundary_guard.py                   historical H18r4 guard
│   ├── v42_boundary_guard_v7.py                PROMOTED guard (H26)
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
│   ├── h19_*.jsonl / h26_*.py                  H-series test suites and runners
│   ├── v42_guard_v7_h26_canonical.json         H26 anchored eval result
│   ├── prism_geometry_trajectory.py            PRISM qh scan across v55–v58
│   └── archive/                                v43–v59 notebook builders, legacy evals
├── tests/                                      797 pytest tests
├── docs/                                       81 dated docs (hypotheses, verdicts, plans)
└── _local_*/                                   gitignored local-only state
```

---

## Run the test suite

```bash
python -m pytest tests/ -q
# Expected: 797 passed, 1 dependency deprecation warning
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

# Terminal 2: start the submitted guard-v7 endpoint
python -m tools.v42_boundary_guard_v7 --upstream http://127.0.0.1:8081 --port 8088

# Now point any OpenAI-compatible client at http://127.0.0.1:8088.
# Canonical attacks return a deterministic refusal; benign turns flow to v42.
```

## Reproduce the H26 anchor

```bash
python experiments/canonical_eval.py \
    --model-id haic-gemma4-v42-guard-v7 \
    --server-url http://127.0.0.1:8088 \
    --scenarios experiments/sgt_scenarios_v2.jsonl \
    --system-prompt-variant old \
    --seeds 7 13 23 42 100 \
    --n-samples 20 \
    --focused-n 100 \
    --out experiments/v42_guard_v7_repro.json \
    --failure-sidecar experiments/v42_guard_v7_repro_failures.jsonl
# Expected: anchor 4d0d7bf05ea2... within the H26 evaluation conditions.
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
   normalization, message iteration, system-role rejection, leet-folding, or
   multi-language matching in `tools/v42_boundary_guard_v7.py` invalidates the
   H26 anchor and requires a new H-series hypothesis.
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
1. `docs/submission_manifest_2026-05-18.md` (5 min — submitted snapshot)
2. `docs/research_record_map.md` (5 min — where the evidence lives)
3. `WRITEUP.md` (5 min — the public claim)
4. `docs/h26_verdict_2026-05-17.md` (10 min — current promoted candidate)
5. `docs/evaluation_doctrine.md` (5 min — the discipline)
6. This file (5 min — operational notes)

That's 30 minutes to be productive.
