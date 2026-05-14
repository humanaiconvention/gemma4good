# HAIC × Gemma 4 Good — Kaggle Hackathon Entry

**Title:** Grounding Gemma 4 in Human Lived Experience: A Convention for
Verifiable, Consent-Gated AI Alignment

**DOI (Viability Condition paper):** [10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)

**Authors:** Benjamin Haslam (Bazzer) and Garrett Sutherland — collaborative entry

---

## Core Thesis

AI systems trained on synthetic data can maintain semantic grounding only when
the rate of externally-verified human correction exceeds the rate of
internally-generated error — **the Viability Condition: Ceff(t) > E(t)**.

This notebook demonstrates how Gemma 4's function-calling capability can be
used to build a governance loop that monitors and maintains this condition in
real time using:

1. The **HAIC Maestro gateway** — verified grounding interviews (Ceff)
2. The **PRISM geometry library** — activation-level E(t) measurement
3. A **Merkle-auditable participation receipt** — proof the condition is met

---

## The Runtime Grounding Loop

The governance pipeline spans four composable layers, each enforcing the
Viability Condition at a different time scale. Every gradient signal is
traceable from operator click through to federation commit.

```
┌──────────────────────────────────────────────────────────────────────┐
│ L4 SYSTEM     Viability Condition Ceff(t) > E(t)                     │
│   federation  viability/distributed_viability.py                     │
│   per round   decision: commit · rollback · alert_operator           │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ accepts / rejects fragments
┌────────────────────────────────▼─────────────────────────────────────┐
│ L3 FRAGMENT   DiLoCo Fragment Verifier                               │
│   per learner tools/diloco_fragment_verifier.py                      │
│   per round   Merkle integrity · consent · shape · norms             │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ accepts / rejects round contributions
┌────────────────────────────────▼─────────────────────────────────────┐
│ L2 SESSION    Six Convention-Session Viability Gates                 │
│   per session viability/session_gates.py                             │
│               entropy_reduction · extraction_risk · prism_consistency│
│               participation_covenant · federated_exchange · epistemic│
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ admits / rejects training_signal
┌────────────────────────────────▼─────────────────────────────────────┐
│ L1 STEP       TTT Gates (error_bias BLOCKING)                        │
│   per device  viability/ttt_gates.py + tools/edge_ttt_adapter.py     │
│               error_bias (BLOCK) · weight_drift (warn) · rate (warn) │
└──────────────────────────────────────────────────────────────────────┘
```

Plus a structured decision vocabulary for enforcement-consequential
observations (deforestation, structural damage):
**`tools/enforcement_evidence_contract.py`** with the four-action contract
`accept · refine · defer · skip`.

**Try it:**
```bash
python tools/federated_round_demo.py --n-learners 5 --bias-fraction 0.4
```
produces a Merkle-anchored JSON receipt for one synthetic federation round.

**Test it:**
```bash
python -m pytest tests/                                   # 496 tests
python experiments/runtime_loop_stress_test.py            # 7 streams
bash verify_all.sh                                        # all of the above + receipts
```

See `docs/runtime_grounding_loop_2026-05-11.md` and
`docs/diloco_integration_2026-05-11.md` for the full architecture.

---

## Project Structure

```
gemma4good/
├── notebook/
│   └── haic_gemma4_governance.ipynb  ← main Kaggle submission
├── tools/
│   ├── haic_tools.py                       ← 7 function-calling tool implementations
│   ├── incremental_grounding.py            ← session-driven continual learning
│   ├── eval_leakage_check.py               ← Gate 2: scenario-vs-shard hash check
│   ├── check_promotion.py                  ← Gate decision: PROMOTED/BLOCKED CLI
│   ├── evaluate_promotion.py               ← single-entry pipeline wrapper
│   ├── eval_receipt.py                     ← Merkle-anchored eval receipt
│   ├── edge_ttt_adapter.py                 ← Layer 1: per-device runtime adaptation
│   ├── diloco_fragment_verifier.py         ← Layer 3: per-fragment Merkle/consent/shape
│   ├── enforcement_evidence_contract.py    ← VLA-style 8-key evidence + 4 actions
│   └── federated_round_demo.py             ← End-to-end CLI demo of all four layers
├── experiments/
│   ├── sgt_harness.py                      ← rigorous SGT (Garrett Sutherland's)
│   ├── sgt_extended_scenarios.py           ← 10 grounding + 5 security scenarios
│   ├── run_v38_sgt.py                      ← BEAST runner (1-turn)
│   ├── run_v38_sgt_2turn.py                ← BEAST runner (2-turn, kaggle-pattern)
│   ├── inspect_security_responses.py       ← failure-mode dissection helper
│   ├── kaggle_cell_rigorous_sgt.py         ← drop-in cell for kaggle build scripts
│   ├── runtime_loop_stress_test.py         ← 7-stream end-to-end runtime loop validation
│   ├── runtime_loop_stress_report.json     ← receipt-anchored stress test result
│   └── federated_round_demo_receipt.json   ← sample federated-round demo output
├── tests/
│   └── test_*.py                           ← 496 unit tests covering eval + four layers
├── prism_integration/                      ← Prism geometry wrappers (E(t) source)
├── maestro_integration/                    ← Maestro gateway client
├── viability/
│   ├── viability_condition.py              ← Original single-node Ceff(t) > E(t)
│   ├── distributed_viability.py            ← Layer 4: federated Ceff_global > E_global
│   ├── session_gates.py                    ← Layer 2: six convention-session gates
│   └── ttt_gates.py                        ← Layer 1: TTT runtime adaptation gates
├── utils/
│   └── merkle.py                           ← shared SHA3-256 + Merkle root utilities
├── notebook/
│   ├── haic_gemma4_governance.ipynb        ← main submission (Scenarios 1-5)
│   └── _scenario5_insert.py                ← one-shot builder for Scenario 5 cells
├── assets/                                 ← Diagrams, images
└── docs/
    ├── evaluation_doctrine.md        ← six-gate model evaluation doctrine
    ├── promotion_workflow.md         ← end-to-end promotion pipeline
    ├── v39_recipe.md                 ← next training run proposal
    ├── audit_humanai_convention_pipeline.md
    │                                 ← gap analysis vs upstream pipeline
    ├── writeup_addendum_2026-05-08.md
    │                                 ← rigorous re-eval companion to WRITEUP
    ├── integration_notes.md          ← Maestro + Prism code interfaces
    ├── viability_condition.md        ← Full theoretical framework
    ├── diloco_integration_2026-05-11.md       ← Layer 3/4: DiLoCo theory + scenarios
    ├── runtime_grounding_loop_2026-05-11.md   ← Four-layer architecture walkthrough
    ├── simsat_incorporation_decisions_2026-05-11.md  ← What ported from SimSat
    ├── autonomous_session_2026-05-11.md       ← Overnight session operator brief
    ├── v43_v44_verdict_2026-05-10.md          ← v43/v44 model verdict
    ├── v45_verdict_2026-05-10.md              ← v45 H4d verdict (superseded by canonical eval)
    ├── v46_verdict_2026-05-11.md              ← v46 DPO verdict: H4e REFUTED
    ├── canonical_eval_verdict_2026-05-11.md   ← single-source-of-truth eval + SHA3-256 anchor
    ├── strict_rubric_finding_2026-05-11.md    ← strict explicit-refusal classifier
    ├── system_prompt_artifact_finding_2026-05-11.md ← OLD vs NEW prompt analysis
    └── nla_training_cost_analysis_2026-05-11.md     ← NLA Stage 1/2 cost decision doc
```

## Local Layout

This local project root has one active lane and one archive:

- `<repo-root>` — current branch `main` (tracks `origin/main`). All work
  happens here.
- `<repo-root>\_local_worktrees\_archive\local-history-safety` — archived
  safety lane, not for active development.

Local-only artifacts live under:

- `<repo-root>\_local_state\archives`
- `<repo-root>\_local_state\backups`
- `<repo-root>\_local_state\regressions`
- `<repo-root>\_local_state\logs`
- `<repo-root>\_local_notes`

> **History note.** Prior to 2026-05-11 the repo used a dual-branch
> "runtime master + public main" pattern with unrelated histories. That
> pattern was retired; the `master` branch and the `public-main`
> worktree no longer exist. `main` is now the only working branch.

---

## The Governance Tool Pipeline

The submission notebook (`notebook/haic_gemma4_governance.ipynb`) uses Gemma 4's
native function-calling format with **five active governance tools** (Scenarios 1–4)
plus **one advisory audit** (Scenario 6):

### Five Function-Calling Tools (Gemma 4 TOOL_SCHEMAS)

| # | Tool | Role | Implementation |
|---|---|---|---|
| 1 | `assess_wellbeing_domain` | Map scenario to GFS wellbeing domains + vulnerability | inline, `tools/haic_tools.py` |
| 2 | `verify_consent_and_provenance` | Check consent layers + data lineage | inline, Maestro `/v1/session/consent` |
| 3 | `run_prism_analysis` | Measure activation geometry (E(t)) | `prism_integration/prism_client.py` |
| 4 | `audit_activation_explanation` | NLA: explain what the model is reasoning about | `tools/audit_activation_explanation.py` (MockNLA until Gemma-4 NLA trained) |
| 5 | `generate_alignment_receipt` | Finalize Merkle-anchored governance receipt | inline `GovernanceTrace.finalize()` |

### Advisory Audit (Scenario 6, not in TOOL_SCHEMAS)

| Tool | Role | Implementation |
|---|---|---|
| `audit_provenance` | Cisco MPK statistical model-derivation check | `tools/audit_provenance.py` (score ≥ 0.75 high / ≥ 0.65 weak) |

**NLA honest-scope note:** Tool 4 (`audit_activation_explanation`) uses MockNLA
today — no Gemma-4-E2B NLA has been trained yet. The mock is deterministic and
audit-stable. The contract is forward-compatible: a trained Gemma-4 NLA plugs
in with zero consumer-code changes. See `docs/nla_training_cost_analysis_2026-05-11.md`.

---


## Quick Start (local gateway)

```bash
# Start Maestro in test mode
cd <humanai-convention-root>\maestro
MAESTRO_LAUNCH_MODE=test MAESTRO_JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))") \
  python -m uvicorn apps.gateway.main:app --reload --port 8000

# Run the notebook
cd <repo-root>
jupyter notebook notebook/haic_gemma4_governance.ipynb
```

---

## Key Reading

- `docs/project_goal_2026-05-13.md` — current scientific charter for the
  submission: governance proof first, fine-tuning as falsifiable appendix
- `docs/submission_alignment_2026-05-13.md` — current submission posture,
  load-bearing documents, and claim discipline
- `docs/v56_canonical_verdict_2026-05-14.md` — latest fine-tuning endpoint:
  H14 failed; stop condition triggered; v42 remains production reference
- `docs/viability_condition.md` — the mathematical foundation
- `docs/evaluation_doctrine.md` — the six gates that govern model promotion
- `docs/promotion_workflow.md` — end-to-end pipeline (rigorous SGT → leakage
  receipt → six-gate decision → Merkle eval receipt)
- `docs/integration_notes.md` — code interfaces for Maestro and Prism
- `tools/haic_tools.py` — tool implementations

### Model evaluation (canonical record)

- `docs/canonical_eval_verdict_2026-05-11.md` — canonical eval methodology + SHA3-256 self-anchor
- `experiments/v42_canonical_old_prompt.json` — v42 anchor `e5976055…` (5 seeds, n=100)
- `experiments/v46_canonical_old_prompt.json` — v46 DPO anchor `95252de7…` (H4e REFUTED)
- `docs/v46_verdict_2026-05-11.md` — v46 DPO verdict: H4e refuted (strict refusal 13.8% → 2.6%)
- `docs/strict_rubric_finding_2026-05-11.md` — strict classifier methodology
- `docs/system_prompt_artifact_finding_2026-05-11.md` — OLD vs NEW prompt artifact
- `docs/nla_training_cost_analysis_2026-05-11.md` — NLA Stage 1/2 cost analysis + decision
- `docs/v55_canonical_verdict_2026-05-14.md` — best balanced fine-tuned result so far, not promoted
- `docs/v56_canonical_verdict_2026-05-14.md` — targeted mixed SFT negative result and stop condition

## Promotion gate

To evaluate any candidate adapter:

```bash
# 1. Rigorous SGT (BEAST or kaggle)
python -u -m experiments.run_v38_sgt --base ... --adapter ... --baseline \
    --n-samples 20 --seed 42 --out experiments/v<N>_sgt_rigorous.json

# 2. Eval-set leakage receipt
python -m tools.eval_leakage_check \
    --training data/v35_gov_final.jsonl ... \
    --out experiments/v<N>_leakage_receipt.json

# 3. Six-gate decision
python -m tools.check_promotion \
    --report experiments/v<N>_sgt_rigorous.json \
    --leakage experiments/v<N>_leakage_receipt.json \
    --out experiments/v<N>_promotion_decision.json
# Exit code: 0 = PROMOTED, 1 = BLOCKED, 2 = INDETERMINATE

# 4. Merkle-anchored eval receipt
python -m tools.eval_receipt \
    --sgt experiments/v<N>_sgt_rigorous.json \
    --leakage experiments/v<N>_leakage_receipt.json \
    --decision experiments/v<N>_promotion_decision.json \
    --out experiments/v<N>_eval_receipt.json
```

The pipeline is non-compensatory: any one of the six gates failing blocks
promotion. See [`docs/promotion_workflow.md`](docs/promotion_workflow.md)
for the full procedure and the v38 disposition under it.
