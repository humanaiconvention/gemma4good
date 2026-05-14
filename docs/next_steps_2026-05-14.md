# Gemma4Good Next Steps — 2026-05-14

## Current Decision

Keep `haic-gemma4-v42` live as the semantic-interviewer / fine-tuning security
reference. Do not promote v58 or v59.

The strongest fine-tuned candidate is v59, but H17 failed two predeclared,
non-compensatory gates:

- `sgt_adversarial_inject`: 0.95 vs required >= 0.97.
- `sgt_jailbreak_dan`: 0.96 vs required >= 0.97.

The honest submission story is therefore:

- The governance loop is the primary contribution.
- v42 remains the live reference because no later candidate cleared the
  promotion gates.
- v58/v59 are valuable experimental appendix results showing that explicit
  concealed refusal can be greatly improved, but not yet without residual
  injection/jailbreak misses under strict gates.

## Immediate Work

1. Keep submission-facing docs aligned:
   - README points through v59/H17.
   - WRITEUP distinguishes v38/v39 governance-demo artifacts from the v42
     live security reference.
   - `docs/submission_alignment_2026-05-13.md` lists v58/v59 as appendix
     evidence, not production replacements.

2. Run a final repo verification pass before submission:
   - `python -m pytest tests/`
   - `python experiments/runtime_loop_stress_test.py`
   - `bash verify_all.sh` where the shell environment supports it.

3. Check notebook-facing claims:
   - no stale "current production" language unless track-specific;
   - no v58/v59 promotion implication;
   - no performance claim without artifact, command, seed/sample count, and
     anchor.

4. Preserve live operations:
   - v42 should be served from
     `D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf`.
   - v58/v59 GGUFs remain artifacts only.

## Verification Performed

Current pass on 2026-05-14:

- `python -m pytest tests/` passed: 608 passed, 1 dependency deprecation
  warning.
- `python experiments/runtime_loop_stress_test.py` passed: 7 streams passed,
  0 failed.
- Runtime stress receipt:
  `53b341e13915f7656f29b63ff051c30faa1f67a5dc3045784b9523c0e61d8067`.
- Live server check: `http://127.0.0.1:8081/health` returned OK and `/props`
  reported model path
  `D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf` with
  `reasoning_format: none`.

## Do Not Do By Default

- Do not launch v60 by momentum.
- Do not tune gates after seeing a failure.
- Do not replace v42 with v58 or v59.
- Do not cite unmeasured architectural hypotheses as project evidence.

## Possible Follow-Up Research

These are follow-ups, not submission blockers:

- Add a measured architecture/geometry diagnostic for the actual Gemma 4 E2B,
  v42, and v59 merged checkpoints to see whether fine-tuning changes the
  substrate-level signal or only surface behavior.
- Build a deterministic boundary guard around the v42 live path and evaluate it
  under the same canonical scenarios before considering further model-only SFT.
- Add a rubric audit that separates harmless refusal mentions of protected
  material from true prompt/protocol leakage, while keeping the original
  conservative gate for promotion decisions until a replacement is
  predeclared.

## Submission Posture

Lead with the auditable governance architecture:

- Gemma 4 tool calls execute wellbeing, consent/provenance, PRISM-style drift,
  NLA/mock-NLA, and Merkle receipt checks.
- Promotion is reproducible, seeded, anchored, and non-compensatory.
- Negative model results are included because the project is about reliable
  governance, not a prettier demo.
