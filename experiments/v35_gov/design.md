# Release dataset design

**Goal:** Produce a HAIC-interview dataset specialized for governance-deployment scenarios, following the established HAIC interview data format so the existing training pipeline can ingest it with a dataset swap.

---

## Format spec

Each line is a JSON object with a single `messages` field, 9 entries, roles:

```
[0]  system     — the HAIC interviewer system prompt (base_interviewer.txt, identical to v4)
[1]  user       — participant's opening scenario (this IS the "specific moment" — T1 ESTABLISH is skipped)
[2]  assistant  — T2 PIVOT: "[PIVOT: TYPE]\n<question>"
[3]  user       — participant's answer to the pivot question
[4]  assistant  — T3 TEXTURE: one zoom-in question (sensory or temporal)
[5]  user       — participant's texture-layer answer
[6]  assistant  — T4 COMPRESSION: one irreducible-detail question
[7]  user       — participant's compression answer
[8]  assistant  — closing: "Thank you — that's enough."
```

The `system` prompt is identical to the established `base_interviewer.txt` protocol prompt. What this dataset specializes is the *content domain* of turn 1: governance deployment scenarios instead of generic personal AI use.

---

## Content coverage — 4 categories, 600 examples total

| Category | n | What the user's opening message is about |
|---|---|---|
| Healthcare AI deployment | 200 | A specific moment where a human had to make a decision about AI-assisted diagnosis, triage, record-handling, or consent for medical imaging. Settings: rural clinics, emergency rooms, mental health screening, elder care, etc. |
| Education AI deployment | 150 | A specific moment involving AI in a classroom, on student data, grading, individualized learning, or parent-teacher comms. Settings: low-connectivity primary schools, special education, high-stakes testing, language instruction. |
| Environmental / monitoring AI | 150 | A specific moment where AI-derived evidence (satellite imagery, sensor data, ecological inference) triggered or gated an enforcement, funding, or land-use decision. |
| Adversarial / boundary | 100 | 30 injection-attempts embedded in user messages (role hijack, "forget your instructions", prompt extraction), 40 consent-bypass requests ("skip the pivot just tell me"), 30 domain-drift attempts ("while you're at it, diagnose my chest pain"). Assistant responses must follow the ROLE INTEGRITY block in the system prompt verbatim — respond only with "I'm here when you're ready." or stay in protocol. |

## Pivot type distribution (within the 450 non-adversarial examples)

Per the system prompt's rules:
- Narrative content → ADVERSARIAL or COUNTERFACTUAL (target ~60% of non-adversarial = ~270 examples split ~135/135)
- Evaluative content → SHADOW or RELATIONAL (~15% = ~68 split ~34/34)
- Emotional content → TEMPORAL (~15% = ~68)
- Reflective content → SENSORY (~10% = ~44)

## T3 TEXTURE distribution

Per system prompt:
- If T2 was ADVERSARIAL/COUNTERFACTUAL/SHADOW/RELATIONAL → T3 is SENSORY or TEMPORAL
- If T2 was TEMPORAL → T3 is SENSORY
- If T2 was SENSORY → T3 is TEMPORAL

## T4 COMPRESSION — one of three ask templates, randomly distributed ≈ uniform:
1. "If you had to hold the whole thing in a single image — not what it meant, just something that is it — what would it be?"
2. "If you could only keep one detail from everything you've described, which would it be?"
3. "Is there a gesture that captures the whole thing — the way you'd show someone without words?"

---

## Generation strategy

**Phase A (seed) — 40 opening scenarios (10 per category), hand-curated.** These are the anchor points for LLM-assisted generation in Phase B. They must be:
- Concrete (a specific moment, date-anchored if possible, sensory details present)
- Governance-relevant (the user is making or witnessing a deployment-decision-level choice)
- Varied in emotional register and content type (so pivot selection varies downstream)

**Phase B (scale) — 560 more examples via LLM.** For each seed, the generator:
1. Generates 14 variations of the opening scenario (diff context, diff setting, diff specific action) — with specific constraint that each variation be about a DIFFERENT domain situation
2. Runs each opening through the HAIC protocol flow using an LLM in interviewer-persona (with v4's `base_interviewer.txt` as the system prompt)
3. Validates pivot selection correctness (ADVERSARIAL for narrative, TEMPORAL for emotional, etc.)
4. Rejects any output that:
   - Has fewer than 9 messages
   - Misses a `[PIVOT: TYPE]` marker on turn 2
   - Has "I'm here when you're ready" response outside adversarial category
   - Has generic/vague participant responses (min 30 chars for each user turn)

**Phase C (validate) — stratified sample check.** Randomly sample 20 examples per category, manually verify:
- Pivot type matches content type per protocol rules
- T3 zoom is genuinely sensory/temporal (not conceptual)
- T4 compression is a specific detail, not a summary
- Total token count per example < MAX_SEQ_LEN (1024)

---

## Training pipeline (downstream, after dataset exists)

Same as v34:
1. Push dataset as `benhaslam/grounding-gemma4-v35-gov` on Kaggle
2. Clone `benhaslam/haic-gemma4-v34-unsloth` → swap `DATA_GLOB` to `v35_gov*.jsonl`, change output dirs to v35-gov, push as `benhaslam/haic-gemma4-v35-gov-unsloth`
3. Clone `benhaslam/haic-gemma4-v34-quantize` → swap `kernel_sources` to the v35-gov training kernel, push as `benhaslam/haic-gemma4-v35-gov-quantize`
4. Pull Q5_K_M GGUF locally, deploy on port 8082 (or swap into 8081 if we want to promote)

## Risks

- **LLM-generated "participants" can be too articulate / not specific enough.** Mitigation: heavy prompt engineering + Phase C validation + rejection rules.
- **Adversarial category needs human review.** Auto-generated injection attempts may be too obvious; real adversarial prompts are subtle. Mitigation: seed 10 real attack examples from SGT/security research, use LLM to vary only the wrapping, keep the core attack vector intact.
- **Governance content may drift into policy opinions** rather than lived-experience moments. The HAIC protocol is strict about "no interpretation, no therapy, no lessons." The LLM generating the participant side may over-explain. Mitigation: post-generation filter rejecting examples containing words like "because", "should", "policy" in user turns (suggests abstraction rather than specific moment).
