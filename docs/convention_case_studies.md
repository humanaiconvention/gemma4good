# The Convention, applied — case studies

*Four working implementations of the HumanAI Convention's discipline
applied to four different domains. Each is a separate repository
and a separate evidence file, but the underlying primitive — anchored,
predeclared, non-compensatory governance with public failure verdicts —
is the same.*

**Status:** Living document, 2026-05-16. Maintained at the gemma4good
repo so the discipline essay can link here; intended for cross-post to
[humanaiconvention.com](https://humanaiconvention.com).

---

## Why this page exists

Each Convention project has been documented in its own repository, in
its own write-up, with its own anchors. A reader landing at any one
of them sees a project. A reader landing at this page sees a pattern.

The pattern is the contribution.

---

## 1. Gemma 4 Good — runtime governance on a frontier model

**Domain:** Cryptographically auditable governance applied to AI
function-calling.

**Where:** [github.com/humanaiconvention/gemma4good](https://github.com/humanaiconvention/gemma4good) ·
Kaggle submission to the Gemma 4 Good Hackathon (May 2026).

**What was anchored:**

- The H18r4 promoted candidate `guard + v42` passed all 13
  non-compensatory predeclared gates. Canonical anchor
  `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`.
- Nine consecutive fine-tuning candidates (v50–v59) failed the same
  gates without gate relaxation. Each failure has a dated verdict
  document in `docs/`.
- The Viability Condition was measured and found VIOLATED for Gemma 4
  E2B (Ceff/E = 0.879). The framework correctly diagnosed the base
  model rather than masking the result.
- Independent reproducibility: a public Kaggle kernel runs the
  H18r4 demo end-to-end in under a minute with a SHA3-anchored
  receipt.

**What this case proves about the Convention:** The discipline holds
under pressure. Faced with nine consecutive negative results across
six weeks of training, the gates did not move. The Convention
promoted a 200-line deterministic regex proxy instead of a
fine-tuned model when the regex passed and the model could not.

**Status:** Promoted, anchored, public.

---

## 2. SimSat — runtime governance in the air-gapped real world

**Domain:** Satellite encounter triage. Minutes between decisions,
no ground round-trip, uplinks too small for weight updates. The
Convention applied where compute and connectivity constraints rule
out everything except a predeclared, deterministic governance layer.

**Where:** [D:/SimSat](https://www.kaggle.com/code/benhaslam/simsat-gemma4-v1-training) ·
Submission to the DPhi Space × Liquid AI "AI in Space" challenge
(May 2026).

**What was anchored:**

- Tiers 1, 2, and 3 of the satellite encounter triage protocol
  passed predeclared evaluation gates.
- The promoted live runtime candidate is `simsat-gemma4-v11` — a
  Gemma-4-E2B + LoRA adapter — with a documented audit trail in
  `V11_AUDIT.md`.
- The TTT (test-time training) safety gates `error_bias` (BLOCK),
  `weight_drift` (warn), and `rate` (warn) are predeclared and
  non-compensatory: a single block-severity violation refuses the
  update.

**What this case proves about the Convention:** The discipline isn't
specific to grounded conversation. It applies wherever an AI system
makes a consequential decision under uncertainty. Satellite triage
is a different domain entirely from the gemma4good submission —
but the gate-shape is the same, and the discipline transfers.

**Status:** Submitted, anchored.

---

## 3. Parameter Golf — reasoning under verifiable constraint

**Domain:** ARC-3 reasoning challenges (Abstraction and Reasoning
Corpus) with an LLM-powered synthesis loop. The Convention applied
to the question: can a reasoning system produce verifiable solutions
under bounded compute?

**Where:** [D:/Parameter_Golf](https://github.com/humanaiconvention/parameter_golf)
(managed by the `haic-parameter-golf` agent).

**What was anchored:**

- The NeuroGolf 2026 ARC solver pipeline produces solutions whose
  intermediate steps are receipted at each stage.
- Solver iterations are evaluated against held-out ARC tasks with
  predeclared pass criteria; promotion of a new solver version
  requires no regression on the previously passing task set.

**What this case proves about the Convention:** Verifiability scales
down. A small reasoning system can produce per-step receipts the
same way a frontier chat assistant can. The Convention's
infrastructure isn't only for large models.

**Status:** Active, in development.

---

## 4. HumanAI Convention core — the protocol itself

**Domain:** Participatory grounding interviews with Merkle-anchored
participation receipts. The substrate that the other three projects
sit on.

**Where:** [github.com/humanaiconvention/humanaiconvention](https://github.com/humanaiconvention/humanaiconvention)
· [humanaiconvention.com](https://humanaiconvention.com).

**What is anchored:**

- The Maestro gateway issues participation receipts with SHA3-256
  Merkle roots over the session content, the consent flags, and
  the corrective signal.
- Five-layer consent gate (transcript, felt-state, training-signal,
  retention, withdrawal) — each layer independently grant-able.
- The five-tool Gemma 4 function-calling pipeline (wellbeing
  assessment, consent verification, PRISM activation geometry, NLA
  explanation, alignment receipt) is the runtime governance layer
  the gemma4good submission demonstrates.
- The Viability Condition paper (DOI [10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681))
  is the theoretical foundation: `C(t) > E(t)` — corrective
  bandwidth must exceed error rate.

**What this case proves about the Convention:** The protocol itself
is a working, public, open-source substrate. The other three
projects use it. Anyone else can use it the same way.

**Status:** Live, public, open-source.

---

## The pattern across all four

| Property | gemma4good | SimSat | Parameter Golf | HAIC core |
|---|---|---|---|---|
| Predeclared gates | ✓ | ✓ | ✓ | ✓ |
| Non-compensatory | ✓ | ✓ | ✓ | ✓ |
| Anchored eval output | SHA3-256 | Receipt-anchored | Step receipts | Merkle-anchored |
| Public failure verdicts | 9 published | TTT block events logged | Held-out regression checks | Withdrawal-attribution events |
| Independent reproducibility | Kaggle kernel | Local + Kaggle | Public ARC eval | Web-based interview |

The framework is the same. The domain isn't.

---

## What an adopter learns from this page

If you build AI systems, the Convention's discipline transfers to
your domain without modification:

1. Pick a candidate change (a new model, a new feature, a new
   policy).
2. Write the gates that constitute "pass" in a dated, committed
   document **before** evaluating.
3. State at least one non-compensatory gate.
4. Hash the evaluation output. Cite the hash everywhere.
5. If a gate fails, publish a one-paragraph verdict.

The four projects above are existence proofs that the doctrine
applies to language model governance, edge AI decisions, reasoning
systems, and the underlying participatory substrate alike. There is
no reason it would not also apply to a recommendation engine, a
medical-imaging classifier, a robot, or a credit-risk model.

The discipline travels. The receipts compound.

---

## How to cite this page

- For an academic paper: cite as the HumanAI Convention's case-study
  index, with the canonical URL above.
- For a regulator submission: reference alongside the
  [compliance one-pager](https://github.com/humanaiconvention/gemma4good/blob/main/docs/compliance_one_pager.md)
  for context on each domain's regulatory exposure.
- For a blog post or essay: link the
  [discipline essay](https://github.com/humanaiconvention/gemma4good/blob/main/docs/discipline_is_the_contribution.md)
  for the theoretical argument and this page for the four working
  examples.

This page is CC0 1.0. Mirror it, adapt it, extend it with your own
adoption story when you have one.
