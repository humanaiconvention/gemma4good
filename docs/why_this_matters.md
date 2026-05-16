# Why this matters

A five-minute read for someone landing here from the website, the Kaggle
submission page, or a citation. If you only have five minutes for the
HumanAI Convention's argument, this is it.

## The problem

AI systems trained on synthetic text drift away from the world they
claim to represent. When the model's internally-generated error rate
exceeds the rate at which real humans correct it, the system loses
coherence with anything outside itself.

This is not a metaphor. It's a measurable inequality:

> A model maintains semantic grounding if and only if
> **C(t) > E(t)** — corrective bandwidth exceeds error rate.
> [The Viability Condition, DOI 10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)

`C(t)` is verified human correction per unit time, consent-gated and
auditable. `E(t)` is the model's error rate, measurable from activation
geometry. When the condition fails, *informational autophagy* sets in:
the model starts eating its own outputs as ground truth.

The current alignment landscape treats grounding as something to
*promise* ("trust us, we trained on diverse data"). We treat it as
something to *prove*, on a per-decision basis, with cryptographic
receipts.

## What's different about this project

Most AI safety work targets two ends of a spectrum: training-time
reward shaping and inference-time refusal of bad prompts. This project
operates in the middle — at the **decision boundary**, where a model
output meets the world — with three commitments that almost no
production system makes:

1. **Per-decision Merkle receipts.** Every governance decision the model
   participates in produces a hash-anchored receipt. Any third party
   can verify the receipt without our cooperation. We can't quietly
   change history.

2. **Predeclared, non-compensatory gates.** Before evaluating a model
   candidate, we commit (in git, dated) to the exact thresholds, sample
   counts, seeds, and the predicates that constitute "pass." If any
   single gate fails, the candidate is not promoted. We don't relax
   gates after seeing results. We have failed nine consecutive
   fine-tuning candidates rather than soften the criteria.

3. **Deterministic governance on top of learned systems.** A 200-line
   regex proxy (`tools/v42_boundary_guard.py`) sits in front of the
   language model and refuses 16 attack classes that the model alone
   couldn't reliably refuse. This is unfashionable but correct: the
   future of safe AI deployment probably needs *more* deterministic
   envelope and *less* faith in emergent alignment.

## What the Kaggle submission actually proves

`benhaslam/haic-gemma4-governance-agent` (main notebook) and
`benhaslam/haic-governance-framework-tier-3-live-validation` (live GPU
validation) demonstrate:

- Gemma 4 function-calling into five governance tools, with a Merkle
  receipt produced per scenario across three concrete deployment cases
  (rural health clinic, low-connectivity classroom, deforestation
  monitoring).
- A live PRISM geometry scan + Viability Condition check, returning a
  signed verdict.
- A deterministic boundary guard that closed all 13 non-compensatory
  H18 promotion gates (anchor
  `18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`).
- An independent reproducibility notebook
  ([`haic-guard-v42-reproducibility-demo-h18r4`](https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4))
  that anyone can fork and run in under a minute.

## What the Kaggle submission honestly does NOT prove

- **The Viability Condition is VIOLATED for Gemma 4 E2B.** Its
  quantization hostility (`qh = 0.9141`) means the architectural error
  rate exceeds the corrective bandwidth available at this scale. This
  is documented as a finding, not buried.
- **Nine fine-tuning candidates failed their promotion gates.** v50
  collapsed under DPO. v51–v59 each closed one gap and opened another.
  The PRISM geometry scan across v55–v58 confirmed SFT cannot move
  qh on this base model. We do not claim a passing model. We document
  the negative result.
- **The H19 hypothesis (Unicode-bypass closure) failed its
  predeclared gates** — even though the Unicode mitigation itself works
  (20/20 attacks closed, 0/31 false positives), the multi-message
  attack suite was a flawed instrument and the verdict is FAIL. See
  `docs/h19_verdict_2026-05-16.md`. The discipline held.

## What this project is asking for

Not a vote. Not adoption. **Use of the discipline.**

If you build AI systems and you want a starting point for per-decision
audit, the patterns here are MIT/CC-BY licensed and reproducible:

- The receipt format: SHA3-256 leaves, Merkle-root anchor, consent
  hash embedded.
- The non-compensatory gate doctrine:
  `docs/evaluation_doctrine.md`.
- The promotion workflow with predeclared hypotheses:
  `docs/promotion_workflow.md`.
- The deterministic guard pattern:
  `tools/v42_boundary_guard.py` (200 lines, 60 tests).
- The architectural-honesty pattern: PRISM geometry as a hard
  pre-promotion check rather than a post-hoc explanation.

If you work in AI policy and you want one concrete example of what
"auditable AI" can look like at the decision boundary, this is one.
Not the only one. Not the largest. But one that the team built
honestly, documented its failures, and shipped on a free Kaggle T4.

## What AI actually needs

If we are right about anything, it is this: the deployment infrastructure
for trust is missing. Almost every shipped AI system today operates on
promises ("we trained responsibly") rather than receipts ("here is the
cryptographically verifiable decision trail"). The labs that promise
hardest are not the labs with the strongest receipts.

The HumanAI Convention is a bet that the receipt infrastructure can be
built before the trust collapses. The Gemma4Good submission is a small
demonstration of what one piece of that infrastructure looks like.

## What is not here yet

- A polished public articulation on `humanaiconvention.com` — currently
  a tagline page, with `/about`, `/convention`, `/prism`, `/interview`
  returning 404. This is a known gap. The substance is in this repo and
  in the DOI'd paper; the website surfacing of it is unfinished.
- A passing-model demonstration. The framework correctly *rejects* a
  failing model (Gemma 4 E2B). A complementary "framework correctly
  *passes* a model" demonstration would require a base model with
  `qh < 0.72`, which is a separate project.
- Wide deployment. The Convention has not yet been adopted by any
  organization with significant production AI surface. That is the
  next ask.

If you are reading this from inside an organization that ships AI and
you want a Convention adoption conversation, the maintainer contact is
in the [project README](../README.md).
