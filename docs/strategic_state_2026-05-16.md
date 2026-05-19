# HumanAI Convention — strategic state

*Where the project is on adoption, not on code. Where the next person
who picks this up — including the operator three months from now —
would need to start.*

**Status:** 2026-05-18 post-submission. Living document. Update when state changes,
not on a schedule.

---

## One-paragraph summary

The Convention's runtime and evaluation infrastructure is technically
complete, publicly anchored, and submitted to the Kaggle Gemma 4 Good
Hackathon. The submitted snapshot is commit `ec7db2e`, indexed in
`docs/submission_manifest_2026-05-18.md`. The promoted candidate is
`guard-v7 + v42` at H26 anchor
`4d0d7bf05ea2cc8d323b08982329455c72a999bd6da5a75a8b136a81b8ad8bb8`.
The remaining work is **adoption**: getting the Frontier-Integration spec,
the discipline essay, and the compliance one-pager in front of named
individuals at frontier labs, regulators, researchers, and journalists.
Adoption is currently bottlenecked on outreach motion, not on more features.

---

## What's in flight

### Active

- **Post-submission aftercare.**
  - Preserve the submitted snapshot boundary (`docs/submission_manifest_2026-05-18.md`).
  - Keep active reader-entry docs aligned with H26.
  - Leave dated verdicts historical rather than rewriting them.
  - Owner: repo operator + any assisting agent.

- **PR #110 on humanai-convention.** Interviewer helper banner.
  Open, build verified, waiting on operator review/merge.

- **Outreach drafts ready in `_local_state/outreach/`.** Targets list +
  four email templates (frontier lab, regulator, researcher,
  journalist). Personalize and send post-Kaggle-deadline.

### Predeclared but not executed

- **H-passing-1** — "framework correctly passes" demonstration on a
  base model with `qh ≤ 0.72`. Precommit and gates in
  `docs/passing_model_demo_plan.md`. Primary candidate Qwen 2.5 7B
  Instruct. One weekend of work.

### Decided but not implemented

- **Gateway deployment to Cloud Run.** Cloudflare tunnels are not a
  credible endpoint for frontier-lab integration. Plan in
  `docs/gateway_deploy_plan.md`. 7-day implementation path.
  Required before serious outreach traction.

---

## Where adoption stands

**The product is ready. The conversation hasn't started.** The
infrastructure side of the Convention has produced more public-record
work than most projects in this space ship. The adoption side is at
zero conversations.

### Audiences

- **Frontier labs.** Spec at `humanai-convention/docs/FRONTIER_INTEGRATION.md`.
  No outreach yet. Target list in `_local_state/outreach/targets.md`
  (Anthropic, DeepMind, OpenAI safety/product leads).
- **Regulators.** Compliance one-pager at
  `docs/compliance_one_pager.md`. No outreach yet. Targets: EU AI
  Office, NIST AISI, state AGs, OECD AI Policy Observatory.
- **Researchers.** Discipline essay at
  `docs/discipline_is_the_contribution.md`. No publication yet.
  Targets: Stanford HAI, Oxford / FHI successor, AI Snake Oil,
  LessWrong / AI Alignment Forum, Zvi Mowshowitz, Scott Alexander.
- **Press.** No outreach yet. Targets: Will Knight (WIRED), Kelsey
  Piper (Future Perfect), Karen Hao, Cade Metz (NYT).

### What's already on the public web

- [github.com/humanaiconvention/gemma4good](https://github.com/humanaiconvention/gemma4good) —
  full source, 797 tests, dated verdict docs, Apache 2.0.
- Three public Kaggle notebooks under `benhaslam` (governance agent,
  Tier 3 live validation, reproducibility demo).
- [humanaiconvention.com](https://humanaiconvention.com) — the public
  site. Updated 2026-05-16 with the guard+v42 substance,
  Frontier-Provider section, "what the convention rejected"
  section, submission artifacts in References, and a dropdown
  hover fix.
- [DOI 10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681) —
  Viability Condition paper.
- [YouTube video](https://youtu.be/p5ZprNkIAEM) — submission video.

### What's notably absent from the public web

- A doctrine essay published *anywhere with traffic* (Substack,
  LessWrong, HN). The essay exists in repo
  (`docs/discipline_is_the_contribution.md`) but has not been
  cross-posted to a venue where it gets read by people who don't
  already know about the project.
- A stable gateway endpoint backing the Frontier-Integration spec.
  (Cloudflare tunnel; rotates weekly.)
- A passing-model demonstration. (Plan ready; experiment not run.)

---

## Standing operating discipline (do not relax)

Listed here so the next person to operate the Convention knows the
invariants without reading the full evaluation_doctrine.md:

1. **Predeclare gates BEFORE running an evaluation.** Write the
   hypothesis with thresholds and predicates as a dated, committed
   file before the eval runs. No exceptions.

2. **Do not tune gates after seeing a failure.** Failed gates produce
   FAIL verdicts. The next hypothesis can revise gates, but only
   with a new precommit.

3. **Every model claim needs receipts.** No model claim ships without
   `(artifact path, eval command, seeds, sample counts, JSON
   self-anchor, predeclared predicate, honest verdict)`. No
   artifact = no verdict.

4. **The guard matching surface is anchored.** Any change to rule
   patterns, normalization, or message iteration in
   `tools/v42_boundary_guard.py` invalidates the H18r4 anchor and
   requires a new H-series hypothesis.

5. **Don't relaunch fine-tuning by momentum.** v60+ is not warranted
   without a new precommitted hypothesis backed by the geometry
   trajectory evidence in
   `experiments/prism_geometry_trajectory_2026-05-15.json`.

6. **Don't accept "the framework was too picky" as a critique without
   running H-passing-1 first.** The framework's calibration is a
   testable claim, not a vibe.

---

## What I (the operator) would say I'm doing

The above is what *the project's state* says. The operator should
add a personal layer below this line: what *the operator* is doing
this week, why, and what would change the answer.

> **(operator-owned section. The text below is a starter that the
> operator should overwrite. The structure is what matters; the
> wording should be operator-voice.)**

### This week

- Filing the Kaggle Gemma 4 Good submission.
- Reviewing PR #110.
- Producing the project video.

### Next two weeks

- Run H20 (clean Unicode-bypass closure).
- Run H-passing-1 (passing-model demo).
- Deploy the gateway to Cloud Run.
- Start outreach (one email per day from the templates).

### Next quarter

- The first frontier-lab conversation, if any of the outreach lands.
- The doctrine essay published to one external venue.
- Convention cited in at least one external paper or post.

### Hard constraints I'm not changing

- Operator runs this on personal time. Pace is set by personal
  bandwidth, not by external schedule.
- The Convention's discipline is non-negotiable. If a conversation
  partner wants the gates relaxed in exchange for adoption, the
  answer is no.

### What would change my mind

- A frontier-lab conversation moving from "interesting" to "let's
  pilot this" → reprioritize toward Cloud Run gateway + a
  hosted SLA tier.
- A regulator citing the compliance one-pager → write a longer
  formal version with attorney review.
- A researcher publishing a critique that lands → run a
  Convention-discipline response cycle (new H-series hypothesis,
  new precommit, anchored eval).

---

## Bus factor

If the operator is unavailable for a month, the work continues by
itself only on the public record side: the repos stay up, the Kaggle
notebooks remain reproducible, the DOI'd paper stays cited. The
adoption side stops entirely. No one else is sending outreach
emails, no one else is moving H20 to execution, no one else is
deploying the gateway.

The minimum effective handoff for adoption continuity is roughly:

1. **Outreach access:** the operator's email account, or a delegated
   sender with the templates and the targets list.
2. **The discipline:** a person who has read this file plus the
   `discipline_is_the_contribution.md` essay can maintain the
   operating principles without further onboarding.
3. **The receipts:** anything anchored stays valid without anyone's
   attention. No bus-factor risk to the existing evidence.

The lowest-risk single action to reduce bus factor: identify one
person (volunteer, intern, second contributor) who can read this
file, the discipline essay, and the FRONTIER_INTEGRATION spec and
send one outreach email on the operator's behalf with the operator
in CC. That single person, when added, doubles the outreach motion
without adding execution risk.

---

## What this document is NOT

- **Not a roadmap.** Roadmaps imply a schedule; the operator's
  bandwidth doesn't support one.
- **Not a strategy doc.** Strategy implies choices among
  alternatives; the choices have all been made and committed to
  in the dated decision docs.
- **Not a project plan.** It's a *state* file — the answer to
  "where is the Convention right now and what's the next move."

Update this file when state changes. Delete it when state changes
so much that the structure no longer fits. Replace it with
something more honest.
