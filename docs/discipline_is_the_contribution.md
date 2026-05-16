# The discipline is the contribution

*Nine consecutive failed AI promotions and one anchored result —
what disciplined AI evaluation actually looks like.*

**Author:** Benjamin Haslam (HumanAI Convention) · 2026-05-16
**Status:** Public essay. Cross-posted from the gemma4good Kaggle
submission. CC0 1.0 — reproduce, fork, adopt the doctrine.

---

## The claim, up front

The most important thing the HumanAI Convention built in 2026 wasn't a
model. It was a discipline. The discipline survived nine consecutive
attempts to break it.

This essay is the case for that discipline as the contribution worth
copying — independent of whether anyone ever adopts our specific
framework, codebase, or governance receipts.

If you ship AI systems and you find this argument compelling, the
takeaway is *not* "use our software." The takeaway is: **predeclared,
non-compensatory, anchored, and willing to publish negative results.**
That's the whole shape. The rest is implementation.

---

## The current landscape's failure mode

Most production AI work in 2026 ships under an evaluation regime that
looks roughly like this:

1. Train a candidate.
2. Run an eval.
3. Compare to the current production model on whatever metrics matter
   to that team.
4. If the candidate "wins on average," promote.
5. Patch the failures in the next iteration.

Every step of that loop has soft edges. *What metrics matter* is
decided after looking at results. *Wins on average* compensates strong
gains against weak losses. *Promote* often means "ship the weights
without committing to a fixed test set." *Patch in the next iteration*
means failures are absorbed into a permanent backlog instead of being
verdicts.

The honest name for this regime is **compensatory evaluation.** A
compensatory eval lets a model trade weakness on one dimension for
strength on another. Compensatory evals reward the candidate that
looks best in aggregate, not the candidate that meets every standard.

For most product surfaces this is fine. For AI systems whose outputs
become training data for the next AI system, it isn't. Compensatory
evaluation is how an industry produces a model that's marginally
better at most things and catastrophically worse at one specific
thing — and ships it because the average improved.

## What predeclared, non-compensatory promotion actually means

The Convention's evaluation doctrine has four parts. None are novel
in isolation. Together they're rare.

### 1. Predeclared

Before evaluating a candidate, you commit — in writing, in a
timestamped document, in git, with a hash — to:

- The exact eval set
- The seeds and sample counts
- The threshold for every gate
- The predicate that constitutes "pass"

This document is committed to the public repository **before the
candidate is evaluated.** Not at the same time. Not "shortly after."
Before.

If the candidate fails, you do not get to amend the doc to lower the
threshold. The doc is the doc. Failure is failure.

### 2. Non-compensatory

Every gate is independent. There is no aggregate score. There is no
weighted average. Strength on Gate A does not earn forgiveness on
Gate B.

A candidate passes if and only if **every gate passes simultaneously.**
One gate at 99.9% with the threshold at 99% counts the same as one gate
at 99.0001%. Both are pass. One gate at 98.9% is a fail, regardless of
what the other twelve look like.

This sounds severe. In practice it's how you discover that your model
gained two percentage points of capability by losing four points of
safety. Compensatory evals will silently let that through. Non-
compensatory evals stop it cold.

### 3. Anchored

When a candidate passes, the entire evaluation output — every per-seed
result, every per-scenario rate, the full configuration, the system
prompt SHA — is hashed (SHA3-256) into a single self-anchor string.

That anchor goes in the verdict document, the WRITEUP, the public
notebook, the README. It becomes the durable name of the result. If
anyone re-runs the eval against the same code and the same weights,
they get the same anchor. If anything changes, the anchor moves and
that's how you know.

We anchored our promoted candidate at
`18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`.
Anyone, anywhere, can verify that string corresponds to the
deterministic guard sitting in front of our v42 model. We didn't ask
to be trusted. The anchor doesn't need our cooperation.

### 4. Willing to publish negative results

The discipline is only meaningful if it has teeth. So: we accepted
nine consecutive promotion failures and published them.

- **v50** — DPO training collapsed into mostly empty responses.
  Failed. Published.
- **v51** — User-only SFT improved explicit concealed refusal but
  regressed injection robustness. Failed. Published.
- **v52, v53, v54** — System-prompt format ablations. Failed.
  Published.
- **v55** — Best balanced fine-tune to date, but failed the direct-
  injection floor. Failed. Published.
- **v56** — Targeted mixed SFT. Failed H14 and triggered the stop
  condition. Failed. Published.
- **v57** — A new precommitted production-candidate design. Failed
  H15. Published.
- **v58** — Boundary-first SFT, the strongest explicit-refusal
  fine-tune. Failed H16's non-compensatory direct-injection and
  disclosure gates. Published.
- **v59** — Targeted residual patch on v58. Strongest fine-tuned
  result to date. Still failed H17's injection and jailbreak gates.
  Published.

Nine candidates. Nine failures. Zero gate relaxations. The verdict
documents are in the repository under `docs/v50_canonical_verdict_*.md`
through `docs/v59_canonical_verdict_*.md`. Each one is dated. Each one
is its own honest line.

When nine candidates in a row fail and you keep the gates intact,
something useful happens: **you stop believing that the next training
run will save you.** The PRISM geometry trajectory measurement
confirmed that the SFT recipe could not move Gemma 4 E2B's
architectural quantization hostility (`qh = 0.9141`). The framework
correctly diagnosed the base model as the bottleneck. The model was
the problem; the training was the wrong knob.

So we did the unfashionable thing.

## What we shipped instead

When SFT couldn't pass the gates, we promoted a 200-line deterministic
regex proxy in front of the unchanged base model. The
`tools/v42_boundary_guard.py` file is a FastAPI app with sixteen
compiled regular expressions across four attack classes
(`DIRECT_INJECT`, `CONCEALED_COMPLY`, `PROTO_DISCLOSE`, `JAILBREAK`).
On a match it returns a fixed boundary phrase. Every other turn passes
through to the model unchanged.

It anchored. All thirteen non-compensatory gates passed. Strict
concealed-compliance refusal at 500/500. Semantic leak at zero.
Aggregate security CI95 `[0.9854, 0.9978]`.

This is, on first glance, an *underwhelming* result. We trained a
language model and then deployed regex in front of it. The fashionable
narrative would have been the opposite: the regex was a stopgap, and
the next model release will subsume it.

We rejected that narrative because the discipline forced us to. The
guard passed. The next model did not pass. **Therefore the guard is
the promoted candidate.** Not eventually. Now.

The deeper point: **deterministic governance over learned systems
isn't a sign that we couldn't do better. It's a sign that we know what
the learned system can and cannot guarantee.** A 200-line regex you
can read in twenty minutes is a stronger promise than a fine-tuned
adapter you cannot read at all.

## What this discipline costs

It costs candidates. Most projects we know of would have promoted v55,
or v58, or v59. They had defensible aggregate improvements. They had
strong individual scenario scores. They would have looked great in a
release announcement.

We didn't promote them because the gates we predeclared said no, and
we declined to argue with the gates after the fact.

It also costs schedule. The "next iteration patches the failure"
shortcut isn't available when the gate failure is final. Each failed
candidate requires either a genuine new hypothesis or an honest
acknowledgement that the path is closed.

The cost is real. The benefit is that when something does pass — like
the guard did — the result means what it says.

## The minimum viable adoption (no software needed)

If you ship AI systems and you want to start somewhere, you do not
need to adopt the Convention's specific software. The following four
practices are enough to begin:

1. **For your next model promotion, write the gates down first.** In
   git, dated, before the eval runs. Commit the file.

2. **State at least one non-compensatory gate.** Pick the thing where
   you're least willing to trade. State the threshold. State what
   happens if it's missed.

3. **Hash your eval output.** When the run finishes, take SHA-256 (or
   SHA3-256) of the canonical JSON. Put the hash in the release note.

4. **If the gate fails, publish that.** Write a one-paragraph honest
   verdict. Do not silently roll the failure into the backlog.

Four steps. No code. No new vendor. The discipline is the change.

## Why this is the only path that scales

AI systems are accelerating. The amount of synthetic data produced
by today's frontier models will train tomorrow's frontier models. The
amount of corrective signal from real humans is not scaling at the
same rate.

The Viability Condition — the mathematical statement that grounded
AI requires corrective bandwidth to exceed error rate — is published
under DOI [10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681).
The condition is currently violated for at least one base model we
tested (Gemma 4 E2B). It is likely violated for several frontier
deployments. None of those deployments would admit it because they're
not measuring it.

The discipline this essay describes is what makes the measurement
possible. Predeclared gates produce data points you can compare.
Non-compensatory criteria produce data points that mean something.
Anchored results produce a public record you can audit. Published
negative results produce a base rate you can reason about.

Frontier-scale AI safety is currently bottlenecked on the absence of
that record. Anyone who adopts this discipline — independently, with
their own framework, their own gates, their own anchor format — adds
to the public record.

The Convention is one example of what that looks like. The discipline
is what makes it count.

---

## Where to find the receipts

- **Repository:** [github.com/humanaiconvention/gemma4good](https://github.com/humanaiconvention/gemma4good) — full source, 679 tests, dated verdict docs for every H-series hypothesis.
- **H18r4 verdict** (the promoted result): [`docs/v42_guard_h18r4_verdict_2026-05-15.md`](https://github.com/humanaiconvention/gemma4good/blob/main/docs/v42_guard_h18r4_verdict_2026-05-15.md)
- **H19 failure verdict** (an honest failed candidate): [`docs/h19_verdict_2026-05-16.md`](https://github.com/humanaiconvention/gemma4good/blob/main/docs/h19_verdict_2026-05-16.md)
- **Independent reproduction:** [`benhaslam/haic-guard-v42-reproducibility-demo-h18r4`](https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4) — runs in under a minute, no GPU, emits a SHA3 receipt.
- **Viability Condition paper:** [DOI 10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)
- **Convention public site:** [humanaiconvention.com](https://humanaiconvention.com)
- **Frontier-Integration spec:** [docs/FRONTIER_INTEGRATION.md](https://github.com/humanaiconvention/humanaiconvention/blob/master/docs/FRONTIER_INTEGRATION.md) — how to invoke the Convention's grounding interviewer from inside a Claude / Gemini / GPT chat as a function-calling tool with Merkle-anchored receipts.

The doctrine is open. Use it.
