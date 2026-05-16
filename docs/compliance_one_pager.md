# HumanAI Convention — Compliance One-Pager

*A single-page mapping of HAIC's runtime governance primitives to current and
near-future AI regulation. For enterprise legal teams, AI compliance officers,
and policy analysts evaluating per-decision audit infrastructure.*

**Status:** Draft, 2026-05-16. Not legal advice. Intended as a starting
point for a one-hour attorney review. The technical claims map to public
artifacts in the [gemma4good](https://github.com/humanaiconvention/gemma4good)
and [humanai-convention](https://github.com/humanaiconvention/humanaiconvention)
repositories.

---

## The thesis in two sentences

Most AI deployments today rely on **promise-based assurance** ("we trained
on diverse data, our refusal rate is X%") that cannot be independently
verified by a regulator, an auditor, or an end user. HAIC ships a working
example of **receipt-based assurance** — every AI decision produces a
cryptographically anchored receipt that any third party can verify without
the vendor's cooperation, and every model promotion passes predeclared,
non-compensatory criteria with public failure verdicts.

This is the primitive most current and forthcoming AI regulations
implicitly require but explicitly lack.

---

## What HAIC ships, mapped to regulatory categories

| HAIC primitive | What it does | Regulatory categories it satisfies |
|---|---|---|
| **Merkle-anchored per-decision receipt** | SHA3-256 hash over the decision, the consent flags, and the corrective signal. Independently verifiable. | Auditability · Traceability · Right of explanation |
| **Five-layer consent gate** (transcript, felt-state, training-signal, retention, withdrawal) | Granular, immutable, granted-not-presumed. | Data subject consent · Purpose limitation · Right of withdrawal-with-attribution |
| **Predeclared, non-compensatory promotion gates** | Thresholds committed to git before evaluation. No gate relaxation after results. | Risk management · Pre-market conformity assessment · Quality management system |
| **Public failure verdicts** | Every rejected candidate has a dated `docs/v<N>_canonical_verdict_*.md` documenting why it failed. | Post-market monitoring · Incident reporting · Transparency obligations |
| **Deterministic boundary guard** | 200-line FastAPI proxy with 16 compiled regex rules; auditable by reading the source. | Article 14 human oversight · Robust security ceiling on learned components |
| **PRISM activation-geometry diagnostic** | Architectural-level measurement of model's quantization hostility. Detects when fine-tuning has no further effect on a base model's structural limits. | Technical documentation · Model evaluation reporting |
| **Open framework + Apache-compatible weights** | All schemas, code, gateway, and verification are public. Adoption does not require trusting the Convention as a vendor. | Open-source / sovereign-AI procurement preferences |

---

## Concrete regulatory mappings

### EU AI Act (Regulation 2024/1689)

| Article | Requirement (paraphrased) | HAIC primitive that satisfies it |
|---|---|---|
| Art. 9 | Risk management system | Predeclared, non-compensatory promotion gates + public verdict record |
| Art. 10 | Data and data governance — including representativeness, purpose limitation, traceability | Five-layer consent gate + Merkle receipt anchoring contribution provenance |
| Art. 11 | Technical documentation | The dated verdict docs are technical documentation by construction |
| Art. 12 | Record-keeping (event logs) | Per-decision Merkle receipt is exactly an event log entry with cryptographic integrity |
| Art. 13 | Transparency and information to users | Receipt produced in-turn; user can verify what corrective signal informed the AI's response |
| Art. 14 | Human oversight | Deterministic boundary guard is an explicit human-readable oversight layer on top of the learned system |
| Art. 15 | Accuracy, robustness, cybersecurity | The H18r4 anchored evaluation with 13 non-compensatory gates demonstrates the form a robustness claim should take |
| Art. 60 | Real-world testing outside AI regulatory sandboxes | The reproducibility Kaggle kernel allows third-party real-world replication |
| Art. 72 | Post-market monitoring | Failed candidate verdicts are post-market monitoring artifacts by definition |

### US — NIST AI Risk Management Framework (AI RMF 1.0)

| Function | Sub-category | HAIC primitive |
|---|---|---|
| **MAP** | MAP-1: Context established, risks identified | Viability Condition paper + Convention `docs/why_this_matters.md` |
| **MEASURE** | MEASURE-2.4: Quantitative metrics tracked | Predeclared gates + Anchored canonical eval |
| **MANAGE** | MANAGE-2.4: Mitigation effectiveness verified | Public verdict record + 9 documented failures |
| **GOVERN** | GOVERN-1.6: Policies for risk tolerance | Non-compensatory promotion doctrine |

### US — State Attorney General activity (notably California, New York, Texas, Massachusetts)

State AGs have signaled in 2025–2026 that **deceptive practice doctrine**
applies to AI systems that make claims their training cannot substantiate.
The receipt-based assurance model is the strongest available defense:
the vendor's claims are reduced to "this anchored evaluation says X" and
the evaluation can be re-run.

### Forthcoming — OECD AI Principles updates (2026)

Public consultation drafts from the OECD AI Policy Observatory call for
**"per-decision auditability primitives suitable for cross-border AI
deployment."** The Merkle receipt format is content-addressed, vendor-
neutral, and exactly the shape that requirement implies.

---

## What a compliance team gets from adopting (selectively)

You do not have to adopt the whole Convention to get the compliance
benefit. Three minimum-viable adoptions, in order of effort:

1. **Lowest effort — 1 day:** Adopt the predeclared, non-compensatory
   promotion-gate doctrine for your next model release. Commit the gates
   to your internal repo before the evaluation runs. Produce a SHA-256
   anchor over the eval output. Cite both in your release note. This
   alone produces an Article 11 / Article 12 / NIST MEASURE-2.4
   compliant record at zero infrastructure cost.

2. **Medium effort — 1 week:** Stand up a local instance of the
   Convention's Maestro gateway. Wire receipt issuance to your
   inference path for one product surface (a customer-facing chat
   feature, an internal tooling decision, anything with audit pressure).
   Receipts now produced per decision, verifiable without Convention's
   cooperation.

3. **Full adoption — 1 quarter:** Integrate the grounding interviewer
   as a function-calling tool in your assistant product. Users opt in to
   anchored grounding sessions for high-stakes turns. Per-decision
   receipts cite the participant contribution by Merkle root. This is
   the [Frontier-Integration spec](https://github.com/humanaiconvention/humanaiconvention/blob/master/docs/FRONTIER_INTEGRATION.md)
   — full GTM and security contract documented.

---

## What this does NOT do

- It is **not** a substitute for an AI impact assessment, a data
  protection impact assessment, or a model card. It is a substrate
  those documents can cite.
- It does **not** legally indemnify a deployer for a deficient training
  pipeline. It produces audit artifacts that can be used in defense,
  prosecution, or regulatory review — neutral to the underlying
  decision quality.
- It does **not** prescribe what the gates should be. Each adopter
  chooses their own thresholds. The discipline is in declaring them
  before the eval runs and refusing to relax them after.

---

## Verification

Every claim in this document maps to a public, verifiable artifact:

- **Gemma 4 Good Kaggle submission:** [github.com/humanaiconvention/gemma4good](https://github.com/humanaiconvention/gemma4good)
- **H18r4 promoted-candidate verdict** (working example of an anchored, gate-passing promotion): [`v42_guard_h18r4_verdict_2026-05-15.md`](https://github.com/humanaiconvention/gemma4good/blob/main/docs/v42_guard_h18r4_verdict_2026-05-15.md)
- **H19 failed-candidate verdict** (working example of a published rejection): [`h19_verdict_2026-05-16.md`](https://github.com/humanaiconvention/gemma4good/blob/main/docs/h19_verdict_2026-05-16.md)
- **Reproducibility notebook** (1-minute independent verification): [Kaggle](https://www.kaggle.com/code/benhaslam/haic-guard-v42-reproducibility-demo-h18r4)
- **Viability Condition paper** (the theoretical foundation): [DOI 10.5281/zenodo.18144681](https://doi.org/10.5281/zenodo.18144681)
- **Frontier-Integration spec:** [FRONTIER_INTEGRATION.md](https://github.com/humanaiconvention/humanaiconvention/blob/master/docs/FRONTIER_INTEGRATION.md)
- **Discipline essay** (the public-facing argument): [`discipline_is_the_contribution.md`](https://github.com/humanaiconvention/gemma4good/blob/main/docs/discipline_is_the_contribution.md)

---

## Contact

For an exploratory conversation about Convention adoption in a
compliance-led context, the maintainers are reachable via the
[`humanaiconvention/humanaiconvention`](https://github.com/humanaiconvention/humanaiconvention)
repository. Particularly relevant if your organization is:

- Preparing an Article 9 risk management system for an EU AI Act
  high-risk classification
- Building post-market monitoring infrastructure for a regulated
  AI deployment
- Procuring AI services under sovereign-data or
  open-source-preferred clauses
- Drafting an AI impact assessment for a regulatory body
- Responding to a state-AG inquiry about an AI deployment

---

*This document is licensed CC0 1.0 (public domain dedication). Adapt
freely for your organization's compliance documentation. Attribution
appreciated but not required.*
