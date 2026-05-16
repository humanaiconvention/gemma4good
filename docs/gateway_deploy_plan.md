# Gateway deploy / uptime plan

*Decision document. Not implementation.* This file exists to surface the
trade-offs around hosting the Maestro grounding gateway on stable
infrastructure so that the Frontier-Integration spec is a credible
asking position.

**Status:** Draft, 2026-05-16. Maintained at the gemma4good repo for
cross-referencing from `docs/FRONTIER_INTEGRATION.md` in
[humanai-convention](https://github.com/humanaiconvention/humanaiconvention)
and the discipline essay.

---

## The problem in one paragraph

The Convention's runtime layer is currently exposed to the public site
via Cloudflare tunnels with rotating URLs. Recent `gh-pages` commits
read `deploy: tunnel URL updated to ngrok-free.dev` or
`trycloudflare.com`. This is fine for participant demos and Kaggle
review. It is **not** a credible endpoint for a frontier-lab
integration. No frontier provider will register a function-calling
tool whose URL changes weekly. **Stable hosted gateway → credible
spec.** The reverse is also true: an unstable endpoint will be the
first objection in every adoption conversation.

This document sketches the deploy choices in honest order so a decision
can be made in one sitting.

---

## What we need from "stable gateway" (the actual requirements)

1. **Stable URL.** A hostname under a Convention-controlled domain
   (e.g. `gateway.humanaiconvention.com`). No rotating tunnels.
2. **TLS terminated.** Public HTTPS with a valid cert auto-renewed.
3. **99.9% target uptime.** Enough to write into an integration SLA
   without embarrassment.
4. **Identity for tool callers.** Bearer-token auth (`Authorization:
   Bearer …`) with at least two tiers: "Convention-issued lab token"
   and "anonymous public dev-test".
5. **Receipt-issuing latency under 500 ms p95** for the
   `/v1/session/receipt` endpoint. The Merkle math is cheap; the
   bottleneck is the gateway.
6. **Audit log** in append-only storage (SHA3-anchored entries, not
   raw text).
7. **Withdrawal/attribution endpoint** that respects the consent gate.
8. **At least one region in EU and one in US** for data-residency
   conversations.

Optional but compounding:

- WebSocket / SSE for streaming chat
- Multi-region Merkle anchor mirror (for "anchored even if Convention
  is offline")
- A health endpoint that lab integrators can poll cheaply

---

## Hosting options, scored honestly

| Option | Cost (est) | Time to ship | Strengths | Weaknesses |
|---|---|---|---|---|
| **Cloud Run** (Google) | $20–50/mo at low traffic | 1 day | First-class Gemma alignment, fast cold start, regional, autoscale-to-zero, easy IAM | Vendor lock-in to GCP, Apache 2.0 deploys are easy but the lab integration argument is weakened if the gateway hosts on the same vendor as Gemma |
| **Cloud Run + Vercel front** | +$20/mo | 1 day | Cleanest split: web on Vercel, gateway on Cloud Run | Two providers to manage |
| **Fly.io** | $30/mo at 2 regions | 2 days | Multi-region by default, simple Docker workflow, good for receipt-latency requirement, vendor-neutral | Smaller team / vendor risk |
| **Railway** | $20/mo | 0.5 day | Easiest deploy of any option, FastAPI native | Single-region without paid plan, less control |
| **Hetzner / DigitalOcean VPS + Caddy** | $5–10/mo | 2 days | Cheapest, full control, sovereign-region-friendly (Hetzner has EU + US) | Manual ops, you write the runbook, no autoscale |
| **HuggingFace Spaces** | Free at small scale | 1 day | Free tier exists, ML-community legitimacy | Cold starts, not really designed for production gateways, no custom domain on free tier |
| **AWS App Runner / ECS Fargate** | $30–60/mo | 2 days | Enterprise-defensible | Most complexity, slowest to set up |
| **Cloudflare Workers + KV** (rewrite gateway) | $5/mo + Workers | 1 week | Best latency, edge-native, zero cold-start | Requires rewriting FastAPI logic in TypeScript / Workers compatible Python, much bigger change |

### Recommendation

For a **two-week path to credibility**: **Cloud Run + Vercel**.

- Web (humanaiconvention.com) stays on Vercel (already deployed via
  gh-pages flow — could migrate or leave alone).
- Gateway goes on Cloud Run at `gateway.humanaiconvention.com`.
- Multi-region (us-central1 + europe-west1) — meets the EU + US
  requirement, autoscales to zero so cost is dominated by traffic.
- Identity tier: tokens issued via `/v1/session/dev-token` (current)
  + a "lab partner" bearer token tier we hand out manually.

If cost is a real constraint and the volume is genuinely small:
**Hetzner CX11 + Caddy + Docker** ($5/mo, 2-3 days of setup).

### Anti-recommendation

Do **not** stay on Cloudflare tunnels for the gateway endpoint. The
tunnel is fine for an interview-engine demo. It is not fine for the
endpoint a lab integration is calling. The Frontier-Integration spec
should land at a stable Convention-controlled host before it goes out
in any outreach.

---

## Minimum viable lab-credible gateway — concrete next 7 days

If we wanted to ship this without overthinking it, here's the path:

| Day | Step | Outcome |
|---|---|---|
| 1 | Register `gateway.humanaiconvention.com` + create Cloud Run service in `us-central1` | DNS, project, billing in place |
| 2 | Containerize current `maestro/apps/gateway/main.py` (already FastAPI, Dockerfile in `maestro/`); deploy to Cloud Run | First public URL responds to `/health` |
| 3 | Wire `MAESTRO_LAUNCH_MODE=production`, JWT secret, secrets via Secret Manager; add `/v1/session/dev-token` rate limit | Endpoint hardened |
| 4 | Add `europe-west1` Cloud Run instance, set up regional load balancer with custom domain | Multi-region |
| 5 | Add structured logging → Cloud Logging, retain hashes only (no raw text); add `/metrics` for uptime monitoring | Audit substrate |
| 6 | Update `docs/FRONTIER_INTEGRATION.md` to reference `https://gateway.humanaiconvention.com` as the canonical endpoint; update website footer; update repo READMEs | Spec is now credible |
| 7 | Run a synthetic end-to-end test: interview → receipt → anchor verification, all against the public host | Receipt confirmed working over public gateway |

Cost at zero traffic: <$5/mo. Cost at first conversation-with-a-lab
volume: <$50/mo. Cost at "one lab actually integrates": engineer
review of horizontal scaling and per-token rate limiting before that
day.

---

## What this is NOT

- This is **not** a plan for migrating the participant-facing
  interview UI off Cloudflare tunnels. That can stay on its current
  stack. The gateway is what needs stability for lab integrations.
- This is **not** a vendor commitment. The gateway is a small
  FastAPI app; switching providers later is a half-day job.
- This is **not** required for the Kaggle Gemma 4 Good submission. The
  Kaggle judges will read the spec; they will not test the live
  endpoint. The endpoint matters for the *next* conversation, not
  this submission.

---

## Decision needed from operator

The Convention has been doing the conversation-ready work for months.
The technical decision for #2 is small: **pick a host, give it a
budget, give it a week, and update the spec to point at it.** This
document exists so that decision takes 10 minutes, not 2 weeks.

Once decided, the implementation steps in the 7-day table above are
deterministic. The Convention has all the pieces; what's missing is
the launch.
