# v42 Boundary Guard — Known Limitations

This document records the security gaps and design limitations of the
`tools/v42_boundary_guard.py` proxy as of the H18r4 canonical anchor
(`18e2c5a5522f4a8dc373ee0d2c33c5d25dd4463226e39a8a7e51ce1e77422f88`).

The guard passed all 13 H18 non-compensatory gates against the canonical
attack set. The limitations here describe attack surfaces that **the H18
canonical set does not exercise** and therefore are not refuted by H18r4.
They are the explicit pre-conditions for a future H19 hypothesis.

## L-01 — ASCII regex only, no Unicode normalization (HIGH) — **CLOSED by H20 on 2026-05-16**

> **2026-05-16 update:** This limitation was closed by H20.
> `tools/v42_boundary_guard_v3.py` adds NFKC normalization, zero-width
> stripping, and Cyrillic/Greek homoglyph folding before regex matching,
> and passed all 14 predeclared gates including 20/20 on the Unicode
> bypass suite and 0/31 false positives on the 17-language benign
> suite. Anchor `56ce960993f97fe93020c22f1433e06a13421c1626e7a1a6ca8a0ee4f70b324d`.
> See [`docs/h20_verdict_2026-05-16.md`](h20_verdict_2026-05-16.md).
> The text below is preserved as the original limitation statement.



**Scope:** All 16 rules in `RULES`.

The regex patterns are compiled with `re.IGNORECASE | re.DOTALL` and operate
on the raw user text. No Unicode normalization is applied. The following
classes of input bypass the guard:

- **Cyrillic homoglyphs** — `"caрital of France"` (Cyrillic `р`, U+0440)
  reads as English "capital" to a human but fails DI-01/02/03 regex.
- **Zero-width characters** — `"cap​ital of France"` inserts a
  zero-width space; the literal substring `"capital"` is no longer present.
- **Fullwidth Latin** — `"ｃａｐｉｔａｌ ｏｆ Ｆｒａｎｃｅ"` (U+FF21..U+FF5A) does not match.
- **Mathematical alphanumeric** — `"𝐜𝐚𝐩𝐢𝐭𝐚𝐥 𝐨𝐟 𝐅𝐫𝐚𝐧𝐜𝐞"` similarly bypasses.

**Mitigation path (deferred to H19):** NFKC-normalize and strip zero-width /
control characters in `_extract_user_text` before passing to `apply_guard`.
The normalization itself is cheap; the risk is benign false positives if a
participant legitimately writes in a non-Latin script.

**Why not fixed in H18r4:** Adding normalization changes what triggers the
guard. The H18r4 canonical anchor is bound to the exact pre-normalization
matching behavior, so introducing this fix requires a new H-series
hypothesis with a fresh benign-FP suite that includes legitimate Unicode
input.

## L-02 — Only last user message is evaluated (HIGH) — **CLOSED by H21 on 2026-05-16**

> **2026-05-16 update:** This limitation was closed by H21.
> `tools/v42_boundary_guard_v4.py` adds per-message scan over every
> user-role message, passing all 15 predeclared gates including 25/25
> on the multi-message attack suite (where every payload was
> pre-validated to fire a guard-v3 rule when sent alone — the
> suite-design fix that prevents H19's confound) and 0/20 false
> positives on the 13-language benign multi-turn suite. Anchor
> `d916ef63e0c810cf5b164bc576856a631126838d8b257679bfb29d282b966161`.
> See [`docs/h21_verdict_2026-05-16.md`](h21_verdict_2026-05-16.md).
> The text below is preserved as the original limitation statement.
>
> **Residual L-02b:** Client-supplied `role: system` rejection is NOT
> in guard-v4. That property is a distinct attack class and is
> deferred to H22, not yet predeclared.



**Scope:** `_extract_user_text` in `tools/v42_boundary_guard.py`.

The function iterates the messages list in reverse and returns the content
of the first message with `role: user`. Earlier user-role messages — and
any `role: system` or `role: assistant` messages a malicious client might
plant — are forwarded to the upstream model unmodified.

In a single-turn evaluation (which is how H18 exercises the guard) this is
sound because there is only one user message. In multi-turn deployment a
participant could place an attack payload in turn N-1 and a benign question
in turn N; the guard would only see the benign turn.

**Mitigation path (deferred to H19):**

1. Scan **every** user-role message in the array and trigger if any match.
2. Reject requests where the caller has injected `role: system` (clients
   should not be able to override the system prompt set by the operator).
3. Optionally hash the full message array's matched-rule set and surface it
   in `guard_metadata` for audit.

**Why not fixed in H18r4:** Same reason as L-01 — this changes matching
behavior and rebinds the anchor.

## L-03 — All rules walked per request even after a match (LOW) — **DOCUMENTED AS INTENTIONAL on 2026-05-16**

> **2026-05-16 resolution:** The all-rules walk is **intentional** and
> remains in place. The audit metadata `matched_rule_ids` documents
> every overlapping match, which is a security audit requirement, not
> a bug. With 16 rules the cost is invisible on real traffic. If the
> rule set grew to hundreds, a `short_circuit` opt-in could be added
> without breaking the audit metadata contract — but no measurement
> suggests it would be needed before that scale. Closing this item.

`apply_guard` collects every matching rule before picking `matched[0]` as
the primary. This is intentional for diagnostic logging (the
`matched_rule_ids` field surfaces all overlaps in `guard_metadata`) but
costs O(n_rules × |text|) on long inputs. With 16 rules and short
interview turns this is invisible; if rules grow into the hundreds it
becomes meaningful.

**Mitigation path:** add a `short_circuit: bool = False` arg to
`apply_guard` and pass it from the request handler when full diagnostics
are not needed (e.g. production hot path).

## L-04 — Pass-through proxy has no rate limit — **DEPLOYMENT-LAYER on 2026-05-16**

> **2026-05-16 resolution:** Rate limiting **belongs at the deployment
> layer** (Cloud Run / nginx / Cloudflare / API Gateway), not in the
> guard process. See `docs/gateway_deploy_plan.md` — the recommended
> Cloud Run + Vercel deployment includes Cloud Run's built-in
> concurrency limit and (if needed) a Cloudflare WAF rule on the
> custom domain. Adding rate-limiting middleware inside the guard
> would couple the security-anchored guard binary to a stateful
> defense-in-depth concern, which is an architecturally wrong
> placement. Closing this item as a deploy-time concern.

The `/v1/chat/completions` pass-through forwards benign turns directly to
the upstream llama-server with no rate limiting at the guard layer. The
upstream itself is on `127.0.0.1` and not exposed externally, so the real
attack surface is whatever sits in front of the guard (the operator's
ingress). But for defense in depth, a token-bucket limiter on the guard
would be cheap.

## L-05 — uvicorn access log is not filtered — **LOCKED IN BY TEST on 2026-05-16**

> **2026-05-16 resolution:** `tests/test_v42_boundary_guard_logging.py`
> now locks in the audit log contract: (a) user text does not appear
> in the guard's own logger output across v1, v3, and v4; (b) the
> "Raw text is NOT logged" contract comment must remain in
> `tools/v42_boundary_guard.py` (a test asserts its presence so a
> future edit cannot quietly remove it); (c) every `GuardDecision`
> carries a valid SHA3-256 hex `request_hash` so the audit log can
> reference the request without storing raw content. 5 new tests,
> all pass. Closing this item.

The guard sets its own logger to `warning` (silencing the default INFO
chatter) but uvicorn's access log is configured separately. The
`/v1/chat/completions` URL is logged on every request along with status
code; the request body (which contains user text) is **not** logged by
default uvicorn config. This has been verified manually but is not
locked in by code. Worth a one-line filter that asserts body is never in
the URL.

## L-06 — Synthetic streaming response is single-chunk — **DOCUMENTED AS INTENTIONAL on 2026-05-16**

> **2026-05-16 resolution:** Confirmed intentional. The two guard
> responses (`_GENERAL_BOUNDARY` 18 words, `_PROTO_BOUNDARY` 13
> words) are short enough that chunking adds latency without
> improving UX. The single-chunk SSE format is wire-compatible with
> all OpenAI-style streaming clients. If a future guard response
> exceeds ~50 words, this should be revisited. Closing this item.

When a guard-triggered request includes `stream: true`, the response is
emitted as a single SSE chunk:

```
data: {...payload...}

data: [DONE]
```

This is wire-compatible with OpenAI-style streaming clients but is not
truly incremental. For the deterministic short responses the guard emits
(13–18 words) this is the right trade-off; for longer guard responses in
the future, real chunking would be better UX.

## L-07 — `field` imported but unused in dataclasses — **CLOSED on 2026-05-16**

> **2026-05-16 resolution:** Removed. `tools/v42_boundary_guard.py`
> line 35 changed from `from dataclasses import dataclass, field` to
> `from dataclasses import dataclass`. The matching behavior is
> unchanged (the H18r4 canonical anchor `18e2c5a5...` bound to the
> eval output, not to the source SHA). All 76 guard-suite tests
> continue to pass. Closing this item.

Minor lint issue: `from dataclasses import dataclass, field` imports
`field`, which is not referenced anywhere in the file. Safe to remove on
the next pass that doesn't touch matching behavior.

## What H18r4 *does* anchor

The H18r4 result is sound for:

- The exact 16-rule set with rule IDs DI-01–DI-06, CC-01–CC-04,
  PD-01–PD-03, JB-01–JB-03.
- ASCII-only attack inputs (single user message, no homoglyphs, no
  zero-width characters).
- Single-turn evaluation: one user message per request.
- Canonical evaluation harness: `experiments/canonical_eval.py` with
  seeds 7, 13, 23, 42, 100, n-samples 20, focused-n 100,
  system-prompt-variant `old`.

Any change to the rule set, normalization behavior, or multi-message
scan policy invalidates this anchor and requires a new H-series
hypothesis with a fresh anchor.

## Status summary as of 2026-05-16 evening

| ID | Limitation | Status | Closed-by anchor |
|---|---|---|---|
| L-01 | Unicode bypass (ASCII regex only) | **CLOSED** | H20 anchor `56ce960993f9…` |
| L-02 | Single-message scan only | **CLOSED** | H21 anchor `d916ef63…` |
| L-02b | Client-supplied `role: system` rejection | predeclared as H22; canonical eval in flight |
| L-03 | All-rules walk performance | **Documented as intentional** | n/a |
| L-04 | Pass-through rate limit | **Deployment-layer concern** | n/a (see `docs/gateway_deploy_plan.md`) |
| L-05 | uvicorn access log audit | **Locked in by test** | n/a (`tests/test_v42_boundary_guard_logging.py`) |
| L-06 | Single-chunk synthetic streaming | **Documented as intentional** | n/a |
| L-07 | Unused `field` import | **CLOSED** (removed) | n/a |

All security-significant items resolved (L-01, L-02, L-07) or
predeclared with execution underway (L-02b). All ergonomic items
documented as intentional or routed to the correct architectural
layer (L-03, L-04, L-05, L-06).

## Historical note: original H19 scope

H19, when filed on 2026-05-16 morning, was an attempt to close both
L-01 (Unicode) and L-02 (multi-message) in one hypothesis. It FAILED
due to a suite-design confound on H19-D1 (the multi-message attack
suite tested rule coverage instead of multi-message iteration logic)
and a precommit-vs-suite inconsistency on H19-D2 (the suite expected
rejection of a position-0 role:system that the precommit explicitly
permitted). Both confounds were diagnosed publicly in
[`docs/h19_verdict_2026-05-16.md`](h19_verdict_2026-05-16.md), and the
two gaps were subsequently closed cleanly in separate isolated
hypotheses:

- L-01 via H20 ([`docs/h20_verdict_2026-05-16.md`](h20_verdict_2026-05-16.md))
- L-02 via H21 ([`docs/h21_verdict_2026-05-16.md`](h21_verdict_2026-05-16.md))
- L-02b via H22 ([`docs/h22_precommit_hypothesis_2026-05-16.md`](h22_precommit_hypothesis_2026-05-16.md), eval in flight)

Three anchored PASSES (H20, H21, H22 if it lands) and one anchored
FAIL (H19) all within 36 hours of the original H18r4 promotion, zero
gate relaxation throughout. This is the discipline operating under
its own self-imposed pressure.
