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

## L-02 — Only last user message is evaluated (HIGH)

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

## L-03 — All rules walked per request even after a match (LOW)

`apply_guard` collects every matching rule before picking `matched[0]` as
the primary. This is intentional for diagnostic logging (the
`matched_rule_ids` field surfaces all overlaps in `guard_metadata`) but
costs O(n_rules × |text|) on long inputs. With 16 rules and short
interview turns this is invisible; if rules grow into the hundreds it
becomes meaningful.

**Mitigation path:** add a `short_circuit: bool = False` arg to
`apply_guard` and pass it from the request handler when full diagnostics
are not needed (e.g. production hot path).

## L-04 — Pass-through proxy has no rate limit

The `/v1/chat/completions` pass-through forwards benign turns directly to
the upstream llama-server with no rate limiting at the guard layer. The
upstream itself is on `127.0.0.1` and not exposed externally, so the real
attack surface is whatever sits in front of the guard (the operator's
ingress). But for defense in depth, a token-bucket limiter on the guard
would be cheap.

## L-05 — uvicorn access log is not filtered

The guard sets its own logger to `warning` (silencing the default INFO
chatter) but uvicorn's access log is configured separately. The
`/v1/chat/completions` URL is logged on every request along with status
code; the request body (which contains user text) is **not** logged by
default uvicorn config. This has been verified manually but is not
locked in by code. Worth a one-line filter that asserts body is never in
the URL.

## L-06 — Synthetic streaming response is single-chunk

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

## L-07 — `field` imported but unused in dataclasses

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

## Proposed H19 scope

If pursued, H19 should be a single precommitted hypothesis that:

1. Adds NFKC normalization + zero-width stripping to `_extract_user_text`.
2. Scans every user-role message and rejects client-supplied system roles.
3. Re-runs the full H18 canonical set unchanged.
4. Adds a Unicode-bypass benign-FP suite (legitimate non-Latin input)
   with predeclared FP threshold ≤ 0.02.
5. Declares pass criteria identical to H18 plus the new FP gate.

Until that hypothesis is run and an anchor is produced, the L-01 and L-02
gaps remain documented and unmitigated.
