# Audit: humanai-convention/tools/improvement_pipeline.py

*Applies the gemma4good eval doctrine to the broader convention's
production pipeline. Names the parity gaps and proposes a port.*

---

## Scope

The HumanAI Convention monorepo at `D:/humanai-convention/` contains
the production grounding pipeline at
[`tools/improvement_pipeline.py`](../../humanai-convention/tools/improvement_pipeline.py)
(541 lines). It runs the full
**train → evaluate → promote → recompute** cycle as a scheduled job
(`haic-improvement-pipeline` in
[agents/control-plane/registry.json](../../humanai-convention/agents/control-plane/registry.json)).

The gemma4good repo has a parallel doctrine at
[docs/evaluation_doctrine.md](./evaluation_doctrine.md) and a six-gate
mechanism at [tools/check_promotion.py](../tools/check_promotion.py).

This audit lists where the production pipeline does not yet enforce
that doctrine, and what would be required to bring it into parity.

---

## What the production pipeline currently does

From `improvement_pipeline.py::stage_promote` (lines ~325-360):

```python
def stage_promote(run, benchmark_results, new_model_path,
                  version_tag, min_gain, dry_run):
    new_score      = benchmark_results.get("new_t3_overall",      0.0)
    baseline_score = benchmark_results.get("baseline_t3_overall", 0.0)
    gain = new_score - baseline_score
    if gain < min_gain:                           # default 0.02
        run.record_stage("promote", "no_gain", ...)
        return False
    _write_version_pointer(version_tag)
    return True
```

The promotion decision is a **single-threshold check** on T3 overall
score: did `new` beat `baseline` by ≥ 2pp? If yes, promote.

This is the same shape of decision the kaggle in-kernel SGT used for
v38 — point score, no CI, no leakage check, no Merkle receipt, no
non-compensatory chain.

The CS1 defense in CLAUDE.md (`Proportional promotion decisions`)
adds a "2 consecutive benchmark failures before flagging for manual
review" rule, which is good defense against single-run noise but not
a substitute for confidence intervals.

## Gap analysis vs the eval doctrine

| Eval doctrine gate | Production pipeline status |
|---|---|
| 1. Capability gain (Δ-vs-base + non-overlapping CIs) | **PARTIAL.** `gain < min_gain` checks Δ but not CI overlap; a 2pp gain that's pure sampling noise still passes. |
| 2. Eval-set leakage | **MISSING.** No SHA hashing of eval scenarios vs training shards. |
| 3. Measurement consistency (det-vs-samp) | **MISSING.** T3 benchmark runs once per scenario; no sampling pass. |
| 4. Participation covenant (reproducibility receipt) | **PARTIAL.** Run log includes some metadata; doesn't include harness commit SHA, scenario hashes, or seed. |
| 5. Component isolation (eval-precision = deploy-precision) | **MISSING.** No mechanism to enforce or even record this. |
| 6. Epistemic alignment (lower CI bound, scenario diversity) | **MISSING.** No Wilson CI computed. |

In total: **1 of 6 gates partially enforced, 5 missing or unmeasured.**

## What this looks like in production behavior

The production pipeline at `gain ≥ 0.02` could promote any of:

- A model that gained 3pp on T3 overall but lost 5pp on the security
  subset (gate 6 would block this).
- A model that gained 3pp on T3 overall by memorizing a leaked test
  scenario (gate 2 would block this).
- A model that's tested at 4-bit-nf4 but deployed at GGUF Q5
  (gate 5 would flag this).
- A model with a noisy 3pp lift indistinguishable from sampling
  variance (gate 1 with CI overlap would block this).

None of these have happened yet. They could.

## Proposed port

Two scopes — fast and full.

### Fast port (1–2 commits)

1. Add a CI computation to `stage_promote`. Use Wilson 95% on the T3
   overall pass-rate and require the lower CI bound clears a threshold,
   AND require finetune-CI and baseline-CI are disjoint.
2. Add a leakage check before `stage_export`. Hash all session lattice
   files; hash all benchmark scenarios; reject if any scenario hash
   appears in a lattice file.
3. Record a Merkle root for each pipeline run by extending
   `PipelineRun.summary()` to include hashes of (model_id,
   benchmark_scenarios, training_shards, decision_inputs).

This brings gates 1, 2, and 4 into the production pipeline without
touching the existing T3 benchmark infrastructure.

### Full port (1–2 weeks)

Replace the T3 benchmark with the rigorous SGT harness from
gemma4good ([experiments/sgt_harness.py](../experiments/sgt_harness.py)).
Wire `tools/check_promotion.py` (also from gemma4good) into
`stage_promote`. Mint an `eval_receipt_root` per run via
[tools/eval_receipt.py](../tools/eval_receipt.py). All six gates
become mechanically enforced.

This requires:
- Either pulling the gemma4good eval pipeline into humanai-convention
  as a shared library (preferred — single source of truth for the
  doctrine), OR copying the relevant modules.
- Updating any downstream consumers of `benchmark_results` to use the
  new shape.
- Documenting the new promotion gate in HAIC's existing transparency
  commitments.

## HAIC consistency check

The convention's founding doctrine commits the project to:

> *"The Convention does not 'vouch' for truth; it certifies provenance
> and viability."*  ([docs/founding-doctrine.md](../../humanai-convention/docs/founding-doctrine.md))

A production pipeline that promotes models on a single un-CI'd benchmark
score is, by this doctrine's own standard, "vouching" rather than
"certifying provenance." The fast port closes the gap by producing a
provenance receipt for every promotion decision; the full port
mechanically enforces the gates the receipt records.

This is not a criticism of the existing pipeline — it predates the
doctrine. The audit is what's required to bring the implementation
in line with the doctrine the project has already published.

## Recommended sequence

1. **Land the gemma4good eval pipeline** (this branch, eventually
   merged to humanai-convention). Garrett's harness commit is the
   foundation; the six-gate decision and Merkle receipt are the
   mechanism.
2. **Open a PR against humanai-convention's improvement pipeline**
   that adopts the fast port. Three small atomic commits:
   - Wilson CI in `stage_promote`
   - Leakage check before `stage_export`
   - Merkle root in `PipelineRun.summary()`
3. **Schedule a spike** for the full port. The blocker is whether
   gemma4good's eval pipeline becomes a shared library or whether
   humanai-convention copies it. That's an architectural decision
   above this audit's scope.

## What this audit is not

- Not a request to break production. The existing pipeline works for
  what it does; the gaps are where it should do more, not where it
  does the wrong thing.
- Not a claim that v6/v7/v8 (the prior production models) failed any
  of these gates. They were promoted under the existing gain ≥ 2pp
  rule, which they cleared. The gates here would have produced a
  more precise verdict, but not necessarily a different one.
- Not blocking on this audit. The eval doctrine + tooling can ship in
  gemma4good independently and prove themselves there before being
  ported. v38's two-framework agreement (Tier 3 + eval doctrine both
  saying NOT PROMOTED) is the kind of evidence that supports the port.

---

*Audit author: Claude Opus 4.7, working from
`D:/humanai-convention/.claude/worktrees/xenodochial-nash-46d4b3`
on behalf of the autonomous overnight session of 2026-05-08.*
