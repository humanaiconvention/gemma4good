# Attribution audit — 2026-05-16

*Quick honest check on whether the submission's collaborator attribution
matches reality. Triggered by an earlier observation that public framing
referenced "collaborative entry" but recent activity logs were
single-author.*

## Finding

**The attribution is in good shape.** Earlier concern is retracted on
review.

## Evidence

Recent commits in the repository show the attribution has been actively
maintained, not stale:

- `8872380` "Attribution: add Garrett Sutherland as co-author
  (collaborative entry)"
- `9ba93d4` "Remove invented 'Garrett rigor' nickname from attribution"
- `e0cd721` "Restore 'Garrett rigor' in internal docs (not for external
  use)"
- `273a5a0` "Remove Garrett description line from external docs (DOI
  adjacency fix)"

The pattern shows the operator has been deliberately careful about
internal vs external framing of Garrett's contribution. Specifically:

- **External docs** (README, WRITEUP, Kaggle metadata): co-author
  framing, attribution to the specific load-bearing artifact Garrett
  authored (the SGT harness at commit `674b5e1` and the rigorous-eval
  methodology that grew from it).
- **Internal docs** (`docs/evaluation_doctrine.md`,
  `docs/audit_humanai_convention_pipeline.md`, etc.): more specific
  references to Garrett's design choices.
- **Invented nicknames or framings that didn't match Garrett's actual
  voice**: removed in `9ba93d4`.

This is exactly the right discipline applied to attribution — the same
"don't relax the standard for convenience" pattern shown in the H-series
verdicts.

## What's referenced

Garrett's load-bearing contributions in the current submission:

| Artifact | Where |
|---|---|
| `experiments/sgt_harness.py` | The rigorous SGT harness used by `experiments/canonical_eval.py` |
| `experiments/sgt_extended_scenarios.py` | The extended scenarios (10 grounding + 5 security) |
| `experiments/run_v38_sgt.py` | The single-turn BEAST runner |
| `experiments/scenarios_loader.py` | Scenario loading utilities |
| `tests/test_eval_leakage_check.py` | Test surface for the leakage check |
| WRITEUP.md citation block | `@misc{haslam_sutherland_2026, author = {Haslam, Benjamin and Sutherland, Garrett}, ...}` |

The collaboration is documented through specific contributions, not
just a co-author line. That's the right shape.

## Status

**No action needed.** The attribution discipline matches the
evaluation discipline. The recent commits demonstrate the same
honest-when-it-costs-something pattern. Closing this audit as
PASS.

## Open question (operator-only, optional)

The "with research collaborator Guilherme Ferrari Brescia" note in
WRITEUP.md refers to a citation in `docs/references` for Brescia's
Orch-OS paper. The framing as "research collaborator" is stronger
than a citation. Worth confirming whether that framing matches
Brescia's actual involvement, or whether "we draw on Brescia's
Orch-OS framework" would be the more accurate phrasing for an
external doc. This is the kind of thing a one-line email to Brescia
("we wanted to confirm we're representing your role accurately —
this is what we wrote") closes cleanly. Not urgent; flagged for
operator decision.

Nothing in the gemma4good repo gets changed by this audit. The
finding is: the work is correctly attributed and there's already an
active discipline maintaining it.
