"""Insert an MPK provenance-check cell into the submission notebook.

One markdown cell + one code cell, placed AFTER Scenario 5 and BEFORE the
final eval section. The code cell is gated behind `MPK_ENABLED = True`
and degrades gracefully if MPK isn't installed or Gemma-4 isn't in MPK's
reference dataset.

Run once; commit the resulting notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("D:/gemma4good/notebook/haic_gemma4_governance.ipynb")


MD_INTRO = """\
---

## Scenario 6 (advisory): Statistical Model-Derivation Audit (Cisco MPK)

The first five governance tools attest that `haic-gemma4-v42` was *trained
with proper consent, geometry-tracked, NLA-explained, and Merkle-anchored*.
This cell adds a sixth, third-party check: **is `haic-gemma4-v42` actually
derived from `google/gemma-4-e2b-it` at the weight level?**

We use Cisco AI Defense's **Model Provenance Kit** (MPK, v1.0.0, released
2026-05-04, Apache-2.0 code + CC BY 4.0 reference dataset). MPK extracts
five weight-level signals — EAS, END, NLF, LEP, WVC — and produces a
composite identity score in [0, 1]. Per Cisco's documented tiers:

| Score | Tier |
|---|---|
| > 0.75 | High-confidence match |
| 0.65–0.75 | Weak match |
| ≤ 0.65 | Not matched |
| `pipeline_score == 1.0` or `MFI tier ≤ 2` | Confirmed match |

**Honest disclaimer (from Cisco's own README):** MPK provides strong
statistical evidence of model derivation but is **NOT cryptographic
proof**. It cannot distinguish "trained from the same template" from
"copied weights." We surface MPK's verdict alongside the prior tools'
output, not as a replacement for any of them.

**Gating:** the cell is behind `MPK_ENABLED = True`. The MPK reference
dataset is 908 MB on first download; set `MPK_ENABLED = False` to skip
this section in environments without the disk budget. If MPK doesn't yet
know about Gemma-4-E2B (it was added to HF Hub recently), the cell
reports `model_not_in_database` and falls back to the PRISM geometry
signal from Scenario 1.
"""


CODE_CELL = '''\
# ── Scenario 6 (advisory): Cisco MPK provenance check ─────────────────────
# Verifies haic-gemma4-v42 derives from google/gemma-4-e2b-it at the
# weight level. Gated behind MPK_ENABLED; uses the gemma4good repo's
# tools/audit_provenance.py wrapper (graceful fallback when MPK isn't
# installed or Gemma-4 isn't in the reference dataset).

MPK_ENABLED = True  # set False to skip the 908 MB dataset download

import sys, subprocess, json
from pathlib import Path

# Bootstrap path to the gemma4good repo modules (matches Scenario 5)
if 'MODULES_AVAILABLE' not in dir():
    MODULES_AVAILABLE = False
if not MODULES_AVAILABLE:
    # Reuse the same logic Scenario 5 uses to find the modules
    cwd = Path('.').resolve()
    for c in [cwd] + list(cwd.parents):
        if (c / 'tools' / 'audit_provenance.py').exists():
            sys.path.insert(0, str(c))
            MODULES_AVAILABLE = True
            break
    if not MODULES_AVAILABLE:
        # Try /kaggle/input
        kaggle_input = Path('/kaggle/input')
        if kaggle_input.exists():
            for marker in kaggle_input.rglob('tools/audit_provenance.py'):
                sys.path.insert(0, str(marker.parent.parent))
                MODULES_AVAILABLE = True
                break

if not MODULES_AVAILABLE:
    print("\\u26a0 Scenario 6 skipped: gemma4good modules not found.")
    print("  See Scenario 5 bootstrap-cell output for setup instructions.")
elif not MPK_ENABLED:
    print("\\u26a0 Scenario 6 skipped: MPK_ENABLED = False")
    print("  Set MPK_ENABLED = True to run the 908 MB provenance check.")
else:
    from tools.audit_provenance import execute_audit_provenance

    # Try to install MPK if not present. Quiet pip; this is a notebook.
    try:
        from shutil import which
        if which("provenancekit") is None:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "provenancekit"],
                stdout=subprocess.DEVNULL,
            )
    except Exception as e:
        print(f"\\u26a0 pip install provenancekit failed: {e}")
        print("  Falling back to graceful unavailable verdict.")

    print("Running Cisco MPK compare:")
    print("  candidate: haic-gemma4-v42")
    print("  reference: google/gemma-4-e2b-it")
    print()
    result = execute_audit_provenance({
        "candidate_model": "haic-gemma4-v42",
        "reference_model": "google/gemma-4-e2b-it",
        "enabled":         True,
    })

    print("=" * 60)
    print("MPK Provenance Audit Result")
    print("=" * 60)
    print(f"  Verdict:           {result[\\"verdict\\"]}")
    if result.get("composite_score") is not None:
        print(f"  Composite score:   {result[\\"composite_score\\"]:.4f}")
    if result.get("five_signals"):
        print(f"  Five signals:")
        for sig in ("EAS", "END", "NLF", "LEP", "WVC"):
            val = result["five_signals"].get(sig)
            if val is not None:
                print(f"    {sig}: {val:.4f}")
    if result.get("mpk_version"):
        print(f"  MPK version:       {result[\\"mpk_version\\"]}")
    print(f"  Audit hash:        {result[\\"audit_hash\\"]}")
    print()
    print(f"  Disclaimer: {result[\\"disclaimer\\"]}")
    print()
    print(f"  Citation:   {result[\\"citation\\"]}")
    if result.get("notes"):
        print()
        print("  Notes:")
        for n in result["notes"]:
            print(f"    - {n}")

    # Fold into the run's audit JSON alongside PRISM + SGT
    try:
        # If a prior cell set up a `governance_traces` dict, append to it.
        # Otherwise, this is a smoke run and we just keep the result above.
        if "governance_traces" in dir():
            governance_traces.setdefault("scenario_6_mpk_provenance", []).append(result)
    except Exception:
        pass
'''


def make_markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def make_code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    # Find the last Scenario 5 cell — the receipt-chain cell with "FEDERATION RECEIPT"
    insert_at = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            if "FEDERATION RECEIPT" in src and "scenario-5-fed-round-1" in src:
                insert_at = i + 1
                break
    if insert_at is None:
        # Fallback: insert before the final eval markdown
        for i, c in enumerate(nb["cells"]):
            if c.get("cell_type") == "markdown" and "Final Evaluation" in "".join(c.get("source", [])):
                insert_at = i
                break
    if insert_at is None:
        raise RuntimeError("Could not find insertion point for Scenario 6 cell")

    new_cells = [make_markdown(MD_INTRO), make_code(CODE_CELL)]
    nb["cells"] = nb["cells"][:insert_at] + new_cells + nb["cells"][insert_at:]
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Inserted 2 new cells at index {insert_at}. New cell count: {len(nb['cells'])}.")


if __name__ == "__main__":
    main()
