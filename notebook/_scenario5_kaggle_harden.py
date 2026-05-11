"""Harden Scenario 5 against Kaggle execution environments.

The original Scenario 5 cells (inserted by `_scenario5_insert.py`) assume the
repo modules are importable from the CWD. On Kaggle, /kaggle/working is the
default CWD and the repo modules aren't present. This patch:

  1. Replaces the first Scenario 5 code cell with a defensive bootstrap that
     locates the runtime grounding loop modules in (a) the CWD or any
     ancestor, (b) /kaggle/input/<dataset>/..., or fails gracefully and sets
     MODULES_AVAILABLE = False.
  2. Wraps each subsequent Scenario 5 code cell with `if MODULES_AVAILABLE:`
     so the notebook continues to execute cleanly even when the modules
     aren't found — the cells print an explanation in that case.

Run once locally; commit the resulting notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("D:/gemma4good/notebook/haic_gemma4_governance.ipynb")


BOOTSTRAP_CELL = '''\
# ── Scenario 5 bootstrap: locate the runtime grounding loop modules ────────
# This cell finds the gemma4good runtime modules (viability/, tools/, utils/)
# on disk. Works in three environments:
#
#   1. LOCAL — repo root on the working directory or any ancestor
#   2. KAGGLE — modules attached as a Kaggle dataset under /kaggle/input/<slug>/
#   3. GITHUB — fallback: clone from the public repo into /kaggle/working
#
# If none succeed, Scenario 5 cells below degrade to "explanation only" so the
# rest of the notebook (Scenarios 1-4 + final eval) continues to execute.

import sys, json, os, subprocess
from pathlib import Path

GITHUB_FALLBACK_URL = "https://github.com/humanaiconvention/gemma4good"
GITHUB_FALLBACK_DIR = "/kaggle/working/gemma4good"


def _find_modules_locally():
    """Search the working directory and ancestors for the modules."""
    cwd = Path(".").resolve()
    candidates = [cwd] + list(cwd.parents)
    for c in candidates:
        if (c / "viability" / "ttt_gates.py").exists():
            return c
    return None


def _find_modules_in_kaggle_input():
    """Recursively search /kaggle/input for the modules."""
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    for marker in root.rglob("viability/ttt_gates.py"):
        return marker.parent.parent
    return None


def _fetch_from_github():
    """Last-resort fallback: shallow-clone the public repo."""
    target = Path(GITHUB_FALLBACK_DIR)
    if (target / "viability" / "ttt_gates.py").exists():
        return target
    if not Path("/kaggle/working").exists():
        return None  # Not on Kaggle; don't write outside CWD
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", GITHUB_FALLBACK_URL, str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if (target / "viability" / "ttt_gates.py").exists():
            return target
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


_modules_root = (
    _find_modules_locally()
    or _find_modules_in_kaggle_input()
    or _fetch_from_github()
)
MODULES_AVAILABLE = _modules_root is not None

if MODULES_AVAILABLE:
    sys.path.insert(0, str(_modules_root))
    print(f"\\u2713 runtime grounding loop modules located at {_modules_root}")
    print(f"  Scenario 5 will run live demonstrations of all four layers.")
else:
    print("\\u26a0 runtime grounding loop modules not found at any of:")
    print("    - current working directory or any ancestor")
    print("    - /kaggle/input/<dataset>/viability/")
    print("    - GitHub fallback clone (network may be disabled)")
    print()
    print("  To run the LIVE Scenario 5 demonstration:")
    print("    LOCAL  : cd to gemma4good repo root, re-run this cell")
    print("    KAGGLE : attach gemma4good as a Kaggle dataset, re-run this cell")
    print(f"    GITHUB : {GITHUB_FALLBACK_URL}")
    print()
    print("  Scenarios 1-4 above are unaffected. Scenario 5 cells will show")
    print("  the demonstration design but skip execution.")
'''


# Each post-bootstrap cell needs to be guarded. We use a simple pattern:
# `if not MODULES_AVAILABLE: ... print explanation ... else: <original code>`
def _guard(original_code: str, scenario_label: str) -> str:
    """Wrap original cell code in the MODULES_AVAILABLE guard."""
    # Indent the original code by 4 spaces for the else branch
    indented = "\n".join("    " + line if line.strip() else line
                         for line in original_code.splitlines())
    return f'''\
if not MODULES_AVAILABLE:
    print("Scenario 5 — {scenario_label} — skipped (modules not available)")
    print("  See the bootstrap cell output above for setup instructions.")
else:
{indented}
'''


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Find the Scenario 5 cells. They are the cells whose source starts with
    # "# ── Federated runtime grounding loop demo" or that follow it,
    # up to (but not including) the final eval markdown cell.
    scenario5_first_idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            if "Federated runtime grounding loop demo (Scenario 5)" in src:
                scenario5_first_idx = i
                break
    if scenario5_first_idx is None:
        raise RuntimeError("Could not find first Scenario 5 cell")

    # The Scenario 5 code cells are 5 consecutive cells starting at scenario5_first_idx.
    # The structure inserted by _scenario5_insert.py is:
    #   [markdown] MD_INTRO
    #   [code]     CODE_FEDERATION_SIM       <-- this is scenario5_first_idx
    #   [code]     CODE_FRAGMENT_PACKAGE
    #   [code]     CODE_SYNCER_VERIFY
    #   [code]     CODE_FEDERATED_ASSESSMENT
    #   [code]     CODE_RECEIPT_CHAIN
    # So we need to harden indices [first_idx, first_idx+5).
    code_indices = list(range(scenario5_first_idx, scenario5_first_idx + 5))

    labels = [
        "L1 simulation (5 clinic learners, week of operator feedback)",
        "L3 fragment packaging (round receipts + LoRA shape summary)",
        "L3 syncer verification (Merkle + consent + shape + norm checks)",
        "L4 federated viability assessment (Ceff_global vs E_global)",
        "Receipt chain (federation Merkle root + zk_digest)",
    ]

    # Replace the first Scenario 5 code cell with the bootstrap PLUS its
    # original content (guarded). This keeps cell count stable.
    first_original = "".join(nb["cells"][scenario5_first_idx].get("source", []))
    new_first_source = BOOTSTRAP_CELL + "\n\n" + _guard(first_original, labels[0])
    nb["cells"][scenario5_first_idx] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_first_source.splitlines(keepends=True),
    }

    # Guard the remaining 4 cells
    for idx, label in zip(code_indices[1:], labels[1:]):
        original = "".join(nb["cells"][idx].get("source", []))
        guarded = _guard(original, label)
        nb["cells"][idx] = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": guarded.splitlines(keepends=True),
        }

    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Notebook hardened. Updated {len(code_indices)} cells starting at index {scenario5_first_idx}.")


if __name__ == "__main__":
    main()
