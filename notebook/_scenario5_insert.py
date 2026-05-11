"""Build the Scenario 5 (Federated Deployment Loop) cells and insert into the
governance notebook. Runs once locally; the notebook then ships with the new
cells baked in.

Inserts after Scenario 4 (cell index 31 in the original) and before Final Eval.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("D:/gemma4good/notebook/haic_gemma4_governance.ipynb")


def make_markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def make_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


MD_INTRO = """\
---

## Scenario 5: Federated Runtime Grounding Loop (TTT × DiLoCo × Viability)

The three deployment scenarios above each describe a *single-site* governance trace. In practice, the rural-clinic, low-connectivity-classroom, and deforestation-monitoring scenarios are inherently *federated*: each clinic, classroom, and station runs Gemma 4 locally, with intermittent connectivity. This scenario demonstrates the three-layer runtime grounding loop that makes federation auditable end-to-end:

```
LAYER 1 (per step)        TTT gates — error_bias BLOCKING + 2 warnings
LAYER 2 (per fragment)    DiLoCo verifier — Merkle + consent + shape + norm
LAYER 3 (per federation)  Distributed Viability Condition — Ceff_global > E_global
```

Every operator-feedback signal is traceable from operator click → step gate → round receipt → federation viability decision. See `docs/runtime_grounding_loop_2026-05-11.md` for the full architecture and `docs/diloco_integration_2026-05-11.md` for the DiLoCo theory.

This scenario simulates a federation of 5 rural-clinic learners going through one week of operator feedback, packaging weekly fragments, and the syncer aggregating them under the federated viability condition.
"""

CODE_FEDERATION_SIM = '''\
# ── Federated runtime grounding loop demo (Scenario 5) ────────────────────
# Imports the three new layers from the same codebase the test suite covers.

import sys, json
from pathlib import Path
REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))

from viability.ttt_gates import TTTTrustSnapshot, TTTUpdateRecord, evaluate_ttt
from viability.distributed_viability import (
    LearnerContribution, MergeQuorumPolicy, assess_federated,
)
from tools.edge_ttt_adapter import EdgeTTTAdapter, OperatorFeedback
from tools.diloco_fragment_verifier import (
    FragmentExpectation, FragmentShape, build_fragment_receipt, verify_fragment,
)
from utils.merkle import sha3_256_hex, merkle_root


# 1. Five rural-clinic learners simulate one week of operator feedback.
def simulate_clinic_week(clinic_id: str, *, bias: float = 0.0, n_sessions: int = 12):
    """Simulate one clinic's week: n_sessions of operator feedback.

    `bias` controls the error distribution. bias=0 → balanced; bias > 0 → model
    over-predicts; bias < 0 → model under-predicts. Strong bias should trigger
    the BLOCKING error_bias gate.
    """
    # Tiny pretend step function: each call returns a per-layer drift dict.
    cumulative = {f"w_{k}": 0.0 for k in range(3)}
    def step_fn(feedback):
        for k in cumulative:
            cumulative[k] += 0.002   # ~0.024 drift after 12 steps; well under 0.30
        return dict(cumulative)

    adapter = EdgeTTTAdapter(step_fn=step_fn)
    consents = {layer: True for layer in (
        "transcript", "felt_state", "gfs_activations", "training_signal", "retention"
    )}

    import random
    rng = random.Random(hash(clinic_id) & 0xFFFFFFFF)
    for i in range(n_sessions):
        # Symmetric noise around bias; if bias=0 errors are zero-mean
        err = bias + rng.gauss(0, 0.4)
        fb = OperatorFeedback(
            session_id=f"{clinic_id}-sess-{i:03d}",
            predicted="model_label",
            operator_label="operator_label",
            error=err,
            consent_layers=dict(consents),
        )
        adapter.step(fb)
    return adapter


print("\\n" + "=" * 70)
print("SCENARIO 5 — FEDERATED RUNTIME GROUNDING LOOP (5 clinics, 1 week)")
print("=" * 70)

# Vary bias across the 5 clinics: 3 healthy (low bias), 1 saturated, 1 systematic-bias
clinic_configs = [
    ("clinic-bolivia",  0.0),
    ("clinic-peru",     0.05),
    ("clinic-ecuador", -0.05),
    ("clinic-colombia", 0.4),   # strong positive bias → error_bias should fire
    ("clinic-chile",   -0.4),   # strong negative bias → error_bias should fire
]

adapters = {}
for cid, bias in clinic_configs:
    adapters[cid] = simulate_clinic_week(cid, bias=bias)
    a = adapters[cid]
    print(f"  {cid:20} bias={bias:+.2f} → applied={a.num_applied():2}/{a.num_applied()+a.num_skipped()}, blocked={a.num_skipped()}")
'''

CODE_FRAGMENT_PACKAGE = '''\
# 2. Each clinic packages its week into a DiLoCo fragment for the syncer.
def make_fragment_for_clinic(clinic_id: str, adapter: EdgeTTTAdapter, round_id: int = 1):
    """Build the round receipt + a synthetic LoRA-shape summary for the verifier."""
    trace = adapter.export_receipt()
    # In reality the per-session receipts come from the four-tool governance pipeline
    # (Scenarios 1-3 above). For the demo we synthesise minimal session receipts that
    # match the consent contract; the Merkle root will anchor real per-session traces
    # in production.
    session_receipts = []
    session_consents = []
    full_consent = {layer: True for layer in (
        "transcript", "felt_state", "gfs_activations", "training_signal", "retention"
    )}
    for i in range(len(trace["history"])):
        session_receipts.append({
            "session_id": f"{clinic_id}-sess-{i:03d}",
            "kind": "maestro_session_trace",
            "ts": "2026-05-11T00:00:00Z",
            "ttt_step": trace["history"][i],
        })
        session_consents.append(dict(full_consent))

    receipt = build_fragment_receipt(
        learner_id=clinic_id,
        round_id=round_id,
        dataset_id=f"haic-clinic-week-{round_id}",
        per_session_receipts=session_receipts,
        per_session_consents=session_consents,
    )

    # Build a fake but well-formed LoRA-shape summary (rank-16 over 7 modules x 35 layers).
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")
    names, shapes, norms = [], {}, {}
    for layer in range(35):
        for module in target_modules:
            for ab in ("lora_A", "lora_B"):
                n = f"layers.{layer}.{module}.{ab}.default.weight"
                names.append(n)
                shapes[n] = (16, 1536) if ab == "lora_A" else (1536, 16)
                # Use the cumulative drift from the TTT trace as the per-tensor norm
                # proxy — keeps the demo self-consistent
                norms[n] = max(trace["final_drift_from_baseline"].get("w_0", 0.01) * 5, 0.01)
    shape = FragmentShape(
        tensor_names=names,
        tensor_shapes=shapes,
        tensor_norms=norms,
        total_bytes=sum(16 * 1536 * 2 for _ in names),  # rough fp16 budget
    )
    return receipt, shape, trace


fragments = {}
for cid, _ in clinic_configs:
    fragments[cid] = make_fragment_for_clinic(cid, adapters[cid], round_id=1)
print(f"\\nPackaged {len(fragments)} fragments for sync round 1.")
'''

CODE_SYNCER_VERIFY = '''\
# 3. Syncer verifies each fragment.
print("\\nSYNCER — fragment verification:")
verified_fragments = []
rejected_fragments = []
for cid, _ in clinic_configs:
    receipt, shape, trace = fragments[cid]
    result = verify_fragment(receipt, shape)
    status = "VERIFIED" if result.verified else f"REJECTED({result.failure_reason})"
    print(f"  {cid:20} → {status}")
    if result.verified:
        verified_fragments.append(cid)
    else:
        rejected_fragments.append((cid, result.failure_reason))

# 4. Now intentionally tamper with one clinic's claimed root and re-verify to
# show the verifier catches it (demonstrates the gate is real, not vacuous).
print("\\n  -- demonstrating tamper detection --")
tampered_receipt, tampered_shape, _ = fragments["clinic-bolivia"]
import copy
bad_receipt = copy.copy(tampered_receipt)
bad_receipt.claimed_round_root = "deadbeef" * 8
tamper_result = verify_fragment(bad_receipt, tampered_shape)
print(f"  clinic-bolivia (TAMPERED claimed_root) → "
      f"{'VERIFIED' if tamper_result.verified else 'REJECTED(' + tamper_result.failure_reason + ')'}")
'''

CODE_FEDERATED_ASSESSMENT = '''\
# 5. Federation-level viability assessment.
contributions = []
for cid, bias in clinic_configs:
    receipt, shape, trace = fragments[cid]
    is_verified = cid in verified_fragments
    # Quantization hostility: model with strong bias has higher hostility
    hostility = 0.05 + abs(bias) * 0.1
    contributions.append(
        LearnerContribution(
            learner_id=cid,
            sessions_per_round=float(adapters[cid].num_applied()),  # only applied steps count toward Ceff
            avg_turns_per_session=6.0,
            consent_grant_rate=0.9,
            quantization_hostility=hostility,
            is_verified=is_verified,
        )
    )

policy = MergeQuorumPolicy(minimum_quorum=3)
fed_result = assess_federated(contributions, policy=policy)

print("\\n" + "=" * 70)
print("FEDERATED VIABILITY ASSESSMENT — Round 1")
print("=" * 70)
print(f"  Total learners:        {fed_result.num_learners_total}")
print(f"  Verified (in Ceff):    {fed_result.num_learners_verified}")
print(f"  Rejected:              {fed_result.num_learners_rejected}")
print(f"  Quorum met (K={fed_result.quorum_minimum})?:   {fed_result.quorum_met}")
print(f"  Ceff_global:           {fed_result.ceff_global:.2f}")
print(f"  E_global:              {fed_result.e_global:.4f}")
print(f"  Merge error (1/√K):    {fed_result.merge_error_estimate:.4f}")
print(f"  Viable globally:       {fed_result.viable_global}")
print(f"  Round recommendation:  {fed_result.round_recommendation.upper()}")
'''

CODE_RECEIPT_CHAIN = '''\
# 6. Receipt chain — the federation as a whole produces a Merkle root over all
# accepted fragments, anchoring the whole round in one verifiable hash.
import json as _json
round_leaves = []
for cid in verified_fragments:
    receipt, _, _ = fragments[cid]
    round_leaves.append(receipt.claimed_round_root)

federation_root = merkle_root(round_leaves) if round_leaves else sha3_256_hex("empty_round")
print("\\nFEDERATION RECEIPT")
print(f"  Round 1 federation Merkle root: {federation_root}")
print(f"  zk_digest = SHA3-256(root || 'scenario-5-fed-round-1'):")
zk = sha3_256_hex(federation_root + "scenario-5-fed-round-1")
print(f"    {zk}")

print("\\n" + "=" * 70)
print("Scenario 5 receipt chain is now anchored. Every gradient step in this")
print("federation's round 1 is traceable from operator click → TTT gate →")
print("DiLoCo verifier → federated viability decision → this 64-char digest.")
print("=" * 70)
'''


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    new_cells = [
        make_markdown(MD_INTRO),
        make_code(CODE_FEDERATION_SIM),
        make_code(CODE_FRAGMENT_PACKAGE),
        make_code(CODE_SYNCER_VERIFY),
        make_code(CODE_FEDERATED_ASSESSMENT),
        make_code(CODE_RECEIPT_CHAIN),
    ]

    # Find the "Scenario 4" markdown cell. Insert AFTER its accompanying code cell.
    # In the survey, Scenario 4 is cell index 30 (markdown) + 31 (code).
    # We want to insert AFTER cell 31 — i.e. at index 32.
    insert_at = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "markdown":
            src = "".join(c.get("source", []))
            if "Scenario 4" in src and "Incremental Grounding Loop" in src:
                # Found it. Insert after the next code cell.
                # Look ahead for the next code cell
                for j in range(i + 1, len(nb["cells"])):
                    if nb["cells"][j].get("cell_type") == "code":
                        insert_at = j + 1
                        break
                break

    if insert_at is None:
        # Fallback: insert before the final-eval markdown cell ("Final Evaluation")
        for i, c in enumerate(nb["cells"]):
            if c.get("cell_type") == "markdown":
                src = "".join(c.get("source", []))
                if "Final Evaluation" in src:
                    insert_at = i
                    break

    if insert_at is None:
        raise RuntimeError("Could not find insertion point in notebook")

    print(f"Inserting {len(new_cells)} new cells at index {insert_at}")
    new_nb_cells = nb["cells"][:insert_at] + new_cells + nb["cells"][insert_at:]
    nb["cells"] = new_nb_cells

    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"Notebook updated. New cell count: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
