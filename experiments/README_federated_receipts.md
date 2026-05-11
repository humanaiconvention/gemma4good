# Federated Round Receipts — Sample Outputs

Each receipt is produced by `tools/federated_round_demo.py` with parameters
mirroring one of the three Gemma4Good deployment scenarios. The receipt is a
self-anchored JSON document with a Merkle root over all accepted fragments
and a zk_digest that can be shared without revealing the underlying session
content.

These are reproducible: re-running with the same `--seed` produces an
identical receipt (only the wall-clock `ts` field differs).

| File | Scenario | Learners | Sessions/Round | Quorum K | Bias fraction | Recommendation |
|---|---|---|---|---|---|---|
| `fed_receipt_clinic.json` | Rural health clinic federation | 5 | 12 (weekly) | 3 | 0% | COMMIT |
| `fed_receipt_classroom.json` | Indonesian low-connectivity classroom | 12 | 20 (weekly) | 6 | 0% | COMMIT |
| `fed_receipt_deforestation.json` | Amazon deforestation monitoring | 20 | 30 (daily) | 10 | 10% | COMMIT |

**`runtime_loop_stress_report.json`** — companion exercise running the
runtime loop through 7 adversarial streams (baseline_clean, systematic_bias,
hostile_fragment, cloud_blackout, consent_denial, poisoning,
federation_collapse). 7/7 expected behaviours observed.

**`federated_round_demo_receipt.json`** — baseline 5-learner demo with 40%
bias fraction (legacy file from initial demo).

## Reproducibility

Each receipt was produced with:

```bash
python tools/federated_round_demo.py \
    --n-learners <N> --bias-fraction <F> --n-sessions <S> --quorum <K> \
    --seed <SEED> --quiet --out experiments/<file>.json
```

The exact commands are recorded in the commit message that introduced each
file. Re-running on a different machine with Python 3.10+ should produce
identical Merkle roots given the same seed; only the `ts` and
`self_anchor` (which includes `ts`) fields differ.

## What the receipt contains

```
{
  "kind": "federated_round_receipt",
  "round_id": 1,
  "config": {n_learners, bias_fraction, n_sessions, quorum},
  "layer_1_per_learner": {learner_id: {bias, n_applied, n_skipped}},
  "layer_2_per_learner": {learner_id: {all_passed, num_passed, failure_reasons}},
  "layer_3_per_learner": {learner_id: {verified, failure_reason, warnings, computed_round_root}},
  "layer_4": {viable_global, ceff_global, e_global, num_verified, num_rejected,
              quorum_met, merge_error_estimate, rejected_learners, round_recommendation},
  "federation_root": "<SHA3-256 over accepted round roots>",
  "zk_digest": "<SHA3-256(federation_root || 'fed-round-1')>",
  "self_anchor": "<SHA3-256 over sorted-keys JSON of this report, excluding self_anchor>"
}
```

The `zk_digest` is the single 64-character value that anyone can use to
verify "this round was governed under the HAIC pipeline" without revealing
any underlying session content.
