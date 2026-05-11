"""Tests for tools/peft_step_fn.py — concrete gradient-step callback.

These tests use a torch-free mock model + optimizer so they run in milliseconds
without a GPU. The contract being verified is that PeftStepFn:
  - captures a baseline lazily on first call
  - computes L2 drift vs baseline correctly
  - returns the {"lora_delta": float} dict EdgeTTTAdapter expects
  - composes cleanly with EdgeTTTAdapter's gate logic
  - reset_baseline() forgets the snapshot
  - update_count tracks invocations
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.edge_ttt_adapter import EdgeTTTAdapter, OperatorFeedback
from tools.peft_step_fn import PeftStepFn


# ── Mock infrastructure (torch-free) ────────────────────────────────────────

class _MockTensor:
    """Minimal stand-in for a torch tensor — just needs .pow().sum().item()
    semantics for L2 norm calculation, and clone()/detach() for state capture."""

    def __init__(self, value: float):
        self.value = value

    def pow(self, p: int) -> "_MockTensor":
        return _MockTensor(self.value ** p)

    def sum(self) -> "_MockTensor":
        return self

    def item(self) -> float:
        return self.value

    def detach(self) -> "_MockTensor":
        return self

    def clone(self) -> "_MockTensor":
        return _MockTensor(self.value)


@dataclass
class _MockModel:
    """Mock PeftModel exposing named_parameters and train/eval."""

    params: dict[str, _MockTensor] = field(default_factory=dict)
    device: str = "cpu"
    train_called: int = 0
    eval_called: int = 0
    grow_per_step: float = 0.0   # how much each tensor "grows" per step

    def named_parameters(self):
        return list(self.params.items())

    def train(self):
        self.train_called += 1

    def eval(self):
        self.eval_called += 1

    def _grow(self):
        """Simulate weights changing after backward+step."""
        for name in self.params:
            self.params[name] = _MockTensor(self.params[name].value + self.grow_per_step)


class _MockLoss:
    """Stand-in for a torch loss tensor."""

    def __init__(self, value: float, model: _MockModel):
        self.value = value
        self.model = model
        self.backward_called = 0

    def backward(self):
        self.backward_called += 1
        # Simulate that backward+step will modify weights
        # (the real model would do this in optimizer.step; mock does it here for simplicity)
        self.model._grow()


class _MockOptimizer:
    """No-op optimizer; the mock model handles weight changes itself."""

    def __init__(self):
        self.step_count = 0
        self.zero_grad_count = 0

    def step(self):
        self.step_count += 1

    def zero_grad(self):
        self.zero_grad_count += 1


def _setup(grow_per_step: float = 0.0) -> tuple[PeftStepFn, _MockModel, _MockOptimizer]:
    """Build a PeftStepFn over mocks with controllable growth rate."""
    model = _MockModel(
        params={
            "layers.0.q_proj.lora_A.weight": _MockTensor(1.0),
            "layers.0.q_proj.lora_B.weight": _MockTensor(0.5),
            "layers.0.k_proj.lora_A.weight": _MockTensor(1.0),
            # Non-LoRA param — must be excluded from drift
            "layers.0.attn.weight": _MockTensor(99.0),
        },
        grow_per_step=grow_per_step,
    )
    optimizer = _MockOptimizer()

    def stub_forward_loss(m, tok, fb):
        return _MockLoss(0.5, m), fb.predicted

    step = PeftStepFn(
        model=model,
        tokenizer=None,
        optimizer=optimizer,
        forward_loss_fn=stub_forward_loss,
    )
    return step, model, optimizer


def _fb(error: float = 0.0) -> OperatorFeedback:
    return OperatorFeedback(
        session_id="s-1",
        predicted="A",
        operator_label="B",
        error=error,
        consent_layers={
            "transcript": True, "felt_state": True, "gfs_activations": True,
            "training_signal": True, "retention": True,
        },
    )


# ── Tests ───────────────────────────────────────────────────────────────────

def test_step_fn_returns_lora_delta_key():
    step, _, _ = _setup(grow_per_step=0.0)
    result = step(_fb())
    assert "lora_delta" in result
    assert isinstance(result["lora_delta"], float)


def test_no_growth_means_zero_drift():
    """If the optimizer doesn't change weights, drift stays 0."""
    step, _, _ = _setup(grow_per_step=0.0)
    for _ in range(5):
        result = step(_fb())
    assert result["lora_delta"] < 1e-6


def test_growth_produces_positive_drift():
    """Growing weights → measurable drift."""
    step, _, _ = _setup(grow_per_step=0.1)
    drifts = [step(_fb())["lora_delta"] for _ in range(3)]
    # Drift should be monotonically increasing
    for i in range(1, len(drifts)):
        assert drifts[i] > drifts[i - 1], f"drift not monotonic: {drifts}"


def test_baseline_captured_lazily_on_first_call():
    """Baseline state is None until the first call."""
    step, _, _ = _setup()
    assert step._baseline_state is None
    step(_fb())
    assert step._baseline_state is not None


def test_reset_baseline_forgets_snapshot():
    step, model, _ = _setup(grow_per_step=0.1)
    step(_fb())
    step(_fb())
    assert step._baseline_state is not None
    step.reset_baseline()
    assert step._baseline_state is None
    # Next call captures NEW baseline at the (already-grown) state
    step(_fb())  # one more growth step
    new_drift = step(_fb())["lora_delta"]
    # Drift should be small after reset (only one growth since the new baseline)
    assert new_drift < 0.5


def test_update_count_increments_per_call():
    step, _, _ = _setup()
    assert step.update_count == 0
    for i in range(5):
        step(_fb())
    assert step.update_count == 5


def test_optimizer_step_and_zero_grad_called_per_step():
    step, _, opt = _setup()
    for _ in range(4):
        step(_fb())
    assert opt.step_count == 4   # via the mock loss.backward() -> implicit via mock
    assert opt.zero_grad_count == 4


def test_train_eval_called_around_each_step():
    """Model is set to train() before backward and eval() after."""
    step, model, _ = _setup()
    step(_fb())
    assert model.train_called == 1
    assert model.eval_called == 1


def test_lora_filter_excludes_non_lora_params():
    """Non-LoRA params shouldn't appear in the snapshot."""
    step, _, _ = _setup()
    snap = step._snapshot_lora_state()
    assert "layers.0.q_proj.lora_A.weight" in snap
    assert "layers.0.attn.weight" not in snap


# ── End-to-end with EdgeTTTAdapter ──────────────────────────────────────────

def test_composes_with_edge_ttt_adapter():
    """The whole point: PeftStepFn slots into EdgeTTTAdapter as the step_fn."""
    step, _, _ = _setup(grow_per_step=0.01)
    adapter = EdgeTTTAdapter(step_fn=step)

    # Run a clean stream (no bias) — all should apply
    import random
    rng = random.Random(42)
    for i in range(10):
        fb = _fb(error=rng.gauss(0, 0.5))
        adapter.step(fb)

    assert adapter.num_applied() == 10
    assert step.update_count == 10
    # The drift dict gets propagated into adapter._snapshot.drift_from_baseline
    assert "lora_delta" in adapter._snapshot.drift_from_baseline


def test_blocking_gate_prevents_step_fn_call():
    """When the BLOCKING error_bias gate fires, step_fn must NOT be invoked."""
    step, model, opt = _setup(grow_per_step=0.01)
    adapter = EdgeTTTAdapter(step_fn=step)

    # Send 10 same-sign positive errors to warm up the BLOCKING gate
    for i in range(10):
        adapter.step(_fb(error=0.5))

    # All 10 should apply (gate is vacuous until window full)
    assert step.update_count == 10

    # 11th same-sign error: gate fires, step_fn must NOT be called
    record = adapter.step(_fb(error=0.5))
    assert not record.applied
    assert record.blocked_by == "error_bias"
    # step.update_count should NOT have advanced
    assert step.update_count == 10


def test_weight_drift_warning_propagates_through_adapter():
    """Large drift from the step_fn should surface as a weight_drift warning
    on subsequent adapter step receipts (post-step gate)."""
    # Grow aggressively so drift exceeds the 0.30 threshold quickly
    step, _, _ = _setup(grow_per_step=2.0)
    adapter = EdgeTTTAdapter(step_fn=step)
    # Diverse errors to keep error_bias clear
    import random
    rng = random.Random(0)
    saw_warning = False
    for i in range(15):
        record = adapter.step(_fb(error=rng.gauss(0, 0.5)))
        if "weight_drift" in record.notes:
            saw_warning = True
            break
    assert saw_warning, "expected weight_drift warning under aggressive growth"
