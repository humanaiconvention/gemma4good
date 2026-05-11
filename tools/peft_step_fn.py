"""
peft_step_fn.py — Concrete gradient-step callback for EdgeTTTAdapter, backed by
a PEFT-wrapped LoRA model.

EdgeTTTAdapter takes a `step_fn(OperatorFeedback) -> dict[str, float]` callback
that performs one gradient update and returns per-weight L2 drift values.
This module provides a production implementation backed by a real
`peft.PeftModel`, plus a clean test-injection point.

Architectural symmetry with SimSat's `OnlineLoRAStepper`
(`D:/SimSat/src/sim/observation_vla/lfm_ttt.py`):
  - SimSat's stepper does its own gate evaluation internally
  - Our step_fn does NOT — the EdgeTTTAdapter has already evaluated the
    three TTT gates (error_bias BLOCKING, weight_drift + update_rate
    WARNING) BEFORE calling step_fn. By the time step_fn runs, the step
    has already been authorized.
  - This separation means the same EdgeTTTAdapter wraps both real peft
    training and mocked/simulated training without code change.

Usage:
    from peft import PeftModel
    from transformers import AutoTokenizer
    import torch

    model = PeftModel.from_pretrained(base_model, "haic-gemma4-v42-adapter")
    tokenizer = AutoTokenizer.from_pretrained("...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step = PeftStepFn(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        forward_loss_fn=my_forward_loss_fn,  # or use the default
    )
    adapter = EdgeTTTAdapter(step_fn=step)

    for feedback in operator_feedback_stream:
        record = adapter.step(feedback)
        ...

Per-tensor drift is reported back to EdgeTTTAdapter for the weight_drift
WARNING gate. By convention, we report ONE drift value keyed `"lora_delta"`
that is the L2 norm of the full LoRA delta vs the pre-TTT baseline. The
gate's threshold (`MAX_TTT_WEIGHT_DRIFT = 0.30`) then applies to the
fractional drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tools.edge_ttt_adapter import OperatorFeedback


# Default forward-loss signature: takes (model, tokenizer, feedback) and returns
# (loss_tensor, predicted_action_string_or_none). Tests inject a simpler stub
# that doesn't need a real torch tensor.
ForwardLossFn = Callable[[Any, Any, OperatorFeedback], tuple[Any, Optional[str]]]


def _default_forward_loss(model: Any, tokenizer: Any,
                          feedback: OperatorFeedback) -> tuple[Any, Optional[str]]:
    """Default forward path for a transformers PeftModel with a tokenizer.

    Treats the operator_label as the supervised target and trains the model
    to produce that response. The 'predicted' field on the feedback is the
    model's prior output, used only for signed-error reporting upstream
    (not for the loss itself, which is computed against operator_label).

    For a custom chat template or a non-Gemma model, supply your own
    forward_loss_fn — the EdgeTTTAdapter contract doesn't care how the loss
    is computed, only that the step happens.

    Raises ImportError if torch is not available. The rest of this module
    (PeftStepFn class, drift tracking, EdgeTTTAdapter integration) is
    torch-free and unit-testable without a GPU.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Default forward_loss_fn requires torch. Inject a custom forward_loss_fn "
            "for torch-free testing or non-torch backends."
        ) from e

    # Tokenize input (the predicted side as the prompt; operator_label as the target).
    # NOTE: this is a minimal default. Real deployments will want to use the model's
    # chat template (tokenizer.apply_chat_template) and proper attention-mask handling.
    prompt = feedback.predicted
    target = feedback.operator_label

    encoded = tokenizer(
        prompt + target,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    input_ids = encoded["input_ids"].to(model.device)

    # Mask out the prompt tokens in the labels so the loss only counts target tokens.
    prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
    prompt_len = prompt_tokens.shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100   # standard HF convention: -100 → ignored in loss

    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss, feedback.predicted


@dataclass
class PeftStepFn:
    """Stateful gradient-step callback for `EdgeTTTAdapter`.

    Constructor parameters
    ----------------------
    model : peft.PeftModel
        The LoRA-wrapped model to update. Must be in train mode for backward
        to work; PeftStepFn sets it on entry and restores on exit.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer / processor for `model`.
    optimizer : torch.optim.Optimizer
        Optimizer over the LoRA parameters. Caller instantiates once.
    forward_loss_fn : Callable, optional
        Signature `(model, tokenizer, feedback) -> (loss_tensor, predicted_action_or_None)`.
        Defaults to `_default_forward_loss` which expects a HuggingFace
        Causal-LM-style model.

    State (read-only)
    -----------------
    update_count : int
        Cumulative number of step_fn invocations.
    initial_lora_l2 : float
        L2 norm of the LoRA-delta at the moment of the first step_fn call.
        Subsequent calls report L2 drift VS THIS baseline. Reset by calling
        `.reset_baseline()` (e.g., after a federation merge).
    """

    model: Any
    tokenizer: Any
    optimizer: Any
    forward_loss_fn: ForwardLossFn = _default_forward_loss

    # ── State (private; not part of the constructor contract) ─────────────
    update_count: int = field(default=0, init=False)
    _baseline_state: Optional[dict[str, Any]] = field(default=None, init=False, repr=False)
    _baseline_l2: float = field(default=0.0, init=False)

    def __call__(self, feedback: OperatorFeedback) -> dict[str, float]:
        """One gradient step. Called by EdgeTTTAdapter after the gates pass.

        Returns
        -------
        dict[str, float]
            `{"lora_delta": <float>}` — the L2 drift of the LoRA delta vs
            the baseline state. EdgeTTTAdapter's weight_drift WARNING gate
            uses this to detect adapter over-fitting.
        """
        # Lazy baseline capture
        if self._baseline_state is None:
            self._baseline_state = self._snapshot_lora_state()
            self._baseline_l2 = self._lora_state_l2(self._baseline_state)

        # Forward + backward + step
        try:
            self.model.train()
            self.optimizer.zero_grad()
            loss, _predicted = self.forward_loss_fn(self.model, self.tokenizer, feedback)
            # The loss may be a torch tensor or a scalar (stub for tests).
            if hasattr(loss, "backward"):
                loss.backward()
                self.optimizer.step()
        finally:
            try:
                self.model.eval()
            except AttributeError:
                pass  # mocks may not have eval()

        self.update_count += 1

        # Compute drift vs baseline
        current_state = self._snapshot_lora_state()
        current_l2 = self._lora_state_l2(current_state)
        if self._baseline_l2 > 1e-9:
            drift = abs(current_l2 - self._baseline_l2) / self._baseline_l2
        else:
            drift = current_l2

        return {"lora_delta": float(drift)}

    def reset_baseline(self) -> None:
        """Forget the current baseline. The next `__call__` will capture a new one.

        Call this after a federation merge has updated the model weights, so the
        weight_drift gate measures drift from the NEW post-merge baseline rather
        than the pre-round one.
        """
        self._baseline_state = None
        self._baseline_l2 = 0.0
        # update_count intentionally NOT reset — that's update_rate's domain
        # and is reset by constructing a new PeftStepFn.

    # ── Internals ─────────────────────────────────────────────────────────

    def _snapshot_lora_state(self) -> dict[str, Any]:
        """Capture {name: param.detach().clone()} for every trainable LoRA tensor.

        Tests can override the model's parameters API (mock provides a
        `named_parameters` iterator); production uses real torch tensors.
        """
        out: dict[str, Any] = {}
        for name, param in self.model.named_parameters():
            if "lora" in name and getattr(param, "requires_grad", True):
                # Use detach + clone to capture a stable snapshot.
                if hasattr(param, "detach"):
                    out[name] = param.detach().clone()
                else:
                    out[name] = param  # mock case
        return out

    def _lora_state_l2(self, state: dict[str, Any]) -> float:
        """L2 norm summed over all LoRA tensors."""
        total_sq = 0.0
        for _, t in state.items():
            if hasattr(t, "pow") and hasattr(t, "sum"):
                # Real torch tensor
                total_sq += float(t.pow(2).sum().item())
            else:
                # Mock: assume it's a float or has .total_sq attribute
                total_sq += float(getattr(t, "l2_sq", t) or 0.0)
        return total_sq ** 0.5
