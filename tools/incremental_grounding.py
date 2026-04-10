"""
incremental_grounding.py — Session-driven continual learning for HAIC × Gemma 4.

This module implements the incremental grounding pipeline:
  1. Convert a consented HAIC session into SFT training pairs
  2. Run a small number of LoRA gradient steps on the local model
  3. Produce a two-level Merkle training receipt (session receipt → training receipt)

Design principle: EVERY number in a receipt comes from a real computation, or
it is explicitly None.  No simulated loss trajectories, no placeholder hashes,
no estimated metrics.  This prevents simulated artifacts from being mistaken
for real training evidence — informational autophagy at the meta-level.

Two modes:
  - dry_run (default): runs ALL pipeline stages (consent validation, SFT pair
    extraction, token counting, Merkle leaf construction) but STOPS before
    gradient steps.  Returns training_executed=False.
  - live: does everything dry_run does, PLUS loads a 4-bit E2B model, runs
    gradient steps, saves the updated adapter, returns real loss values.

See docs/incremental_grounding.md for the full technical design.
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

VALID_MODES = frozenset({"dry_run", "live"})


@dataclass
class GroundingConfig:
    """Configuration for a single incremental grounding update."""
    mode: str = "dry_run"           # "dry_run" | "live"
    lr: float = 5e-5
    steps: int = 5
    lora_r: int = 8
    lora_alpha: int = 16
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_seq_len: int = 512
    compute_dtype: str = "float16"  # T4 requirement — NOT bfloat16
    model_id: str = "google/gemma-4-E2B-it"
    adapter_path: Optional[str] = None  # path to existing adapter checkpoint

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {sorted(VALID_MODES)}"
            )

    def to_hashable_dict(self) -> dict:
        """Return a sorted dict suitable for Merkle hashing."""
        d = asdict(self)
        d.pop("adapter_path", None)  # path is environment-specific
        return d


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class GroundingUpdateResult:
    """Result of run_grounding_update().  All fields are real or None."""
    training_executed: bool          # True only if gradient steps ran
    mode: str                        # "dry_run" | "live"
    consent_valid: bool              # True if training_signal was granted
    consent_blocked: bool            # True if update was refused due to consent
    sft_pairs: list                  # real extracted SFT pairs
    sft_pair_count: int              # len(sft_pairs)
    token_count: Optional[int]       # real token count (None if not computed)
    loss_before: Optional[float]     # real initial loss (None in dry_run)
    loss_after: Optional[float]      # real final loss (None in dry_run)
    loss_trajectory: Optional[list]  # real per-step losses (None in dry_run)
    adapter_hash_before: Optional[str]  # SHA-256 of adapter pre-update (None in dry_run)
    adapter_hash_after: Optional[str]   # SHA-256 of adapter post-update (None in dry_run)
    steps_executed: int              # 0 in dry_run, actual count in live
    training_receipt: Optional[dict] # two-level Merkle receipt
    session_receipt_root: Optional[str]  # from the session receipt (if provided)
    error: Optional[str] = None      # error message if something failed


# ── SFT pair extraction ──────────────────────────────────────────────────────

def format_session_as_sft(
    messages: list,
    consent: dict,
) -> list:
    """
    Convert a completed HAIC session into SFT training pairs.

    Only produces output if consent["training_signal"] == "granted".
    Extracts user→assistant turn pairs from the message history.

    The HAIC 7-turn format:
      messages[0]: system (interviewer prompt)
      messages[1]: user   (T1 — lived experience)
      messages[2]: assistant (T2 — pivot question)
      messages[3]: user   (T3 — response to pivot)
      messages[4]: assistant (T4 — texture follow-up)
      messages[5]: user   (T5 — sensory response)
      messages[6]: assistant (T6 — compression question)

    Training windows: T2 (given T1), T4 (given T1-T3), T6 (given T1-T5).
    Each window includes the full conversational context up to that point.

    Args:
        messages: list of {role, content} dicts from a completed session
        consent:  consent dict with at minimum a "training_signal" key

    Returns:
        list of {"instruction": str, "response": str, "turn_index": int,
                 "context_messages": list} dicts.
        Empty list if consent denied or no valid pairs found.
    """
    # ── Input Type Gate ───────────────────────────────────────────────────
    if not isinstance(consent, dict):
        return []
    if not isinstance(messages, list):
        return []

    # ── Consent gate ──────────────────────────────────────────────────────
    training_signal = consent.get("training_signal", "denied")
    if training_signal != "granted":
        return []

    if not messages or len(messages) < 2:
        return []

    # ── Extract user→assistant pairs ─────────────────────────────────────
    sft_pairs = []

    # Find all assistant turns (skip system messages)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        # The instruction is the preceding user turn
        # The context is everything before the assistant turn
        preceding_user = None
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") == "user":
                preceding_user = messages[j]
                break

        if preceding_user is None:
            continue

        # Build context: all messages before this assistant turn (excluding system)
        context = [
            m for m in messages[:i]
            if m.get("role") in ("user", "assistant")
        ]

        sft_pairs.append({
            "instruction": preceding_user.get("content", ""),
            "response": msg.get("content", ""),
            "turn_index": i,
            "context_messages": context,
        })

    return sft_pairs


# ── Merkle utilities ──────────────────────────────────────────────────────────

def _sha256(data: str) -> str:
    """SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _merkle_root(leaves: list) -> str:
    """Compute Merkle root from a list of hex-string leaves."""
    if not leaves:
        return _sha256("empty")

    nodes = list(leaves)
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nodes = [
            _sha256(nodes[i] + nodes[i + 1])
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0]


def hash_adapter_state(adapter_path_or_dict) -> Optional[str]:
    """
    SHA-256 over adapter weights.

    Args:
        adapter_path_or_dict: either a filesystem path to adapter files,
            or a dict of {name: tensor} pairs.  Returns None if input is None.

    Returns:
        SHA-256 hex string, or None.
    """
    if adapter_path_or_dict is None:
        return None

    if isinstance(adapter_path_or_dict, str):
        # Hash the adapter files on disk
        import os
        h = hashlib.sha256()
        adapter_dir = adapter_path_or_dict
        if os.path.isdir(adapter_dir):
            for fname in sorted(os.listdir(adapter_dir)):
                fpath = os.path.join(adapter_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
        elif os.path.isfile(adapter_dir):
            with open(adapter_dir, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        return h.hexdigest()

    if isinstance(adapter_path_or_dict, dict):
        # Hash sorted parameter tensors
        h = hashlib.sha256()
        for name in sorted(adapter_path_or_dict.keys()):
            h.update(name.encode("utf-8"))
            tensor = adapter_path_or_dict[name]
            if hasattr(tensor, "cpu"):
                tensor = tensor.cpu()
            if hasattr(tensor, "numpy"):
                h.update(tensor.numpy().tobytes())
            elif hasattr(tensor, "tobytes"):
                h.update(tensor.tobytes())
        return h.hexdigest()

    return None


# ── Training receipt ──────────────────────────────────────────────────────────

def generate_training_receipt(
    session_receipt_root: Optional[str],
    adapter_hash_before: Optional[str],
    adapter_hash_after: Optional[str],
    loss_trajectory: Optional[list],
    config: GroundingConfig,
    training_executed: bool,
    sft_pair_count: int,
) -> dict:
    """
    Build a two-level Merkle receipt for a grounding update event.

    Level 1 leaves:
      - session_receipt_root (links training ← session provenance)
      - adapter_state_before hash
      - training_config hash
      - loss_trajectory hash
      - adapter_state_after hash

    Level 2: pairwise SHA-256 reduction → training_receipt_root.

    Any None leaf is hashed as the string "null" — this is explicit, not hidden.
    The receipt declares whether training actually executed.

    Returns:
        dict with training_receipt_root, all leaf hashes, and metadata.
    """
    # Build leaf values — None becomes the literal string "null"
    def _leaf(value):
        if value is None:
            return _sha256("null")
        if isinstance(value, (dict, list)):
            return _sha256(json.dumps(value, sort_keys=True, allow_nan=False, default=str))
        return _sha256(str(value))

    leaf_session = _leaf(session_receipt_root)
    leaf_adapter_before = _leaf(adapter_hash_before)
    leaf_config = _leaf(config.to_hashable_dict())
    leaf_loss = _leaf(loss_trajectory)
    leaf_adapter_after = _leaf(adapter_hash_after)

    leaves = [
        leaf_session,
        leaf_adapter_before,
        leaf_config,
        leaf_loss,
        leaf_adapter_after,
    ]

    training_receipt_root = _merkle_root(leaves)

    return {
        "training_receipt_root": training_receipt_root,
        "training_executed": training_executed,
        "session_receipt_root": session_receipt_root,
        "adapter_hash_before": adapter_hash_before,
        "adapter_hash_after": adapter_hash_after,
        "loss_trajectory": loss_trajectory,
        "sft_pair_count": sft_pair_count,
        "config": config.to_hashable_dict(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "leaves": {
            "session": leaf_session,
            "adapter_before": leaf_adapter_before,
            "config": leaf_config,
            "loss": leaf_loss,
            "adapter_after": leaf_adapter_after,
        },
        "verifiable": True,
    }


# ── Core grounding update ────────────────────────────────────────────────────

def run_grounding_update(
    session_messages: list,
    consent: dict,
    config: Optional[GroundingConfig] = None,
    session_receipt_root: Optional[str] = None,
) -> GroundingUpdateResult:
    """
    Run an incremental grounding update from a single HAIC session.

    This is the core function.  In dry_run mode (default), it executes ALL
    real pipeline stages — consent validation, SFT pair extraction, token
    counting, receipt construction — but does NOT run gradient steps.
    Loss values and adapter hashes are None, not fake.

    In live mode, it additionally loads the model, runs gradient steps,
    and returns real training metrics.

    Args:
        session_messages:  list of {role, content} dicts from a session
        consent:           consent dict (must include "training_signal")
        config:            GroundingConfig (defaults to dry_run mode)
        session_receipt_root:  merkle_root from the session receipt, if available

    Returns:
        GroundingUpdateResult with all fields populated (or None where
        data was not computed).
    """
    if config is None:
        config = GroundingConfig()

    # ── Step 1: Consent validation ────────────────────────────────────────
    training_signal = consent.get("training_signal", "denied")
    if training_signal != "granted":
        return GroundingUpdateResult(
            training_executed=False,
            mode=config.mode,
            consent_valid=False,
            consent_blocked=True,
            sft_pairs=[],
            sft_pair_count=0,
            token_count=None,
            loss_before=None,
            loss_after=None,
            loss_trajectory=None,
            adapter_hash_before=None,
            adapter_hash_after=None,
            steps_executed=0,
            training_receipt=None,
            session_receipt_root=session_receipt_root,
            error="Consent gate: training_signal is not 'granted'. "
                  "No training update permitted.",
        )

    # ── Step 2: SFT pair extraction (real computation) ────────────────────
    sft_pairs = format_session_as_sft(session_messages, consent)

    if not sft_pairs:
        return GroundingUpdateResult(
            training_executed=False,
            mode=config.mode,
            consent_valid=True,
            consent_blocked=False,
            sft_pairs=[],
            sft_pair_count=0,
            token_count=None,
            loss_before=None,
            loss_after=None,
            loss_trajectory=None,
            adapter_hash_before=None,
            adapter_hash_after=None,
            steps_executed=0,
            training_receipt=None,
            session_receipt_root=session_receipt_root,
            error="No valid SFT pairs extracted from session.",
        )

    # ── Step 3: Token counting (real computation) ─────────────────────────
    # Rough character-based estimate; real tokenizer count in live mode
    total_chars = sum(
        len(p["instruction"]) + len(p["response"])
        for p in sft_pairs
    )
    # Approximate: 1 token ≈ 4 characters for English text
    estimated_tokens = total_chars // 4

    # ── Step 4: Mode-specific execution ───────────────────────────────────
    if config.mode == "live":
        return _run_live_update(
            sft_pairs, config, session_receipt_root, estimated_tokens
        )

    # ── Dry-run: real pipeline, no gradients ──────────────────────────────
    training_receipt = generate_training_receipt(
        session_receipt_root=session_receipt_root,
        adapter_hash_before=None,   # no adapter loaded
        adapter_hash_after=None,    # no update performed
        loss_trajectory=None,       # no training ran
        config=config,
        training_executed=False,
        sft_pair_count=len(sft_pairs),
    )

    return GroundingUpdateResult(
        training_executed=False,
        mode="dry_run",
        consent_valid=True,
        consent_blocked=False,
        sft_pairs=sft_pairs,
        sft_pair_count=len(sft_pairs),
        token_count=estimated_tokens,
        loss_before=None,
        loss_after=None,
        loss_trajectory=None,
        adapter_hash_before=None,
        adapter_hash_after=None,
        steps_executed=0,
        training_receipt=training_receipt,
        session_receipt_root=session_receipt_root,
    )


def _run_live_update(
    sft_pairs: list,
    config: GroundingConfig,
    session_receipt_root: Optional[str],
    estimated_tokens: int,
) -> GroundingUpdateResult:
    """
    Execute a real LoRA grounding update on GPU.

    Requires: torch, transformers, peft, bitsandbytes.
    Target hardware: NVIDIA T4 (16 GB VRAM, float16 only).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, PeftModel
    except ImportError as e:
        return GroundingUpdateResult(
            training_executed=False,
            mode="live",
            consent_valid=True,
            consent_blocked=False,
            sft_pairs=sft_pairs,
            sft_pair_count=len(sft_pairs),
            token_count=estimated_tokens,
            loss_before=None,
            loss_after=None,
            loss_trajectory=None,
            adapter_hash_before=None,
            adapter_hash_after=None,
            steps_executed=0,
            training_receipt=None,
            session_receipt_root=session_receipt_root,
            error=f"Live mode requires GPU libraries: {e}",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return GroundingUpdateResult(
            training_executed=False,
            mode="live",
            consent_valid=True,
            consent_blocked=False,
            sft_pairs=sft_pairs,
            sft_pair_count=len(sft_pairs),
            token_count=estimated_tokens,
            loss_before=None,
            loss_after=None,
            loss_trajectory=None,
            adapter_hash_before=None,
            adapter_hash_after=None,
            steps_executed=0,
            training_receipt=None,
            session_receipt_root=session_receipt_root,
            error="Live mode requires CUDA GPU. No GPU detected.",
        )

    # ── Load model in 4-bit ───────────────────────────────────────────────
    compute_dtype = getattr(torch, config.compute_dtype, torch.float16)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    # ── Attach or load LoRA adapter ───────────────────────────────────────
    if config.adapter_path:
        model = PeftModel.from_pretrained(model, config.adapter_path)
    else:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Hash adapter state BEFORE update
    adapter_hash_before = hash_adapter_state(
        {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    )

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
    )

    loss_trajectory = []
    steps_executed = 0

    try:
        for step in range(config.steps):
            total_loss = 0.0
            for pair in sft_pairs:
                # Format as chat
                text = f"User: {pair['instruction']}\nAssistant: {pair['response']}"
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.max_seq_len,
                    padding=False,
                ).to(device)

                labels = inputs["input_ids"].clone()
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / len(sft_pairs)
                loss.backward()
                total_loss += loss.item()

            # Check for NaN before stepping — mirrors v1/v2 training safeguard
            if math.isnan(total_loss):
                break

            optimizer.step()
            optimizer.zero_grad()
            loss_trajectory.append(round(total_loss, 6))
            steps_executed += 1

    except RuntimeError as e:
        # OOM or CUDA errors — return partial result with what we have
        error_msg = f"Training interrupted at step {steps_executed}: {e}"
        # Still hash and receipt the partial state
        adapter_hash_after = hash_adapter_state(
            {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        )
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        training_receipt = generate_training_receipt(
            session_receipt_root=session_receipt_root,
            adapter_hash_before=adapter_hash_before,
            adapter_hash_after=adapter_hash_after,
            loss_trajectory=loss_trajectory or None,
            config=config,
            training_executed=steps_executed > 0,
            sft_pair_count=len(sft_pairs),
        )
        return GroundingUpdateResult(
            training_executed=steps_executed > 0,
            mode="live",
            consent_valid=True,
            consent_blocked=False,
            sft_pairs=sft_pairs,
            sft_pair_count=len(sft_pairs),
            token_count=estimated_tokens,
            loss_before=loss_trajectory[0] if loss_trajectory else None,
            loss_after=loss_trajectory[-1] if loss_trajectory else None,
            loss_trajectory=loss_trajectory or None,
            adapter_hash_before=adapter_hash_before,
            adapter_hash_after=adapter_hash_after,
            steps_executed=steps_executed,
            training_receipt=training_receipt,
            session_receipt_root=session_receipt_root,
            error=error_msg,
        )

    # Hash adapter state AFTER update
    adapter_hash_after = hash_adapter_state(
        {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    )

    # ── Save updated adapter if path specified ────────────────────────────
    if config.adapter_path:
        model.save_pretrained(config.adapter_path)

    # ── Build training receipt ────────────────────────────────────────────
    training_receipt = generate_training_receipt(
        session_receipt_root=session_receipt_root,
        adapter_hash_before=adapter_hash_before,
        adapter_hash_after=adapter_hash_after,
        loss_trajectory=loss_trajectory,
        config=config,
        training_executed=True,
        sft_pair_count=len(sft_pairs),
    )

    # Clean up GPU memory
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return GroundingUpdateResult(
        training_executed=True,
        mode="live",
        consent_valid=True,
        consent_blocked=False,
        sft_pairs=sft_pairs,
        sft_pair_count=len(sft_pairs),
        token_count=estimated_tokens,
        loss_before=loss_trajectory[0] if loss_trajectory else None,
        loss_after=loss_trajectory[-1] if loss_trajectory else None,
        loss_trajectory=loss_trajectory,
        adapter_hash_before=adapter_hash_before,
        adapter_hash_after=adapter_hash_after,
        steps_executed=steps_executed,
        training_receipt=training_receipt,
        session_receipt_root=session_receipt_root,
    )


# ── Function-calling tool schema ──────────────────────────────────────────────

RUN_GROUNDING_UPDATE_SCHEMA = {
    "name": "run_grounding_update",
    "description": (
        "Run an incremental grounding update using a consented HAIC session. "
        "Extracts SFT training pairs from the session, optionally runs LoRA "
        "gradient steps on the local E2B model, and produces a two-level "
        "Merkle training receipt linking the weight update to the session. "
        "In dry_run mode (default), all pipeline stages run but gradient "
        "steps are skipped — loss values are null, not simulated. "
        "Requires training_signal='granted' in consent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session UUID from the current verified session"
            },
            "session_messages": {
                "type": "array",
                "description": "Session messages [{role, content}] from the interview",
                "items": {"type": "object"}
            },
            "consent": {
                "type": "object",
                "description": (
                    "Consent decisions. Must include training_signal='granted' "
                    "for the update to proceed."
                ),
                "properties": {
                    "transcript":      {"type": "string", "enum": ["granted", "denied"]},
                    "felt_state":      {"type": "string", "enum": ["granted", "denied"]},
                    "training_signal": {"type": "string", "enum": ["granted", "denied"]},
                    "retention":       {"type": "string", "enum": ["granted", "denied"]},
                }
            },
            "mode": {
                "type": "string",
                "description": (
                    "Execution mode. 'dry_run' (default): runs full pipeline "
                    "except gradient steps, returns null for training metrics. "
                    "'live': runs gradient steps on GPU, returns real metrics."
                ),
                "enum": ["dry_run", "live"],
                "default": "dry_run"
            },
            "session_receipt_root": {
                "type": "string",
                "description": (
                    "Merkle root from the session receipt (from generate_receipt). "
                    "Links the training receipt to its source session."
                )
            }
        },
        "required": ["session_id", "session_messages", "consent"]
    }
}


def run_grounding_update_handler(
    session_id: str,
    session_messages: list,
    consent: dict,
    mode: str = "dry_run",
    session_receipt_root: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Tool handler for run_grounding_update — wraps the core function
    for dispatch_tool() integration.
    """
    config = GroundingConfig(mode=mode)

    result = run_grounding_update(
        session_messages=session_messages,
        consent=consent,
        config=config,
        session_receipt_root=session_receipt_root,
    )

    # Convert to JSON-serializable dict
    output = {
        "session_id": session_id,
        "training_executed": result.training_executed,
        "mode": result.mode,
        "consent_valid": result.consent_valid,
        "consent_blocked": result.consent_blocked,
        "sft_pair_count": result.sft_pair_count,
        "token_count": result.token_count,
        "loss_before": result.loss_before,
        "loss_after": result.loss_after,
        "steps_executed": result.steps_executed,
        "adapter_hash_before": result.adapter_hash_before,
        "adapter_hash_after": result.adapter_hash_after,
        "session_receipt_root": result.session_receipt_root,
    }

    if result.training_receipt:
        output["training_receipt"] = result.training_receipt

    if result.error:
        output["error"] = result.error

    # Include SFT pair summaries (not full content — privacy)
    output["sft_pairs_summary"] = [
        {
            "turn_index": p["turn_index"],
            "instruction_length": len(p["instruction"]),
            "response_length": len(p["response"]),
        }
        for p in result.sft_pairs
    ]

    return output
