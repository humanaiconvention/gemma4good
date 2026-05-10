"""Tests for experiments/run_rigorous_sgt_gguf.py.

Pure-Python — uses a mock Llama class to exercise the backend wiring
without loading an actual GGUF (which requires llama-cpp-python and a
several-GB file). The integration test (does v39's GGUF actually load
and respond?) is a separate manual smoke test once the artifact exists.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Backend contract ────────────────────────────────────────────────────────


class TestMakeGGUFBackend:

    def _mock_llm(self, response_text: str = "[PIVOT: DEEPEN] Tell me about a moment."):
        """Build a mock Llama instance that returns canned responses."""
        m = MagicMock()
        m.create_chat_completion.return_value = {
            "choices": [{"message": {"content": response_text}}]
        }
        return m

    def test_backend_returns_callable(self):
        with patch("llama_cpp.Llama", autospec=False) as mock_llama_cls:
            mock_llama_cls.return_value = self._mock_llm()
            from experiments.run_rigorous_sgt_gguf import make_gguf_backend
            generate, llm = make_gguf_backend("fake.gguf")
            assert callable(generate)
            assert llm is not None

    def test_backend_passes_seed_to_llama(self):
        with patch("llama_cpp.Llama", autospec=False) as mock_llama_cls:
            mock_llm = self._mock_llm()
            mock_llama_cls.return_value = mock_llm
            from experiments.run_rigorous_sgt_gguf import make_gguf_backend
            generate, _ = make_gguf_backend("fake.gguf")
            generate("hello", seed=12345, sample=True)
            kwargs = mock_llm.create_chat_completion.call_args.kwargs
            assert kwargs["seed"] == 12345

    def test_backend_greedy_sets_temperature_zero(self):
        with patch("llama_cpp.Llama", autospec=False) as mock_llama_cls:
            mock_llm = self._mock_llm()
            mock_llama_cls.return_value = mock_llm
            from experiments.run_rigorous_sgt_gguf import make_gguf_backend
            generate, _ = make_gguf_backend("fake.gguf", temperature=0.7)
            generate("hello", seed=42, sample=False)
            kwargs = mock_llm.create_chat_completion.call_args.kwargs
            assert kwargs["temperature"] == 0.0
            assert kwargs["top_p"] == 1.0

    def test_backend_sampling_uses_provided_temperature(self):
        with patch("llama_cpp.Llama", autospec=False) as mock_llama_cls:
            mock_llm = self._mock_llm()
            mock_llama_cls.return_value = mock_llm
            from experiments.run_rigorous_sgt_gguf import make_gguf_backend
            generate, _ = make_gguf_backend(
                "fake.gguf", temperature=0.7, top_p=0.9
            )
            generate("hello", seed=42, sample=True)
            kwargs = mock_llm.create_chat_completion.call_args.kwargs
            assert kwargs["temperature"] == 0.7
            assert kwargs["top_p"] == 0.9

    def test_backend_returns_assistant_content(self):
        with patch("llama_cpp.Llama", autospec=False) as mock_llama_cls:
            mock_llama_cls.return_value = self._mock_llm("hello world")
            from experiments.run_rigorous_sgt_gguf import make_gguf_backend
            generate, _ = make_gguf_backend("fake.gguf")
            out = generate("any prompt", seed=42, sample=False)
            assert out == "hello world"

    def test_backend_includes_system_prompt(self):
        with patch("llama_cpp.Llama", autospec=False) as mock_llama_cls:
            mock_llm = self._mock_llm()
            mock_llama_cls.return_value = mock_llm
            from experiments.run_rigorous_sgt_gguf import make_gguf_backend
            generate, _ = make_gguf_backend(
                "fake.gguf", system_prompt="custom system prompt"
            )
            generate("user msg", seed=42, sample=False)
            messages = mock_llm.create_chat_completion.call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "custom system prompt"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "user msg"


# ── End-to-end with mocked GGUF and the real run_sgt ────────────────────────


class TestRunSgtAgainstMockedGGUF:

    def test_run_sgt_consumes_gguf_backend(self):
        """The whole point: a mocked GGUF backend should plug into run_sgt
        the same way the HF backend does, and produce a valid SGT result."""
        with patch("llama_cpp.Llama", autospec=False) as mock_llama_cls:
            mock_llm = MagicMock()
            mock_llm.create_chat_completion.return_value = {
                "choices": [{"message": {
                    "content": "[PIVOT: DEEPEN] Tell me about a moment when..."
                }}]
            }
            mock_llama_cls.return_value = mock_llm

            from experiments.run_rigorous_sgt_gguf import make_gguf_backend
            from experiments.sgt_harness import run_sgt, DEFAULT_SCENARIOS

            generate, _ = make_gguf_backend("fake.gguf")
            result = run_sgt(generate, n_samples=2, seed=42, model_id="test")
            assert "deterministic" in result
            assert "sampling" in result
            assert result["deterministic"]["model_id"] == "test"
            # All grounding scenarios get [PIVOT: DEEPEN] response → all PASS
            assert result["deterministic"]["grounding_passes"] == \
                   sum(1 for s in DEFAULT_SCENARIOS if s.kind == "grounding")
