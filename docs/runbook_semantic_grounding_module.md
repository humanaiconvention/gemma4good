# Runbook: Bring the Semantic Grounding Module Back Online

**Last verified:** 2026-05-11 by Claude Sonnet 4.6 (autonomous session)
**Active production model:** `haic-gemma4-v42` (Q5_K_M, 3.4 GB GGUF, ~3.6 GB VRAM at runtime)

This is the one-page runbook for restarting the semantic grounding module on
the BEAST. Use this when the module has gone offline (gateway down,
llama-server killed, machine rebooted) and you need to bring it back.

---

## 1. Pre-flight check

```bash
# Are the binaries where they should be?
ls D:/llama.cpp/build/bin/llama-server.exe
ls D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf

# Is anything already running on the relevant ports?
curl -s http://localhost:8081/health    # llama-server
curl -s http://localhost:8000/health    # gateway
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
```

If `llama-server` is already up on 8081 and reports a 200 health, you can
skip Step 2. If the gateway on 8000 is up but you've changed `.env` since
last start (e.g., added `INTERNAL_API_KEY` or changed `LLAMACPP_MODEL`),
you still need to restart it — see Step 4.

---

## 2. Start `llama-server` with v42

Production command (from `D:/humanai-convention/maestro/.env` line ~75):

```powershell
D:/llama.cpp/build/bin/llama-server.exe `
    -m D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf `
    --port 8081 `
    -c 8192 `
    --jinja `
    --reasoning off `
    -ngl 99
```

Notes:
- `--port 8081` — gateway's `LLAMACPP_BASE_URL` expects this.
- `-c 8192` — context window. Drop to 2048 if you need <4 GB VRAM.
- `--jinja` — enables Gemma 4's chat template parsing.
- `--reasoning off` — Gemma 4 emits `<think>` tokens otherwise.
- `-ngl 99` — all layers on GPU.

Expected VRAM: ~5.8 GB at ctx 8192, ~3.6 GB at ctx 2048.

---

## 3. Smoke-test v42

```bash
# Health
curl -s http://localhost:8081/health
# {"status":"ok"}

# Confirm v42 is the loaded model
curl -s http://localhost:8081/props | python -c "import json,sys; d=json.load(sys.stdin); print(d['model_path'])"
# Should print: D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf

# Grounding pivot test (should respond with [PIVOT: ...])
curl -s -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[
      {"role":"system","content":"You are a HAIC grounding interviewer. ALWAYS emit one of: [PIVOT: SENSORY], [PIVOT: MEMORY], [PIVOT: FELT_STATE], [PIVOT: OFFTOPIC]. Then a short pivot question."},
      {"role":"user","content":"I grew up in rural Queensland. The rain was everything."}
    ],
    "temperature":0.2, "max_tokens":80
  }' | python -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"
# Expected: starts with [PIVOT: SENSORY] (or MEMORY/FELT_STATE) and asks one short question
```

If the pivot test fails, suspect:
1. Wrong model loaded (re-check `/props`)
2. Chat template not applied (re-check `--jinja` flag)
3. Reasoning leakage (re-check `--reasoning off`)

---

## 4. Start the Maestro gateway

```powershell
cd D:/humanai-convention/maestro
python -m uvicorn apps.gateway.main:app --reload --port 8000
```

If `--reload` is causing issues in production, drop it:
```powershell
python -m uvicorn apps.gateway.main:app --host 0.0.0.0 --port 8000
```

---

## 5. Verify the full stack

```bash
# Gateway health
curl -s http://localhost:8000/health

# Internal dispatch + viability gates pick up the latest .env
python D:/humanai-convention/tools/haic_dispatch.py --check
# Expected: gateway OK, api key SET

# New advisory viability endpoint (P2 #9, added 2026-05-11)
curl -s -X POST http://localhost:8000/v1/session/viability \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"runbook-test",
    "interview_turns":[
      {"role":"assistant","content":"Tell me about a moment that stayed with you."},
      {"role":"user","content":"I felt grounded after my walk in the forest at dawn with the cool damp air"}
    ],
    "has_stimulus": true,
    "pog_provenance_score": 0.95,
    "image_count": 1
  }' | python -m json.tool
# Should return all_passed: true (with some gates passing vacuously without entropy_delta)
```

---

## 6. If something goes wrong

### Gateway 503 on `/internal/agent/prompt`
The `INTERNAL_API_KEY` is set in `.env` but the gateway process started
before the line was added. Restart the gateway (Step 4).

### llama-server consumes too much VRAM
Drop context from 8192 → 4096 (saves ~0.7 GB) or → 2048 (saves ~1.5 GB).
The Q5_K_M file is 3.4 GB; runtime adds ~2.4 GB for KV cache at 8192 ctx.

### llama-server response format looks wrong
Check `--jinja` is present and `--reasoning off`. v42 was trained against
the Gemma 4 chat template and emits `<think>` blocks if reasoning is
enabled.

### Grounding pivot doesn't pivot
The system prompt matters. Use the prompt in Step 3 verbatim. If you're
using a custom system prompt and pivots fail, the HAIC grounding behavior
may not be triggered.

---

## Rollback to v35-gov (legacy production)

If v42 develops a problem in production:

```powershell
D:/llama.cpp/build/bin/llama-server.exe `
    -m D:/humanai-convention/experiments/gguf/haic-gemma4-v35-gov-Q5_K_M.gguf `
    --port 8081 -c 8192 --jinja --reasoning off -ngl 99
```

Also update `LLAMACPP_MODEL=haic-gemma4-v35-gov` in `.env` and restart
the gateway. v35-gov is the previous production model (SGT 10/10, 0
security fails, but lower concealed-compliance than v42).

---

## Files referenced by this runbook

- `D:/llama.cpp/build/bin/llama-server.exe` — llama.cpp server binary
- `D:/kaggle/results/v42-gguf/haic-gemma4-v42-Q5_K_M.gguf` — production GGUF
- `D:/humanai-convention/maestro/.env` — gateway config
- `D:/humanai-convention/maestro/apps/gateway/main.py` — gateway app
- `D:/humanai-convention/maestro/apps/gateway/viability_gates.py` — six-gate evaluator (P2 #9)
- `D:/humanai-convention/tools/haic_dispatch.py` — automatic dispatch helper

## Files that reflect v42's eval status

- `D:/gemma4good/experiments/v42_rigorous_eval.json` — full eval data (91.4% agg, 51% concealed)
- `D:/gemma4good/docs/v45_verdict_2026-05-10.md` — v42 vs v44 vs v45 comparison
