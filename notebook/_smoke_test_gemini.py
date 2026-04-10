"""
Smoke test the Gemini API path of haic_gemma4_governance.ipynb without
needing a GPU. Loads the API key from .env, makes one round-trip via
google.genai, and exercises the function-call regex parser end-to-end.

Does NOT load Gemma 4 locally — that's deferred to Kaggle.
"""

from __future__ import annotations

import json
import os
import sys
import re
from pathlib import Path

# Read .env
env_path = Path(__file__).resolve().parents[1] / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

api_key = os.environ.get("GOOGLE_API_KEY")
assert api_key, "GOOGLE_API_KEY not found in .env"
print(f"GOOGLE_API_KEY loaded ({len(api_key)} chars, ends ...{api_key[-4:]})")

# Mimic cell 5's gemini path exactly
print("\n[1/4] Importing google.genai...")
from google import genai as google_genai
print("  ok")

print("[2/4] Creating client...")
# Try multiple models in order; the first one with quota wins.
# 2.0-flash sometimes has limit:0 on free-tier projects; 1.5-flash usually works.
CANDIDATE_MODELS = [
    os.environ.get("GEMINI_MODEL_NAME"),  # explicit override wins if set
    # Gemma 4 first — this is the notebook's actual target model and IS
    # callable via the Gemini API on this key (verified 2026-04-07).
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    # Gemma 3 fallbacks
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    # Gemini fallbacks (some have limit:0 on free-tier projects)
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]
CANDIDATE_MODELS = [m for m in CANDIDATE_MODELS if m]
client = google_genai.Client(api_key=api_key)
print(f"  ok (will try models: {CANDIDATE_MODELS})")

print("[3/4] Sending one tool-call test prompt...")
test_prompt = """<SYSTEM>
You are a governance agent that emits structured function calls.
When you decide to call a tool, emit it on its own line in this exact format:

  <function_call>
  {"name": "assess_wellbeing_domain", "arguments": {"domain": "health", "context": "rural clinic patient triage", "evidence": "2 nurses, 1 doctor, 50 patients/day"}}
  </function_call>

Available tools: assess_wellbeing_domain, verify_consent_chain, run_prism_analysis, generate_alignment_receipt.

Reason briefly in plain text first, then emit one function call.
</SYSTEM>

<USER>
Scenario: A rural health clinic in sub-Saharan Africa wants to deploy an AI triage system. Limited staff (1 doctor, 2 nurses). Begin governance evaluation by assessing wellbeing impact on the patient population.
</USER>

<ASSISTANT>
"""

text = None
used_model = None
last_error = None
for model_name in CANDIDATE_MODELS:
    print(f"  trying {model_name}...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=test_prompt,
            config={
                "max_output_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
            },
        )
        text = response.text or ""
        used_model = model_name
        print(f"  ok via {model_name} ({len(text)} chars returned)")
        print("\n  --- model output (first 600 chars) ---")
        print("  " + text[:600].replace("\n", "\n  "))
        print("  --- end ---")
        break
    except Exception as e:
        msg = str(e)
        last_error = (type(e).__name__, msg)
        # Quota / rate-limit / not-available → try next model
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg or "NOT_FOUND" in msg or "404" in msg:
            print(f"  skip ({type(e).__name__}: short-circuit, trying next)")
            continue
        # Other errors are fatal
        print(f"  FAIL: {type(e).__name__}: {msg}")
        sys.exit(1)

if text is None:
    print(f"\n  FAIL: all candidate models exhausted. last error: {last_error}")
    sys.exit(1)

print("\n[4/4] Parsing tool calls from output...")
# Mimic the cell 11 regex parser
def parse_function_calls(text):
    pattern = re.compile(
        r"<function_call>\s*(\{.*?\})\s*</function_call>",
        re.DOTALL,
    )
    calls = []
    for match in pattern.finditer(text):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError as e:
            calls.append({"_parse_error": str(e), "raw": match.group(1)[:200]})
    return calls

calls = parse_function_calls(text)
print(f"  Parsed {len(calls)} tool call(s)")
for i, c in enumerate(calls):
    print(f"  [{i}] {json.dumps(c, indent=2)[:300]}")

if not calls:
    print("\n  WARNING: model output contained no <function_call> blocks. The")
    print("  notebook's regex parser may need a different prompt format for Gemini")
    print("  vs Gemma. Check the system prompt format in cell 11.")
    sys.exit(2)

print("\n[OK] Gemini API path round-trip + tool-call parsing succeeded.")
