"""
v42 Boundary Guard v7 — H26 candidate (multi-language rule extension on top of v6)

Per the H26 precommit at ``docs/h26_precommit_hypothesis_2026-05-17.md``:

> A guard variant that retains all guard-v6 behavior (16 English rules,
> Unicode normalization, leet-fold pre-pass, per-message scan, system-
> role rejection) AND additionally runs eleven new multi-language rules
> against a pre-homoglyph-fold normalized surface (NFKC + zero-width
> strip only, no Cyrillic→Latin fold) — one rule per language, each
> targeting the joint occurrence of the native word for "capital" with
> a country name from a curated list of 15 major countries.

guard-v7 = guard-v6 + ``_normalize_for_match_pre_fold`` +
``MULTILANG_RULES`` (11 rules: ja, ko, zh, ru, es, fr, de, pt, ar, hi, he).

The H26 hypothesis tests whether L-09 (native-language attack bypass,
discovered in H25) can be closed by this narrowly-scoped extension
without introducing benign multilingual false positives.

Rule shape per language:
    (country_name_in_lang){within 30 chars}(capital_word_in_lang)
    OR
    (capital_word_in_lang){within 30 chars}(country_name_in_lang)

All multi-language rules emit guard_class DIRECT_INJECT and the same
deterministic refusal phrasing as DI-01.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

# Re-use the v1 rule set, decision dataclass, hash helper, completion
# builder, and pass-through allowlist verbatim.
from tools.v42_boundary_guard import (
    RULES,
    Rule,
    GuardDecision,
    _build_completion,
    _sha3,
    _PASSTHROUGH_GET_ALLOWLIST,
)

# Re-use the v2 normalization function (post-fold surface for English/leet).
from tools.v42_boundary_guard_v2 import _normalize_for_match

# Re-use v4's per-message user extraction.
from tools.v42_boundary_guard_v4 import _extract_user_texts

# Re-use v5's system-role injection detector.
from tools.v42_boundary_guard_v5 import _detect_system_role_injection

# Re-use v6's leet fold.
from tools.v42_boundary_guard_v6 import _leet_fold


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [guard-v7] %(levelname)s %(message)s",
)
log = logging.getLogger("v42_guard_v7")


# ── Pre-fold normalization (Surface 3: for multi-language matching) ──────────


def _normalize_for_match_pre_fold(text: str) -> str:
    """NFKC + zero-width / control-char strip ONLY — no homoglyph fold.

    The v6 ``_normalize_for_match`` applies a Cyrillic → Latin homoglyph
    fold (с → c, о → o, а → a, ...) so that English regex rules can
    catch Cyrillic-homoglyph attack variants. That fold destroys
    legitimate Cyrillic words like "столица" (Russian for "capital").

    For multi-language matching we need the original script preserved,
    so this surface stops at NFKC + Cf/Cc stripping. It's intentionally
    a strict subset of the v6 pipeline.

    NFKC handles fullwidth forms (e.g. "ｆｒａｎｃｅ" → "france") and
    presentation forms in CJK/Devanagari/Arabic. Zero-width strip
    handles ZWSP/ZWJ/RLO injected as an obfuscation.
    """
    if not text:
        return text
    nfkc = unicodedata.normalize("NFKC", text)
    stripped_chars = []
    for ch in nfkc:
        cat = unicodedata.category(ch)
        if cat in ("Cf",) or (cat == "Cc" and ch not in "\n\r\t"):
            continue
        stripped_chars.append(ch)
    return "".join(stripped_chars)


# ── Multi-language rule set ──────────────────────────────────────────────────


_GENERAL_BOUNDARY = (
    "I can't engage with that request. "
    "What lived experience with AI should we explore instead?"
)


def _mlr(rule_id: str, pattern: str) -> Rule:
    """Build a multi-language DI rule with consistent flags + response.

    All multilang rules use IGNORECASE (matters for Latin-script langs;
    no-op for CJK/Arabic/Devanagari/Hebrew which have no case) and
    DOTALL so the `.{0,30}` gap can span newlines if a payload
    embeds them.
    """
    return Rule(
        rule_id=rule_id,
        pattern=re.compile(pattern, re.IGNORECASE | re.DOTALL),
        guard_class="DIRECT_INJECT",
        response_text=_GENERAL_BOUNDARY,
    )


# Country-name lists per language.  Each language's pattern is the joint
# occurrence of the capital-word with a country from the language's
# native-script country list, in either order, within 30 chars.

# Japanese: 首都 = capital. Native country names in katakana/kanji.
_JA_COUNTRIES = (
    "フランス|ドイツ|スペイン|イタリア|日本|中国|ロシア|ブラジル|"
    "インド|アメリカ|イギリス|英国|米国|カナダ|オーストラリア|"
    "メキシコ|アルゼンチン"
)
_RULE_ML_DI_JA = _mlr(
    "ML-DI-ja",
    rf"(?:{_JA_COUNTRIES}).{{0,30}}首都|首都.{{0,30}}(?:{_JA_COUNTRIES})",
)

# Korean: 수도 = capital.
_KO_COUNTRIES = (
    "프랑스|독일|스페인|이탈리아|일본|중국|러시아|브라질|"
    "인도|미국|영국|캐나다|호주|오스트레일리아|멕시코|아르헨티나"
)
_RULE_ML_DI_KO = _mlr(
    "ML-DI-ko",
    rf"(?:{_KO_COUNTRIES}).{{0,30}}수도|수도.{{0,30}}(?:{_KO_COUNTRIES})",
)

# Chinese Simplified: 首都 / 首府 = capital.
_ZH_COUNTRIES = (
    "法国|德国|西班牙|意大利|日本|中国|俄罗斯|俄国|巴西|"
    "印度|美国|英国|加拿大|澳大利亚|墨西哥|阿根廷"
)
_RULE_ML_DI_ZH = _mlr(
    "ML-DI-zh",
    rf"(?:{_ZH_COUNTRIES}).{{0,30}}(?:首都|首府)|"
    rf"(?:首都|首府).{{0,30}}(?:{_ZH_COUNTRIES})",
)

# Russian: столиц = capital stem (covers столица/столицы/столице/столицу).
# Country stems with case-ending tolerance using character classes.
_RU_COUNTRIES = (
    r"франци[ияею]|германи[ияею]|испани[ияею]|итали[ияею]|"
    r"япони[ияею]|кита[яйюе]|росси[ияею]|бразили[ияею]|"
    r"инди[ияею]|америк[аиуеой]|англи[ияею]|британи[ияею]|"
    r"канад[аыуеой]|австрали[ияею]|мексик[аиуеой]|аргентин[аыуеой]|"
    r"сша"
)
_RULE_ML_DI_RU = _mlr(
    "ML-DI-ru",
    rf"(?:{_RU_COUNTRIES}).{{0,30}}столиц|столиц.{{0,30}}(?:{_RU_COUNTRIES})",
)

# Spanish: capital de/del + country.
_ES_COUNTRIES = (
    r"francia|alemania|españa|italia|japón|china|rusia|brasil|"
    r"india|estados\s+unidos|reino\s+unido|inglaterra|canadá|"
    r"australia|méxico|mexico|argentina"
)
_RULE_ML_DI_ES = _mlr(
    "ML-DI-es",
    rf"\bcapital\b.{{0,20}}\b(?:de|del)\b\s+(?:{_ES_COUNTRIES})",
)

# French: capitale de/du/d' + country.
_FR_COUNTRIES = (
    r"france|allemagne|espagne|italie|japon|chine|russie|brésil|"
    r"inde|états[\s\-]?unis|etats[\s\-]?unis|royaume[\s\-]?uni|"
    r"angleterre|canada|australie|mexique|argentine"
)
_RULE_ML_DI_FR = _mlr(
    "ML-DI-fr",
    rf"\bcapitale\b.{{0,20}}\b(?:de|du|d'|de\s+la)\b\s*(?:{_FR_COUNTRIES})",
)

# German: Hauptstadt von/der/des + country.
_DE_COUNTRIES = (
    r"frankreich|deutschland|spanien|italien|japan|china|russland|"
    r"brasilien|indien|usa|vereinigte\s+staaten|großbritannien|"
    r"grossbritannien|england|kanada|australien|mexiko|argentinien"
)
_RULE_ML_DI_DE = _mlr(
    "ML-DI-de",
    rf"\bhauptstadt\b.{{0,20}}\b(?:von|der|des|de)\b\s*(?:{_DE_COUNTRIES})",
)

# Portuguese: capital de/do/da + country.
_PT_COUNTRIES = (
    r"frança|alemanha|espanha|itália|japão|china|rússia|brasil|"
    r"índia|estados\s+unidos|reino\s+unido|inglaterra|canadá|"
    r"austrália|méxico|mexico|argentina"
)
_RULE_ML_DI_PT = _mlr(
    "ML-DI-pt",
    rf"\bcapital\b.{{0,20}}\b(?:de|do|da)\b\s+(?:{_PT_COUNTRIES})",
)

# Arabic: عاصمة = capital. Country names in Arabic script.
_AR_COUNTRIES = (
    "فرنسا|ألمانيا|إسبانيا|إيطاليا|اليابان|الصين|روسيا|"
    "البرازيل|الهند|الولايات\\s+المتحدة|أمريكا|المملكة\\s+المتحدة|"
    "إنجلترا|كندا|أستراليا|المكسيك|الأرجنتين"
)
_RULE_ML_DI_AR = _mlr(
    "ML-DI-ar",
    rf"عاصمة.{{0,30}}(?:{_AR_COUNTRIES})|(?:{_AR_COUNTRIES}).{{0,30}}عاصمة",
)

# Hindi: राजधानी = capital. Country names in Devanagari.
_HI_COUNTRIES = (
    "फ्रांस|जर्मनी|स्पेन|इटली|जापान|चीन|रूस|ब्राज़ील|ब्राज़िल|"
    "भारत|अमेरिका|ब्रिटेन|कनाडा|ऑस्ट्रेलिया|मेक्सिको|अर्जेंटीना"
)
_RULE_ML_DI_HI = _mlr(
    "ML-DI-hi",
    rf"राजधानी.{{0,30}}(?:{_HI_COUNTRIES})|(?:{_HI_COUNTRIES}).{{0,30}}राजधानी",
)

# Hebrew: בירה = capital. Country names in Hebrew script.
_HE_COUNTRIES = (
    "צרפת|גרמניה|ספרד|איטליה|יפן|סין|רוסיה|ברזיל|"
    "הודו|ארצות\\s+הברית|בריטניה|אנגליה|קנדה|אוסטרליה|"
    "מקסיקו|ארגנטינה"
)
_RULE_ML_DI_HE = _mlr(
    "ML-DI-he",
    rf"בירה.{{0,30}}(?:{_HE_COUNTRIES})|(?:{_HE_COUNTRIES}).{{0,30}}בירה",
)


MULTILANG_RULES: list[Rule] = [
    _RULE_ML_DI_JA,
    _RULE_ML_DI_KO,
    _RULE_ML_DI_ZH,
    _RULE_ML_DI_RU,
    _RULE_ML_DI_ES,
    _RULE_ML_DI_FR,
    _RULE_ML_DI_DE,
    _RULE_ML_DI_PT,
    _RULE_ML_DI_AR,
    _RULE_ML_DI_HI,
    _RULE_ML_DI_HE,
]


# ── Guard logic ──────────────────────────────────────────────────────────────


def apply_guard_v7(user_texts: list[str]) -> tuple[GuardDecision, list[int]]:
    """Evaluate every user message under guard-v7's three-surface pipeline.

    Surfaces per message:
      1. ``_normalize_for_match(text)`` — NFKC + ZW strip + homoglyph fold.
         Matched against the 16 English RULES.
      2. ``_leet_fold(surface_1)`` — digit-to-letter substitution.
         Matched against the 16 English RULES (skipped if no digits).
      3. ``_normalize_for_match_pre_fold(text)`` — NFKC + ZW strip ONLY,
         preserves Cyrillic / CJK / Arabic / Devanagari / Hebrew.
         Matched against the 11 MULTILANG_RULES.

    First-match-wins across messages and surfaces; matched_rule_ids
    lists every distinct rule that fired anywhere.
    """
    raw_joined = "\x1f".join(user_texts)
    request_hash = _sha3(raw_joined)

    matched_rules: list[Rule] = []
    matched_indices: list[int] = []
    seen_rule_ids: set[str] = set()

    def _scan(surface: str, rule_list: list[Rule], idx: int) -> None:
        for rule in rule_list:
            if rule.pattern.search(surface):
                if rule.rule_id not in seen_rule_ids:
                    matched_rules.append(rule)
                    seen_rule_ids.add(rule.rule_id)
                if idx not in matched_indices:
                    matched_indices.append(idx)

    for idx, text in enumerate(user_texts):
        norm = _normalize_for_match(text)
        folded = _leet_fold(norm)
        prefold = _normalize_for_match_pre_fold(text)

        # English rules over the v6 surfaces (post-fold + leet).
        _scan(norm, RULES, idx)
        if folded != norm:
            _scan(folded, RULES, idx)

        # Multi-language rules over the pre-fold surface only.
        _scan(prefold, MULTILANG_RULES, idx)

    if not matched_rules:
        return (
            GuardDecision(
                guard_triggered=False,
                guard_class=None,
                matched_rule_ids=[],
                response_source="model",
                response_text=None,
                request_hash=request_hash,
            ),
            [],
        )

    primary = matched_rules[0]
    return (
        GuardDecision(
            guard_triggered=True,
            guard_class=primary.guard_class,
            matched_rule_ids=[r.rule_id for r in matched_rules],
            response_source="guard",
            response_text=primary.response_text,
            request_hash=request_hash,
        ),
        matched_indices,
    )


# ── FastAPI app ──────────────────────────────────────────────────────────────


DEFAULT_UPSTREAM = "http://127.0.0.1:8081"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    upstream = getattr(app.state, "upstream", DEFAULT_UPSTREAM)
    timeout = httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=10.0)
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=8)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        app.state.http_client = client
        app.state.upstream = upstream
        log.info(
            "guard-v7 ready: upstream=%s english_rules=%d multilang_rules=%d",
            upstream, len(RULES), len(MULTILANG_RULES),
        )
        yield
        log.info("guard-v7 shutdown")


app = FastAPI(title="v42 Boundary Guard v7 (H26)", version="0.7.0", lifespan=lifespan)
app.state.upstream = DEFAULT_UPSTREAM


def _client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def _upstream_url(request: Request) -> str:
    return request.app.state.upstream


def _bad_gateway(detail: str, exc: Exception | None = None) -> Response:
    log.warning("upstream_error %s: %s", detail, exc if exc else "")
    payload = {"error": {"type": "bad_gateway", "message": detail}}
    return Response(
        content=json.dumps(payload),
        status_code=502,
        media_type="application/json",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "v42_boundary_guard_v7"}


@app.get("/guard/rules")
async def list_rules() -> list[dict[str, str]]:
    return [
        {
            "rule_id": r.rule_id,
            "guard_class": r.guard_class,
            "pattern": r.pattern.pattern,
            "surface": "english+leet" if r in RULES else "multilang",
        }
        for r in (list(RULES) + list(MULTILANG_RULES))
    ]


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    # H22-D2a (inherited): reject client-supplied role:system at any non-first position.
    if _detect_system_role_injection(body):
        log.info("system_role_in_history: rejecting request (HTTP 400)")
        raise HTTPException(
            status_code=400,
            detail="system_role_in_history: only the first message may be role:system",
        )

    user_texts = _extract_user_texts(body)
    decision, matched_indices = apply_guard_v7(user_texts)

    log.info(
        "guard_decision request_hash=%s triggered=%s class=%s rules=%s msgs=%s source=%s",
        decision.request_hash[:16],
        decision.guard_triggered,
        decision.guard_class,
        decision.matched_rule_ids,
        matched_indices,
        decision.response_source,
    )

    if decision.guard_triggered:
        model_id = body.get("model", "haic-gemma4-v42-guard-v7")
        stream = body.get("stream", False)
        payload = _build_completion(decision, model_id, stream=stream)
        payload["guard_metadata"]["matched_message_indices"] = matched_indices

        if stream:
            chunk = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"

            async def gen():
                yield chunk.encode()

            return StreamingResponse(gen(), media_type="text/event-stream")
        return Response(
            content=json.dumps(payload),
            media_type="application/json",
        )

    client = _client(request)
    upstream = _upstream_url(request)
    try:
        upstream_resp = await client.post(
            f"{upstream}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
        )
    except httpx.TimeoutException as exc:
        return _bad_gateway("upstream timeout", exc)
    except httpx.HTTPError as exc:
        return _bad_gateway("upstream unreachable", exc)

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )


@app.api_route("/{path:path}", methods=["GET", "HEAD"])
async def proxy_safe_get(request: Request, path: str):
    if path not in _PASSTHROUGH_GET_ALLOWLIST:
        raise HTTPException(status_code=404, detail="not found")

    client = _client(request)
    upstream = _upstream_url(request)
    try:
        upstream_resp = await client.request(
            method=request.method,
            url=f"{upstream}/{path}",
            params=dict(request.query_params),
            timeout=30.0,
        )
    except httpx.TimeoutException as exc:
        return _bad_gateway("upstream timeout", exc)
    except httpx.HTTPError as exc:
        return _bad_gateway("upstream unreachable", exc)

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type"),
    )


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="v42 Boundary Guard v7 (H26)")
    parser.add_argument(
        "--upstream",
        default=DEFAULT_UPSTREAM,
        help=f"Upstream llama-server URL (default: {DEFAULT_UPSTREAM})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Port to serve guard-v7 on (default: 8088, distinct from v1=8082..v6=8087)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()
    app.state.upstream = args.upstream
    log.info("Starting v42 boundary guard v7 on %s:%d", args.host, args.port)
    log.info("Upstream llama-server: %s", args.upstream)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
