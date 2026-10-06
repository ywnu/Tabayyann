from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
import httpx
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tabayyann_query_guard_v9 import classify_query, retrieve_hits

# Anthropic is loaded only when a valid grounded answer needs the LLM.
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Tabayyan final answer policy (fail closed)
# ---------------------------------------------------------------------------
ANSWER_POLICY_FILE = os.getenv(
    "TABAYYANN_ANSWER_POLICY_FILE",
    "tabayyan_answer_policy_v1.json",
)
ANSWER_POLICY_PATH = ROOT / ANSWER_POLICY_FILE

_REQUIRED_POLICY_STATUSES = {
    "supported",
    "insufficient_evidence",
    "conflict",
    "needs_clarification",
    "out_of_scope",
    "not_found",
}

_REQUIRED_HARD_RULES = {
    "supported requires sources.length >= 1",
    "no fabricated citations",
    "no fabricated hadith or ayah text",
    "no model-memory fallback when evidence is missing",
    "citation remains required even if user asks to omit it",
}


def _load_answer_policy() -> dict[str, Any]:
    """Load and validate the final answer policy. Backend fails closed if invalid."""
    if not ANSWER_POLICY_PATH.exists():
        raise RuntimeError(
            f"Required answer policy file is missing: {ANSWER_POLICY_PATH}"
        )

    try:
        with ANSWER_POLICY_PATH.open("r", encoding="utf-8") as f:
            policy = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load answer policy: {ANSWER_POLICY_PATH}"
        ) from exc

    statuses = policy.get("statuses")
    if not isinstance(statuses, dict):
        raise RuntimeError("Answer policy statuses are missing or invalid.")

    missing_statuses = _REQUIRED_POLICY_STATUSES - set(statuses)
    if missing_statuses:
        raise RuntimeError(
            "Answer policy is incomplete. Missing statuses: "
            f"{sorted(missing_statuses)}"
        )

    hard_rules = policy.get("hard_rules")
    if not isinstance(hard_rules, list):
        raise RuntimeError("Answer policy hard_rules are missing or invalid.")

    missing_rules = _REQUIRED_HARD_RULES - set(hard_rules)
    if missing_rules:
        raise RuntimeError(
            "Answer policy is missing mandatory hard rules: "
            f"{sorted(missing_rules)}"
        )

    return policy


ANSWER_POLICY = _load_answer_policy()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

# ElevenLabs TTS: Wiam voice selected by the user.
ELEVENLABS_VOICE_ID = os.getenv(
    "ELEVENLABS_VOICE_ID",
    "R5kMoWNNTn84ezIJA53m",
)
ELEVENLABS_MODEL_ID = os.getenv(
    "ELEVENLABS_MODEL_ID",
    "eleven_multilingual_v2",
)
MAX_TTS_TEXT_CHARS = 2500
MAX_ELEVENLABS_CALLS_PER_PROCESS = int(
    os.getenv("TABAYYANN_MAX_ELEVENLABS_CALLS", "25")
)


# ElevenLabs Speech-to-Text (Scribe) for reliable mobile microphone support.
ELEVENLABS_STT_MODEL_ID = os.getenv(
    "ELEVENLABS_STT_MODEL_ID",
    "scribe_v2",
)
MAX_STT_AUDIO_BYTES = 8 * 1024 * 1024
MAX_ELEVENLABS_STT_CALLS_PER_PROCESS = int(
    os.getenv("TABAYYANN_MAX_ELEVENLABS_STT_CALLS", "25")
)

TOP_K = 6
MAX_TOKENS = 350
MAX_QUESTION_CHARS = 1000

# Cost / loop safety:
# - exactly one Claude request per /ask call
# - no automatic retries
# - one request processed at a time
# - hard per-process call budget for local testing
MAX_CLAUDE_CALLS_PER_PROCESS = int(os.getenv("TABAYYANN_MAX_CLAUDE_CALLS", "25"))

_retriever = None
_grounding = None
_anthropic_client = None
_claude_calls = 0
_elevenlabs_calls = 0
_elevenlabs_stt_calls = 0
_lock = threading.Lock()
_tts_lock = threading.Lock()
_stt_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TTS_TEXT_CHARS)


class Citation(BaseModel):
    id: int
    source: str
    reference: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    status: str
    answer: str
    citations: list[Citation]
    sources: list[str]
    model: str | None
    usage: dict[str, int]
    claude_calls_this_process: int
    claude_call_limit: int


app = FastAPI(
    title="Tabayyann Islamic RAG API",
    version="1.9.0",
    description="Grounded Islamic Q&A over approved local sources.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Frontend files served by the same FastAPI service (single public URL).
# ---------------------------------------------------------------------------
INDEX_FILE = ROOT / "index.html"
MAIN_LOGO_FILE = ROOT / "MainLogo.PNG"
SUB_LOGO_FILE = ROOT / "SubLogo.png"


@app.get("/", include_in_schema=False)
def frontend_index():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html is missing.")
    return FileResponse(INDEX_FILE)


@app.get("/MainLogo.PNG", include_in_schema=False)
def frontend_main_logo():
    if not MAIN_LOGO_FILE.exists():
        raise HTTPException(status_code=404, detail="MainLogo.PNG is missing.")
    return FileResponse(MAIN_LOGO_FILE)


@app.get("/SubLogo.png", include_in_schema=False)
def frontend_sub_logo():
    if not SUB_LOGO_FILE.exists():
        raise HTTPException(status_code=404, detail="SubLogo.png is missing.")
    return FileResponse(SUB_LOGO_FILE)


@app.get("/favicon.ico", include_in_schema=False)
def frontend_favicon():
    if not SUB_LOGO_FILE.exists():
        raise HTTPException(status_code=404, detail="SubLogo.png is missing.")
    return FileResponse(SUB_LOGO_FILE)




def _import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


GROUNDING_FILE = os.getenv(
    "TABAYYANN_GROUNDING_FILE",
    "unified_04_prepare_grounded_answer (4).py",
)


def _find_grounding_script() -> Path:
    # Use the explicitly validated grounding file.
    # Never choose a grounding implementation by modification time.
    path = ROOT / GROUNDING_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Validated grounding file not found: {path}. "
            "Set TABAYYANN_GROUNDING_FILE only if you intentionally validated another file."
        )
    return path


def _load_retriever_once():
    global _retriever

    if _retriever is not None:
        return _retriever

    path = ROOT / "unified_02_retriever_v7.py"
    if not path.exists():
        raise FileNotFoundError(path)

    module = _import_module(path, "tabayyann_backend_retriever_v7")
    cls = getattr(module, "UnifiedRetriever", None)
    if cls is None:
        raise RuntimeError("UnifiedRetriever class was not found in V7.")

    print("Loading UnifiedRetriever once for backend...")
    _retriever = cls()
    print("UnifiedRetriever is ready.")
    return _retriever


def _load_grounding_once():
    global _grounding

    if _grounding is not None:
        return _grounding

    path = _find_grounding_script()
    print(f"Grounding layer: {path.name}")
    _grounding = _import_module(path, "tabayyann_backend_grounding")
    return _grounding


def _get_anthropic_client():
    global _anthropic_client

    if _anthropic_client is not None:
        return _anthropic_client

    if Anthropic is None:
        raise RuntimeError(
            "Anthropic SDK is not installed. Run: python -m pip install anthropic"
        )

    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Set it in the Run Configuration "
            "or terminal before starting the backend."
        )

    if not key.startswith("sk-ant-"):
        raise RuntimeError("ANTHROPIC_API_KEY does not look valid.")

    _anthropic_client = Anthropic(api_key=key)
    return _anthropic_client


def _extract_text(message: Any) -> str:
    parts = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            value = getattr(block, "text", None)
            if value:
                parts.append(str(value).strip())
    return "\n".join(parts).strip()


def _usage_dict(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}

    out = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out


def _stop_reason(message: Any) -> str | None:
    value = getattr(message, "stop_reason", None)
    return str(value) if value is not None else None


def _valid_citations(answer: str, evidence_count: int) -> bool:
    ids = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    if not ids:
        return False
    return all(1 <= n <= evidence_count for n in ids)


def _used_citation_ids(answer: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})


def _norm_for_quote_check(value: str) -> str:
    value = value or ""
    value = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", value)
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("ى", "ي").replace("ة", "ه")
    value = re.sub(r"[^\u0600-\u06FF0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _quoted_segments(answer: str) -> list[str]:
    segments = []
    for pattern in (r"«([^»]+)»", r'"([^"]+)"'):
        for match in re.findall(pattern, answer or ""):
            value = str(match).strip()
            if len(value) >= 8:
                segments.append(value)
    return segments


def _quotes_are_grounded(answer: str, packet: dict) -> bool:
    quotes = _quoted_segments(answer)
    if not quotes:
        return True

    evidence_texts = []
    for evidence in packet.get("evidence", []) or []:
        for key in ("text", "title", "reference"):
            value = evidence.get(key)
            if value:
                evidence_texts.append(_norm_for_quote_check(str(value)))

    for quote in quotes:
        normalized_quote = _norm_for_quote_check(quote)
        if not normalized_quote:
            continue
        if not any(normalized_quote in evidence_text for evidence_text in evidence_texts):
            return False
    return True



def _remove_sentences_with_ungrounded_quotes(answer: str, packet: dict) -> str:
    """
    Local safety fallback with ZERO extra LLM/API calls.

    If Claude produced a literal quotation that cannot be matched to the
    retrieved evidence, remove only the sentence containing that quotation.
    """
    quotes = _quoted_segments(answer)
    if not quotes:
        return answer

    evidence_texts = []
    for evidence in packet.get("evidence", []) or []:
        for key in ("text", "title", "reference"):
            value = evidence.get(key)
            if value:
                evidence_texts.append(_norm_for_quote_check(str(value)))

    bad_quotes = []
    for quote in quotes:
        normalized_quote = _norm_for_quote_check(quote)
        if normalized_quote and not any(
            normalized_quote in evidence_text
            for evidence_text in evidence_texts
        ):
            bad_quotes.append(quote)

    if not bad_quotes:
        return answer

    parts = re.split(r"(?<=[.!؟?])\\s+|(?<=؛)\\s+", answer or "")
    kept = []

    for part in parts:
        part_norm = _norm_for_quote_check(part)
        contains_bad_quote = any(
            _norm_for_quote_check(quote) in part_norm
            for quote in bad_quotes
        )
        if not contains_bad_quote and part.strip():
            kept.append(part.strip())

    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"\\s{2,}", " ", cleaned)
    return cleaned


def _contains_unrequested_consensus_claim(question: str, answer: str) -> bool:
    answer_norm = _norm_for_quote_check(answer)
    question_norm = _norm_for_quote_check(question)

    consensus_terms = ("اجماع", "اتفاق العلماء", "اتفق العلماء")
    answer_has = any(term in answer_norm for term in consensus_terms)
    question_asks = any(term in question_norm for term in consensus_terms)
    return answer_has and not question_asks


def _remove_unrequested_consensus_claims(question: str, answer: str) -> str:
    """Remove only generic consensus wording when the user did not ask about consensus."""
    question_norm = _norm_for_quote_check(question)
    if any(term in question_norm for term in ("اجماع", "اتفاق العلماء", "اتفق العلماء")):
        return answer

    cleaned = answer or ""

    # Narrow substitutions only; do not alter the substantive ruling or citations.
    patterns = [
        r"\s*بالنص\s*والإجماع",
        r"\s*بالإجماع",
        r"\s*بإجماع\s+العلماء",
        r"\s*باتفاق\s+العلماء",
        r"\s*وقد\s+اتفق\s+العلماء",
        r"\s*واتفق\s+العلماء",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+([،,.؛:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _arabic_focus_terms(value: str) -> list[str]:
    """Very small local normalizer used only to reduce topic leakage in retrieved hits."""
    value = value or ""
    value = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", value)
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("ى", "ي").replace("ة", "ه")
    tokens = re.findall(r"[\u0600-\u06FF]+", value)

    stop = {
        "ما", "ماذا", "هل", "وش", "ايش", "ابي", "اريد", "اعرف", "عن",
        "في", "من", "على", "الى", "او", "و", "ثم", "هذا", "هذه", "ذلك",
        "حكم", "الحكم", "يجوز", "جائز", "حرام", "حلال", "صحيح", "تحقق",
        "العباره", "المساله", "السؤال", "لو", "سمحت", "بسرعه", "بدون",
        "تخمين", "ممكن", "لي", "لها", "له", "علي", "عليها",
    }

    out = []
    for token in tokens:
        if token.startswith("ال") and len(token) > 4:
            token = token[2:]
        if len(token) < 3 or token in stop:
            continue
        if token not in out:
            out.append(token)
    return out


def _hit_searchable_text(hit: Any) -> str:
    parts = []
    for name in ("title", "reference", "text", "source_name", "source"):
        value = getattr(hit, name, None)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _focus_hits_on_question(raw_hits: list[Any], question: str) -> list[Any]:
    """
    If a narrow question has at least two meaningful anchor terms and at least two
    retrieved hits match all/max anchors, keep those maximally focused hits.
    Otherwise preserve the original retrieval unchanged.
    """
    if not raw_hits:
        return raw_hits

    anchors = _arabic_focus_terms(question)
    if len(anchors) < 2:
        return raw_hits

    scored = []
    for hit in raw_hits:
        hit_terms = set(_arabic_focus_terms(_hit_searchable_text(hit)))
        score = sum(1 for term in anchors if term in hit_terms)
        scored.append((score, hit))

    max_score = max(score for score, _ in scored)
    if max_score < 2:
        return raw_hits

    focused = [hit for score, hit in scored if score == max_score]

    # Require at least two strong hits before pruning so we do not over-filter
    # when only one result happens to share all wording with the question.
    if len(focused) < 2:
        return raw_hits

    return focused


def _call_claude_once(packet: dict) -> tuple[str, Any]:
    global _claude_calls

    if _claude_calls >= MAX_CLAUDE_CALLS_PER_PROCESS:
        raise HTTPException(
            status_code=429,
            detail=(
                "Claude safety budget reached for this backend process. "
                "Restart intentionally or raise TABAYYANN_MAX_CLAUDE_CALLS."
            ),
        )

    client = _get_anthropic_client()
    llm = packet["llm_payload"]

    strict_system_rules = (
        "\n\nقواعد إخراج إلزامية لهذا النظام:"
        "\n- حدّد داخليًا الموضوع الدقيق الذي سأل عنه المستخدم، ثم أجب عنه وحده."
        "\n- قد تحتوي حزمة الأدلة على نتائج قريبة لكنها خارج موضوع السؤال؛ تجاهلها بالكامل."
        "\n- تجاهل أي أحكام أو مسائل جانبية تظهر في الأدلة إذا لم يطلبها المستخدم صراحة."
        "\n- ممنوع إضافة موضوع ثانٍ لمجرد أنه ورد في نتيجة الاسترجاع."
        "\n- لا تستخدم عناوين، ولا قوائم، ولا أقسام."
        "\n- اكتب جوابًا عربيًا قصيرًا من 40 إلى 60 كلمة تقريبًا."
        "\n- اذكر الحكم المباشر أولًا، ثم دليلًا أو دليلين كحد أقصى."
        "\n- استخدم الاستشهادات داخل الجمل مثل [1] أو [2]."
        "\n- لا تكرر الحكم أو الدليل."
        "\n- لا تستخدم كلمة إجماع أو اتفاق العلماء إلا إذا كان سؤال المستخدم نفسه عن الإجماع أو الاتفاق."
        "\n- لا تستخدم علامات اقتباس للنصوص الشرعية في الإجابة؛ لخّص المعنى بصياغتك مع الاستشهاد بالمصدر."
        "\n- لا تنقل آية أو حديثًا أو قول عالم نقلًا حرفيًا ما لم يكن ذلك ضروريًا جدًا، والأصل هو التلخيص."
        "\n- لا تنسب قولًا إلى عالم إلا إذا كان الدليل المستشهد به من مصدره نفسه."
        "\n- ممنوع استخدام المعرفة العامة للنموذج لسد أي نقص في الأدلة المسترجعة."
        "\n- ممنوع اختلاق مرجع أو نص آية أو حديث أو نسبة غير موجودة في الأدلة."
        "\n- إذا لم تكفِ الأدلة للحكم، لا تخمّن ولا تُكمل من الذاكرة."
        "\n- إذا كان السؤال عن حكم واحد، أنهِ الإجابة فور اكتمال حكمه ودليله."
        "\n- يجب أن تكون آخر جملة مكتملة، ثم توقف."
    )

    concise_user_prompt = llm["user"]
    strict_system_prompt = llm["system"] + strict_system_rules

    # Exactly ONE external LLM call in this function. No loop, no retry.
    # Count the attempt before sending it so the process budget remains conservative
    # even if the external request fails after being submitted.
    _claude_calls += 1
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=strict_system_prompt,
        messages=[
            {
                "role": "user",
                "content": concise_user_prompt,
            }
        ],
    )

    return _extract_text(message), message


def _packet_from_question(question: str, route: str) -> dict:
    retriever = _load_retriever_once()
    grounding = _load_grounding_once()

    raw_hits = retrieve_hits(retriever, question, route, top_k=TOP_K)
    raw_hits = _focus_hits_on_question(raw_hits, question)

    retrieval_meta = {
        "retriever_file": str(ROOT / "unified_02_retriever_v7.py"),
        "adapter": f"backend:tabayyann_query_guard_v9:{route}",
    }

    return grounding.prepare_packet(
        question=question,
        raw_hits=raw_hits,
        retrieval_meta=retrieval_meta,
        top_k=TOP_K,
    )


def _citations_from_packet(packet: dict) -> list[dict]:
    rows = []
    for i, evidence in enumerate(packet.get("evidence", []), 1):
        rows.append(
            {
                "id": i,
                "source": evidence.get("source") or "Unknown Source",
                "reference": evidence.get("reference"),
                "url": evidence.get("url"),
            }
        )
    return rows


def _sources_from_citations(citations: list[dict]) -> list[str]:
    sources: list[str] = []
    for row in citations:
        source = row.get("source")
        if source and source not in sources:
            sources.append(source)
    return sources



def _validate_final_supported_answer(
    *,
    status: str,
    answer: str,
    citations: list[dict],
    sources: list[str],
) -> tuple[bool, str]:
    """Final fail-closed check before any grounded PASS answer reaches the UI."""
    if status != "PASS":
        return True, ""

    if not answer.strip():
        return False, "empty supported answer"

    if not citations:
        return False, "supported answer has no citations"

    if not sources:
        return False, "supported answer has no sources"

    citation_ids = {int(row.get("id")) for row in citations if row.get("id") is not None}
    used_ids = set(_used_citation_ids(answer))
    if not used_ids or not used_ids.issubset(citation_ids):
        return False, "answer citations are not backed by returned evidence"

    for row in citations:
        source = str(row.get("source") or "").strip()
        if not source or source == "Unknown Source":
            return False, "citation source is missing"

    return True, ""


@app.on_event("startup")
def startup_event():
    # Expensive local assets are loaded once, not once per question.
    _load_grounding_once()
    _load_retriever_once()
    print("=" * 90)
    print("TABAYYANN BACKEND READY")
    print(f"Model: {MODEL}")
    print(f"Answer policy: {ANSWER_POLICY_FILE} (v{ANSWER_POLICY.get('version', 'unknown')})")
    print(f"Claude calls safety limit: {MAX_CLAUDE_CALLS_PER_PROCESS} per process")
    print("No retry loop is implemented.")
    print("=" * 90)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "retriever_loaded": _retriever is not None,
        "model": MODEL,
        "guard": "tabayyann_query_guard_v9.py",
        "grounding_file": GROUNDING_FILE,
        "answer_policy_loaded": True,
        "answer_policy_file": ANSWER_POLICY_FILE,
        "answer_policy_version": str(ANSWER_POLICY.get("version", "unknown")),
        "frontend_configured": INDEX_FILE.exists(),
        "claude_calls": _claude_calls,
        "claude_call_limit": MAX_CLAUDE_CALLS_PER_PROCESS,
        "tts_configured": bool(os.getenv("ELEVENLABS_API_KEY", "").strip()),
        "tts_voice_id": ELEVENLABS_VOICE_ID,
        "tts_model": ELEVENLABS_MODEL_ID,
        "tts_calls": _elevenlabs_calls,
        "tts_call_limit": MAX_ELEVENLABS_CALLS_PER_PROCESS,
        "stt_configured": bool(os.getenv("ELEVENLABS_API_KEY", "").strip()),
        "stt_model": ELEVENLABS_STT_MODEL_ID,
        "stt_calls": _elevenlabs_stt_calls,
        "stt_call_limit": MAX_ELEVENLABS_STT_CALLS_PER_PROCESS,
    }


@app.post("/stt")
def speech_to_text(file: UploadFile = File(...)):
    """
    Convert a short microphone recording to text using ElevenLabs Scribe.
    The API key stays backend-only. One external STT request per call, no retry.
    """
    global _elevenlabs_stt_calls

    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="خدمة تحويل الصوت إلى نص غير مهيأة على الخادم.",
        )

    filename = Path(file.filename or "recording.webm").name
    content_type = str(file.content_type or "application/octet-stream")

    audio = file.file.read(MAX_STT_AUDIO_BYTES + 1)

    if not audio:
        raise HTTPException(
            status_code=400,
            detail="التسجيل الصوتي فارغ.",
        )

    if len(audio) > MAX_STT_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="التسجيل الصوتي طويل جدًا. سجلي سؤالًا أقصر.",
        )

    with _stt_lock:
        if _elevenlabs_stt_calls >= MAX_ELEVENLABS_STT_CALLS_PER_PROCESS:
            raise HTTPException(
                status_code=429,
                detail="تم الوصول إلى الحد الآمن لطلبات تحويل الصوت إلى نص.",
            )

        _elevenlabs_stt_calls += 1

        try:
            response = httpx.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={
                    "xi-api-key": key,
                    "Accept": "application/json",
                },
                data={
                    "model_id": ELEVENLABS_STT_MODEL_ID,
                },
                files={
                    "file": (
                        filename,
                        audio,
                        content_type,
                    )
                },
                timeout=60.0,
            )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail="انتهت مهلة خدمة التعرف على الصوت. حاولي مرة أخرى.",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail="تعذر الاتصال بخدمة التعرف على الصوت.",
            ) from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=(
                    "خدمة التعرف على الصوت رفضت الطلب "
                    f"(status {response.status_code})."
                ),
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="خدمة التعرف على الصوت أعادت استجابة غير صالحة.",
            ) from exc

        transcript = str(data.get("text") or "").strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="لم أتمكن من التعرف على الكلام بوضوح.",
            )

        return {
            "text": transcript,
            "language_code": data.get("language_code"),
            "model": ELEVENLABS_STT_MODEL_ID,
            "stt_calls_this_process": _elevenlabs_stt_calls,
            "stt_call_limit": MAX_ELEVENLABS_STT_CALLS_PER_PROCESS,
        }


@app.post("/tts")
def tts(body: TTSRequest):
    """
    Convert an already-produced answer to speech using the configured ElevenLabs voice.

    Safety / cost rules:
    - API key stays backend-only.
    - Maximum one ElevenLabs request per /tts call.
    - No automatic retries.
    - Hard per-process TTS call limit.
    """
    global _elevenlabs_calls

    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs TTS is not configured on the backend.",
        )

    text = str(body.text or "").strip()

    # TTS cleanup: never read citation markers such as [1], [١], [1, 2], or 【2】 aloud.
    # Remove invisible bidi marks first because Arabic text can place them inside brackets.
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = re.sub(
        r"[\[【]\s*[0-9٠-٩۰-۹]+(?:\s*[,،\-–—]\s*[0-9٠-٩۰-۹]+)*\s*[\]】]",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise HTTPException(status_code=400, detail="TTS text is empty.")

    if len(text) > MAX_TTS_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"TTS text exceeds {MAX_TTS_TEXT_CHARS} characters.",
        )

    with _tts_lock:
        if _elevenlabs_calls >= MAX_ELEVENLABS_CALLS_PER_PROCESS:
            raise HTTPException(
                status_code=429,
                detail=(
                    "ElevenLabs TTS safety budget reached for this backend process. "
                    "Restart intentionally or raise TABAYYANN_MAX_ELEVENLABS_CALLS."
                ),
            )

        payload = json.dumps(
            {
                "text": text,
                "model_id": ELEVENLABS_MODEL_ID,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        url = (
            "https://api.elevenlabs.io/v1/text-to-speech/"
            f"{ELEVENLABS_VOICE_ID}?output_format=mp3_44100_128"
        )

        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )

        # Count the attempt immediately before the paid/external call.
        _elevenlabs_calls += 1

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                audio = response.read()
        except urllib.error.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"ElevenLabs TTS request failed with status {exc.code}.",
            ) from exc
        except urllib.error.URLError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not connect to ElevenLabs TTS: {type(exc.reason).__name__}: {exc.reason}",
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="ElevenLabs TTS request timed out.",
            ) from exc

        if not audio:
            raise HTTPException(
                status_code=502,
                detail="ElevenLabs returned an empty audio response.",
            )

        return Response(
            content=audio,
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Tabayyann-TTS-Voice": ELEVENLABS_VOICE_ID,
            },
        )


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    guard = classify_query(question)
    action = guard.get("action")
    route = guard.get("route")

    # Terminal Guard V9 states: NEVER call Claude.
    if action == "needs_clarification":
        return AskResponse(
            status="NEEDS_CLARIFICATION",
            answer="السؤال يحتاج توضيحًا قبل البحث. اذكري المسألة أو النص أو الفعل المقصود بشكل محدد.",
            citations=[],
            sources=[],
            model=None,
            usage={},
            claude_calls_this_process=_claude_calls,
            claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
        )

    if action == "out_of_scope":
        return AskResponse(
            status="OUT_OF_SCOPE",
            answer="هذا السؤال خارج نطاق تبيّن، المخصص للإجابة من المصادر الشرعية المعتمدة في النظام.",
            citations=[],
            sources=[],
            model=None,
            usage={},
            claude_calls_this_process=_claude_calls,
            claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
        )

    if action == "insufficient_evidence":
        return AskResponse(
            status="INSUFFICIENT_EVIDENCE",
            answer="لا توجد أدلة موثقة كافية في المصادر المعتمدة تسمح لي بالجزم، ولن أختلق جوابًا أو مرجعًا.",
            citations=[],
            sources=[],
            model=None,
            usage={},
            claude_calls_this_process=_claude_calls,
            claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
        )

    if action == "not_found":
        return AskResponse(
            status="NOT_FOUND",
            answer="لم أجد المرجع أو النص المطلوب بشكل موثوق في المصادر المعتمدة، لذلك لن أنسب نصًا أو رقمًا غير متأكد منه.",
            citations=[],
            sources=[],
            model=None,
            usage={},
            claude_calls_this_process=_claude_calls,
            claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
        )

    if action == "conflict":
        return AskResponse(
            status="CONFLICT",
            answer="لا يصح أن أختار رأيًا من عندي عند اختلاف المصادر. يلزم عرض الأقوال الموثقة كما هي دون ترجيح اعتباطي.",
            citations=[],
            sources=[],
            model=None,
            usage={},
            claude_calls_this_process=_claude_calls,
            claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
        )

    # Only these actions are allowed to continue into retrieval.
    if action not in {"retrieve", "supported", "conflict_with_evidence"}:
        return AskResponse(
            status="ABSTAIN",
            answer="تعذر تحديد مسار آمن للإجابة، لذلك لن أتابع دون تحقق.",
            citations=[],
            sources=[],
            model=None,
            usage={},
            claude_calls_this_process=_claude_calls,
            claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
        )

    if not route:
        return AskResponse(
            status="ABSTAIN",
            answer="تعذر تحديد مصدر البحث المناسب بأمان، لذلك لن أتابع دون تحقق.",
            citations=[],
            sources=[],
            model=None,
            usage={},
            claude_calls_this_process=_claude_calls,
            claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
        )

    # Serial execution prevents accidental concurrent double-spend during testing.
    with _lock:
        try:
            packet = _packet_from_question(question, route)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

        decision = packet["decision"]
        citations = _citations_from_packet(packet)

        # Consensus/conflict path: retrieve evidence, expose it,
        # but never ask Claude to choose a side or invent consensus.
        if action == "conflict_with_evidence":
            if not citations:
                return AskResponse(
                    status="INSUFFICIENT_EVIDENCE",
                    answer="لم أسترجع أدلة موثقة كافية لفحص دعوى الاتفاق، لذلك لن أجزم بوجود إجماع أو خلاف.",
                    citations=[],
                    sources=[],
                    model=None,
                    usage={},
                    claude_calls_this_process=_claude_calls,
                    claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
                )

            return AskResponse(
                status="CONFLICT",
                answer=(
                    "هذه صياغة تتضمن دعوى اتفاق عام، ولا ينبغي الجزم بها أو اختيار رأي من عندي. "
                    "أعرض لك المصادر المرتبطة التي استرجعها النظام للمراجعة دون ترجيح اعتباطي."
                ),
                citations=citations,
                sources=_sources_from_citations(citations),
                model=None,
                usage={},
                claude_calls_this_process=_claude_calls,
                claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
            )

        if not decision.get("can_answer"):
            return AskResponse(
                status="INSUFFICIENT_EVIDENCE",
                answer=(
                    "لا توجد أدلة مسترجعة كافية وموثقة للإجابة عن السؤال بثقة. "
                    "يُفضّل الرجوع إلى مصدر شرعي موثوق أو مفتٍ مؤهل."
                ),
                citations=citations,
                sources=_sources_from_citations(citations),
                model=None,
                usage={},
                claude_calls_this_process=_claude_calls,
                claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
            )

        # Supported + grounded only: at most ONE Claude call.
        try:
            answer, message = _call_claude_once(packet)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Claude API error: {exc}")

        stop = _stop_reason(message)
        answer = _remove_unrequested_consensus_claims(question, answer)

        # Local safety fallback: if a literal quotation cannot be matched
        # to retrieved evidence, remove only the sentence that contains it.
        # No retry and no additional Claude call.
        if not _quotes_are_grounded(answer, packet):
            answer = _remove_sentences_with_ungrounded_quotes(answer, packet)

        citation_ok = _valid_citations(answer, len(packet.get("evidence", [])))
        quotes_ok = _quotes_are_grounded(answer, packet)
        consensus_ok = not _contains_unrequested_consensus_claim(question, answer)

        if not answer:
            status = "LLM_EMPTY_RESPONSE"
        elif stop == "max_tokens":
            # Never expose a visibly cut-off religious answer to the frontend.
            return AskResponse(
                status="REVIEW_REQUIRED",
                answer="تعذر إكمال الجواب ضمن الحد الآمن للإخراج، لذلك لن أعرض إجابة مبتورة. أعد صياغة السؤال بشكل أكثر تحديدًا.",
                citations=[],
                sources=[],
                model=MODEL,
                usage=_usage_dict(message),
                claude_calls_this_process=_claude_calls,
                claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
            )
        elif not citation_ok:
            status = "REVIEW_REQUIRED"
        elif not quotes_ok:
            return AskResponse(
                status="REVIEW_REQUIRED",
                answer="تم حجب الإجابة لأن فيها نقلًا حرفيًا لم أتمكن من مطابقته مباشرة مع النصوص المسترجعة.",
                citations=[],
                sources=[],
                model=MODEL,
                usage=_usage_dict(message),
                claude_calls_this_process=_claude_calls,
                claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
            )
        elif not consensus_ok:
            return AskResponse(
                status="REVIEW_REQUIRED",
                answer="تم حجب الإجابة لأنها تضمنت دعوى إجماع أو اتفاق لم يطلبها السؤال.",
                citations=[],
                sources=[],
                model=MODEL,
                usage=_usage_dict(message),
                claude_calls_this_process=_claude_calls,
                claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
            )
        else:
            status = "PASS"

        used_ids = set(_used_citation_ids(answer))
        used_citations = [
            row for row in citations
            if row["id"] in used_ids
        ]
        used_sources = _sources_from_citations(used_citations)

        policy_ok, policy_reason = _validate_final_supported_answer(
            status=status,
            answer=answer,
            citations=used_citations,
            sources=used_sources,
        )
        if not policy_ok:
            print(f"POLICY BLOCK: {policy_reason}")
            return AskResponse(
                status="REVIEW_REQUIRED",
                answer=(
                    "تم حجب الإجابة لأن شروط التوثيق النهائي لم تكتمل، "
                    "لذلك لن أعرض جوابًا غير موثّق."
                ),
                citations=[],
                sources=[],
                model=MODEL,
                usage=_usage_dict(message),
                claude_calls_this_process=_claude_calls,
                claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
            )

        return AskResponse(
            status=status,
            answer=answer,
            citations=used_citations,
            sources=used_sources,
            model=MODEL,
            usage=_usage_dict(message),
            claude_calls_this_process=_claude_calls,
            claude_call_limit=MAX_CLAUDE_CALLS_PER_PROCESS,
        )



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,  # IMPORTANT: no auto-reload loop during paid API testing.
        log_level="info",
    )
