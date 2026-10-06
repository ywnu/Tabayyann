from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "unified_rag" / "grounded_answer_packets"

RETRIEVER_CANDIDATES = [
    "unified_02_retriever_v7.py",
    "unified_02_retriever_v6.py",
    "unified_02_retriever_v5.py",
    "unified_02_retriever_v4.py",
    "unified_02_retriever_v3.py",
    "unified_02_retriever_v2.py",
    "unified_02_retriever.py",
]

KNOWN_SOURCES = {
    "quran": ("Quran", "quran"),
    "القرآن": ("Quran", "quran"),

    "bukhari": ("Sahih al-Bukhari", "hadith"),
    "sahih_bukhari": ("Sahih al-Bukhari", "hadith"),
    "sahih al-bukhari": ("Sahih al-Bukhari", "hadith"),
    "صحيح البخاري": ("Sahih al-Bukhari", "hadith"),

    "ibn_baz": ("Ibn Baz", "fatwa"),
    "ibn baz": ("Ibn Baz", "fatwa"),
    "binbaz": ("Ibn Baz", "fatwa"),
    "ابن باز": ("Ibn Baz", "fatwa"),

    "ibn_uthaymeen": ("Ibn Uthaymeen", "fatwa"),
    "uthaymeen": ("Ibn Uthaymeen", "fatwa"),
    "ibn uthaymeen": ("Ibn Uthaymeen", "fatwa"),
    "ابن عثيمين": ("Ibn Uthaymeen", "fatwa"),

    "tafsir_ibn_kathir": ("Tafsir Ibn Kathir", "tafsir"),
    "ibn_kathir": ("Tafsir Ibn Kathir", "tafsir"),
    "ibn kathir": ("Tafsir Ibn Kathir", "tafsir"),
    "tafsir ibn kathir": ("Tafsir Ibn Kathir", "tafsir"),
    "ابن كثير": ("Tafsir Ibn Kathir", "tafsir"),
}

TEXT_KEYS = [
    "text", "content", "passage", "chunk_text", "document", "body",
    "hadith_text", "tafsir_text", "ayah_text_uthmani", "ayah_text_search",
    "answer",
]
SOURCE_KEYS = ["source", "source_name", "dataset", "origin"]
TITLE_KEYS = ["title", "chapter_name", "book_name", "surah_name_ar", "name"]
REFERENCE_KEYS = ["source_reference", "reference", "citation", "ref"]
URL_KEYS = ["url", "source_url", "link"]
SCORE_KEYS = [
    "fused_score", "score", "final_score", "rrf_score", "hybrid_score",
    "similarity", "dense_score", "retrieval_score",
]
ID_KEYS = [
    "global_unit_id", "unit_id", "global_id", "chunk_id", "record_id",
    "parent_id", "occurrence_id", "ayah_key", "id",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def first_value(d: dict, keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, "", []):
            return d[key]
    return default


def flatten_hit(hit: Any) -> dict:
    """
    Convert V7 SearchResult objects (dataclass/object), dicts, or plain values
    into one flat dictionary. Metadata is merged only as a fallback so the
    top-level V7 fields remain authoritative.
    """
    if isinstance(hit, dict):
        base = dict(hit)
    elif dataclasses.is_dataclass(hit):
        base = {
            field.name: getattr(hit, field.name)
            for field in dataclasses.fields(hit)
        }
    elif hasattr(hit, "__dict__"):
        base = dict(vars(hit))
    else:
        return {"text": clean_text(hit)}

    meta = base.get("metadata")
    if isinstance(meta, dict):
        for k, v in meta.items():
            base.setdefault(k, v)

    return base


def canonical_source(raw: Any, hit: dict) -> tuple[str, str]:
    pieces = [
        clean_text(raw),
        clean_text(hit.get("source_type")),
        clean_text(hit.get("dataset")),
        clean_text(hit.get("origin")),
    ]
    hay = " ".join(pieces).lower()

    for needle, (name, source_type) in KNOWN_SOURCES.items():
        if needle.lower() in hay:
            return name, source_type

    raw_clean = clean_text(raw)
    return (raw_clean or "Unknown Source", clean_text(hit.get("source_type")) or "unknown")


def derive_quran_reference(hit: dict) -> str:
    surah_name = clean_text(hit.get("surah_name_ar"))
    ayah = hit.get("ayah_number")
    ayah_key = clean_text(hit.get("ayah_key"))

    if surah_name and ayah not in (None, ""):
        return f"سورة {surah_name}، الآية {ayah}"
    if ayah_key:
        return f"القرآن الكريم، الآية {ayah_key}"
    return ""


def derive_bukhari_reference(hit: dict) -> str:
    # Keep the human-facing citation clean. Some chapter headings in the
    # source PDF carry OCR noise, so we do not expose the chapter text here.
    # The raw chapter_name remains preserved in metadata for audit purposes.
    bits = ["صحيح البخاري"]

    book = clean_text(hit.get("book_name"))
    num = hit.get("hadith_number")
    page = hit.get("source_page")

    if book:
        bits.append(f"كتاب {book}")
    if num not in (None, ""):
        bits.append(f"حديث رقم {num}")
    if page not in (None, ""):
        bits.append(f"ص {page}")

    return "، ".join(bits)


def derive_uthaymeen_reference(hit: dict) -> str:
    """
    Ibn Uthaymeen unified rows currently have no external URL/title/reference.
    Build a readable citation from the stored question text and a concise
    internal corpus ID. The complete unit ID remains in packet metadata.
    """
    text = clean_text(hit.get("text"))
    unit_id = clean_text(
        hit.get("global_unit_id")
        or hit.get("unit_id")
        or hit.get("chunk_id")
        or hit.get("record_id")
    )

    question_label = ""
    m = re.search(r"(?:السؤال\s*[:：]\s*)(.+?)(?:\s+الجواب\s*[:：]|$)", text)
    if m:
        q = clean_text(m.group(1))
        if len(q) > 120:
            q = q[:117].rstrip() + "..."
        question_label = q

    display_id = ""
    if unit_id:
        if unit_id.startswith("ibn_uthaymeen::"):
            unit_id = unit_id[len("ibn_uthaymeen::"):]
        # First UUID / stable record component is sufficient for display.
        display_id = unit_id.split("::", 1)[0]
        if len(display_id) > 48:
            display_id = display_id[:45] + "..."

    bits = ["فتاوى ابن عثيمين"]
    if question_label:
        bits.append(f"السؤال: {question_label}")
    if display_id:
        bits.append(f"مرجع داخلي: {display_id}")

    return "، ".join(bits)


def derive_tafsir_reference(hit: dict) -> str:
    ayah_key = clean_text(hit.get("primary_ayah_key") or hit.get("ayah_key"))
    start_surah = hit.get("start_surah")
    start_ayah = hit.get("start_ayah")
    end_surah = hit.get("end_surah")
    end_ayah = hit.get("end_ayah")

    if ayah_key:
        return f"تفسير ابن كثير، {ayah_key}"

    if start_surah not in (None, "") and start_ayah not in (None, ""):
        if end_surah not in (None, "") and end_ayah not in (None, ""):
            if str(start_surah) == str(end_surah) and str(start_ayah) == str(end_ayah):
                return f"تفسير ابن كثير، {start_surah}:{start_ayah}"
            return f"تفسير ابن كثير، {start_surah}:{start_ayah}–{end_surah}:{end_ayah}"
        return f"تفسير ابن كثير، {start_surah}:{start_ayah}"

    return ""


def derive_reference(source: str, hit: dict) -> str:
    # Prefer source-specific, richer references before falling back to the
    # generic reference field.
    if source == "Quran":
        ref = derive_quran_reference(hit)
        if ref:
            return ref

    if source == "Sahih al-Bukhari":
        ref = derive_bukhari_reference(hit)
        if ref:
            return ref

    if source == "Ibn Uthaymeen":
        ref = derive_uthaymeen_reference(hit)
        if ref:
            return ref

    if source == "Ibn Baz":
        title = clean_text(first_value(hit, TITLE_KEYS))
        if title:
            return f"فتوى ابن باز: {title}"

    if source == "Tafsir Ibn Kathir":
        ref = derive_tafsir_reference(hit)
        if ref:
            return ref

    explicit = clean_text(first_value(hit, REFERENCE_KEYS))
    if explicit:
        return explicit

    title = clean_text(first_value(hit, TITLE_KEYS))
    if title:
        return title

    unit_id = clean_text(
        hit.get("global_unit_id")
        or hit.get("unit_id")
        or hit.get("chunk_id")
        or hit.get("record_id")
    )
    if source != "Unknown Source" and unit_id:
        return f"{source} corpus unit: {unit_id}"

    return ""


def score_value(hit: dict) -> float | None:
    for key in SCORE_KEYS:
        value = hit.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            if value not in (None, ""):
                return float(value)
        except Exception:
            pass
    return None


def normalize_hit(hit: Any, rank: int) -> dict:
    h = flatten_hit(hit)

    raw_source = first_value(h, SOURCE_KEYS, "")
    source, source_type = canonical_source(raw_source, h)

    text = clean_text(first_value(h, TEXT_KEYS, ""))
    title = clean_text(first_value(h, TITLE_KEYS, ""))
    url = clean_text(first_value(h, URL_KEYS, ""))
    item_id = clean_text(first_value(h, ID_KEYS, ""))

    reference = derive_reference(source, h)

    result = {
        "rank": rank,
        "id": item_id or None,
        "source": source,
        "source_type": source_type,
        "title": title or None,
        "text": text,
        "reference": reference or None,
        "url": url or None,
        "score": score_value(h),
        "metadata": {},
    }

    keep_meta = [
        "global_index", "global_unit_id", "unit_id", "parent_id",
        "fused_score", "dense_score", "dense_rank", "bm25_score", "bm25_rank",
        "ayah_key", "ayah_number", "surah_number", "surah_name_ar",
        "page", "juz", "hadith_number", "book_name", "chapter_name",
        "scholar", "category", "subcategory", "source_page", "record_id", "chunk_id",
        "occurrence_id", "primary_ayah_key", "start_surah", "start_ayah",
        "end_surah", "end_ayah", "women_category",
    ]
    for key in keep_meta:
        if key in h and h[key] not in (None, "", []):
            result["metadata"][key] = h[key]

    return result


def extract_hits(obj: Any) -> list:
    if obj is None:
        return []

    if isinstance(obj, list):
        return obj

    if isinstance(obj, tuple):
        for part in obj:
            hits = extract_hits(part)
            if hits:
                return hits
        return []

    if isinstance(obj, dict):
        for key in (
            "results", "hits", "items", "retrieval_results",
            "documents", "evidence", "data",
        ):
            value = obj.get(key)
            if isinstance(value, list):
                return value

        # Sometimes ranks are stored as dict values.
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())

    return []


def retriever_file() -> Path | None:
    for name in RETRIEVER_CANDIDATES:
        path = ROOT / name
        if path.exists():
            return path
    return None


def import_module(path: Path):
    spec = importlib.util.spec_from_file_location("cira_unified_retriever", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)

    # Important for Python 3.14 and modules that use @dataclass/type inspection:
    # register the module before executing it so sys.modules[cls.__module__] exists.
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def public_signatures(module) -> list[dict]:
    rows = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if inspect.isfunction(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(?)"
            rows.append({"type": "function", "name": name, "signature": sig})
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(?)"
            rows.append({"type": "class", "name": name, "signature": sig})
    return rows


def call_with_signature(func, query: str, top_k: int):
    sig = inspect.signature(func)
    kwargs = {}

    query_names = ["query", "q", "question", "text", "user_query"]
    topk_names = ["top_k", "k", "limit", "n_results", "num_results"]

    params = sig.parameters

    query_param = next((x for x in query_names if x in params), None)
    topk_param = next((x for x in topk_names if x in params), None)

    if query_param:
        kwargs[query_param] = query
        if topk_param:
            kwargs[topk_param] = top_k
        return func(**kwargs)

    # Safe fallback only for functions with one or two required positional params.
    required = [
        p for p in params.values()
        if p.default is inspect._empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]

    if len(required) == 1:
        return func(query)
    if len(required) == 2 and topk_param:
        return func(query, top_k)

    raise TypeError(f"Unsupported callable signature: {sig}")


def run_retriever(query: str, top_k: int) -> tuple[list, dict]:
    path = retriever_file()
    if path is None:
        raise RuntimeError("No unified_02_retriever*.py file was found.")

    module = import_module(path)

    function_names = [
        "retrieve", "search", "retrieve_query", "run_retrieval",
        "unified_retrieve", "run_query", "query",
    ]

    errors = []

    for name in function_names:
        func = getattr(module, name, None)
        if callable(func) and inspect.isfunction(func):
            try:
                raw = call_with_signature(func, query, top_k)
                hits = extract_hits(raw)
                if hits:
                    return hits, {
                        "retriever_file": str(path),
                        "adapter": f"function:{name}",
                    }
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")

    class_names = ["UnifiedRetriever", "Retriever", "HybridRetriever"]
    for class_name in class_names:
        cls = getattr(module, class_name, None)
        if not inspect.isclass(cls):
            continue
        try:
            instance = cls()
        except Exception as e:
            errors.append(f"{class_name}(): {type(e).__name__}: {e}")
            continue

        for method_name in function_names:
            method = getattr(instance, method_name, None)
            if not callable(method):
                continue
            try:
                raw = call_with_signature(method, query, top_k)
                hits = extract_hits(raw)
                if hits:
                    return hits, {
                        "retriever_file": str(path),
                        "adapter": f"class:{class_name}.{method_name}",
                    }
            except Exception as e:
                errors.append(
                    f"{class_name}.{method_name}: {type(e).__name__}: {e}"
                )

    diagnostics = {
        "retriever_file": str(path),
        "public_callables": public_signatures(module),
        "adapter_errors": errors,
    }
    diag_path = ROOT / "UNIFIED_RETRIEVER_ADAPTER_DIAGNOSTIC.json"
    diag_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raise RuntimeError(
        "Retriever file found, but no compatible callable returned results. "
        f"Diagnostic written to: {diag_path}"
    )


def load_retrieval_json(path: Path) -> tuple[list, dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    hits = extract_hits(obj)
    return hits, {
        "retriever_file": None,
        "adapter": "retrieval-json",
        "retrieval_json": str(path),
    }


def evidence_decision(evidence: list[dict]) -> dict:
    usable = [
        e for e in evidence
        if e["text"] and len(e["text"]) >= 20
    ]

    source_names = []
    for e in usable:
        if e["source"] not in source_names:
            source_names.append(e["source"])

    referenced = [e for e in usable if e.get("reference") or e.get("url")]

    scholar_sources = {
        e["source"] for e in usable
        if e["source"] in {"Ibn Baz", "Ibn Uthaymeen"}
    }

    if not usable:
        status = "ABSTAIN_NO_EVIDENCE"
        can_answer = False
        reason = "No usable retrieved evidence."
    elif not referenced:
        status = "ABSTAIN_NO_CITABLE_REFERENCE"
        can_answer = False
        reason = "Evidence exists, but no citable reference or URL was found."
    elif len(usable) == 1 and len(usable[0]["text"]) < 120:
        status = "LIMITED_EVIDENCE"
        can_answer = False
        reason = "Only one short evidence item was retrieved."
    else:
        status = "ANSWER_FROM_EVIDENCE"
        can_answer = True
        reason = "Usable citable evidence is available."

    return {
        "status": status,
        "can_answer": can_answer,
        "reason": reason,
        "usable_evidence_count": len(usable),
        "citable_evidence_count": len(referenced),
        "source_count": len(source_names),
        "sources": source_names,
        "multi_source": len(source_names) >= 2,
        "scholar_comparison_required": len(scholar_sources) >= 2,
        "scholar_sources": sorted(scholar_sources),
    }


def citation_label(e: dict, number: int) -> str:
    ref = e.get("reference")
    url = e.get("url")

    if ref and url and ref != url:
        return f"[{number}] {e['source']} — {ref} — {url}"
    if ref:
        return f"[{number}] {e['source']} — {ref}"
    if url:
        return f"[{number}] {e['source']} — {url}"
    return f"[{number}] {e['source']}"


def make_llm_payload(question: str, evidence: list[dict], decision: dict) -> dict:
    evidence_lines = []

    for i, e in enumerate(evidence, 1):
        evidence_lines.append(
            "\n".join(
                [
                    f"[{i}] SOURCE: {e['source']}",
                    f"TYPE: {e['source_type']}",
                    f"REFERENCE: {e.get('reference') or 'غير متوفر'}",
                    f"URL: {e.get('url') or 'غير متوفر'}",
                    f"TEXT: {e['text']}",
                ]
            )
        )

    system_prompt = """أنت مساعد شرعي يعتمد حصراً على الأدلة المسترجعة المرفقة.

قواعد إلزامية:
1) لا تستخدم أي معلومة شرعية غير موجودة في الأدلة المرفقة.
2) لا تخترع آية أو حديثاً أو فتوى أو مرجعاً أو رابطاً.
3) اربط كل دعوى جوهرية برقم دليل مثل [1] أو [2].
4) إذا كانت الأدلة غير كافية، صرّح بعدم كفايتها ولا تكمل من المعرفة العامة.
5) إذا ظهرت أقوال لابن باز وابن عثيمين، لا تدمجها وكأنها قول واحد. انسب كل قول لصاحبه.
6) إذا ظهر اختلاف حقيقي بين الأدلة، اعرضه بوضوح ولا ترجّح من عندك.
7) النص القرآني والحديثي يُنقل من الدليل نفسه ولا يُعاد إنشاؤه من الذاكرة.
8) لا تجعل التفسير أو الفتوى بديلاً عن نص القرآن أو الحديث إذا كان النص موجوداً ضمن الأدلة.
9) الجواب يكون واضحاً ومختصراً ومباشراً.
10) عند الحاجة لفتوى شخصية أو حالة تتوقف على تفاصيل غير متاحة، اطلب الرجوع لمفتٍ مؤهل."""

    if not decision["can_answer"]:
        instruction = (
            "قرار طبقة الاستناد: لا تُصدر جواباً شرعياً حاسماً. "
            "اذكر أن الأدلة المسترجعة غير كافية للإجابة الموثوقة."
        )
    else:
        instruction = (
            "قرار طبقة الاستناد: يمكن صياغة جواب من الأدلة فقط، "
            "مع الاستشهاد بأرقامها."
        )

    if decision["scholar_comparison_required"]:
        instruction += (
            " توجد أدلة من أكثر من عالم؛ افصل الأقوال ولا تفترض الاتفاق."
        )

    user_prompt = (
        f"السؤال:\n{question}\n\n"
        f"{instruction}\n\n"
        "الأدلة المسترجعة:\n\n"
        + "\n\n".join(evidence_lines)
    )

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


def prepare_packet(
    question: str,
    raw_hits: list,
    retrieval_meta: dict,
    top_k: int,
) -> dict:
    normalized = []

    for rank, hit in enumerate(raw_hits[:top_k], 1):
        item = normalize_hit(hit, rank)
        if item["text"]:
            normalized.append(item)

    # Exact semantic duplicate guard.
    # Some sources can contain the same answer under more than one internal
    # corpus unit. Deduplicate by normalized full text + source, not by ID.
    deduped = []
    seen = set()
    for item in normalized:
        normalized_text = re.sub(r"\s+", " ", item["text"]).strip()
        fingerprint = hashlib.sha256(
            (item["source"] + "\n" + normalized_text).encode("utf-8")
        ).hexdigest()

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        deduped.append(item)

    for i, item in enumerate(deduped, 1):
        item["citation_id"] = i
        item["citation_label"] = citation_label(item, i)

    decision = evidence_decision(deduped)
    llm_payload = make_llm_payload(question, deduped, decision)

    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieval": retrieval_meta,
        "decision": decision,
        "evidence": deduped,
        "citation_map": {
            str(i): {
                "source": e["source"],
                "reference": e.get("reference"),
                "url": e.get("url"),
                "id": e.get("id"),
            }
            for i, e in enumerate(deduped, 1)
        },
        "llm_payload": llm_payload,
        "answer_contract": {
            "must_be_grounded": True,
            "must_cite_claims": True,
            "must_not_invent_sources": True,
            "must_separate_scholar_opinions": True,
            "must_abstain_if_decision_disallows_answer": True,
        },
    }


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^\w\u0600-\u06FF-]+", "_", text, flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:60] or "query"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a grounded answer packet from the unified retriever."
    )
    parser.add_argument("question", nargs="?", help="User question")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--retrieval-json",
        type=Path,
        help="Use an existing retrieval JSON instead of calling the retriever.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--inspect-retriever",
        action="store_true",
        help="Inspect the newest unified retriever public callables.",
    )
    args = parser.parse_args()

    if args.inspect_retriever:
        path = retriever_file()
        if path is None:
            print("No unified retriever file found.")
            sys.exit(1)
        module = import_module(path)
        print("=" * 100)
        print(f"RETRIEVER: {path.name}")
        print("=" * 100)
        for row in public_signatures(module):
            print(f"{row['type']:8} {row['name']}{row['signature']}")
        return

    question = clean_text(args.question)
    if not question:
        question = clean_text(input("اكتب السؤال: "))

    if not question:
        print("ERROR: Empty question.")
        sys.exit(1)

    if args.retrieval_json:
        if not args.retrieval_json.exists():
            print(f"ERROR: File not found: {args.retrieval_json}")
            sys.exit(1)
        raw_hits, retrieval_meta = load_retrieval_json(args.retrieval_json)
    else:
        try:
            raw_hits, retrieval_meta = run_retriever(question, args.top_k)
        except Exception as e:
            print("=" * 100)
            print("GROUNDING LAYER COULD NOT CALL THE RETRIEVER YET")
            print("=" * 100)
            print(str(e))
            print()
            print("Run this to inspect the retriever interface:")
            print(f'  python "{Path(__file__).name}" --inspect-retriever')
            sys.exit(2)

    packet = prepare_packet(
        question=question,
        raw_hits=raw_hits,
        retrieval_meta=retrieval_meta,
        top_k=args.top_k,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        out = args.output
        if not out.is_absolute():
            out = ROOT / out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"{stamp}_{safe_slug(question)}.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    d = packet["decision"]

    print("=" * 100)
    print("GROUNDED ANSWER PACKET READY")
    print("=" * 100)
    print(f"Question:                    {question}")
    print(f"Retriever:                   {retrieval_meta.get('retriever_file')}")
    print(f"Adapter:                     {retrieval_meta.get('adapter')}")
    print(f"Evidence items:              {len(packet['evidence'])}")
    print(f"Sources:                     {', '.join(d['sources']) or '-'}")
    print(f"Decision:                    {d['status']}")
    print(f"Can answer:                  {d['can_answer']}")
    print(f"Scholar comparison required: {d['scholar_comparison_required']}")
    print()
    print("CITATIONS")
    print("-" * 100)
    for e in packet["evidence"]:
        print(e["citation_label"])
    print()
    print(f"Output: {out}")
    print()
    print("NEXT: pass packet['llm_payload'] to the LLM. Do not send raw source folders to the LLM.")


if __name__ == "__main__":
    main()
