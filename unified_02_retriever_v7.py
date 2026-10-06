from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("Install it with: pip install sentence-transformers")
    sys.exit(1)


# ======================================================================================
# UNIFIED RETRIEVER V7 — ACTIVE FIVE SOURCES
#
# Query -> common E5 dense search + global BM25 -> Reciprocal Rank Fusion (RRF)
#       -> optional explicit-source filter -> parent dedup -> source diversification
#
# IMPORTANT:
# - Does NOT modify any source data.
# - Does NOT merge old source-specific raw scores.
# - Uses a common ranking layer across the already-built unified catalog.
# - Arabic normalization is retrieval-only; displayed/stored source text is untouched.
# ======================================================================================


ROOT = Path(__file__).resolve().parent
UNIFIED_DIR = ROOT / "unified_rag"

UNITS_PATH = UNIFIED_DIR / "unified_units.jsonl"
EMBEDDINGS_PATH = UNIFIED_DIR / "unified_embeddings.npy"
BUILD_VALIDATION_PATH = UNIFIED_DIR / "unified_build_validation.json"

MODEL_NAME = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "

RRF_K = 60
DENSE_WEIGHT = 1.00
BM25_WEIGHT = 0.85

DENSE_CANDIDATES = 250
BM25_CANDIDATES = 250

DEFAULT_TOP_K = 10
DEFAULT_MAX_PER_SOURCE = 3

BM25_K1 = 1.5
BM25_B = 0.75

EXACT_QUERY_SUBSTRING_BONUS = 0.020
TITLE_TOKEN_COVERAGE_BONUS = 0.012
FOCUSED_EXACT_SUBSTRING_BONUS = 0.030
QURAN_TO_TAFSIR_BRIDGE_BONUS = 0.065
QURAN_EXACT_ANCHOR_BONUS = 0.020
QURAN_ANCHOR_MIN_BM25 = 4.0
QURAN_EXACT_PHRASE_ANCHOR_BONUS = 0.080

# Soft source priors used only when the query intent is clear.
INTENT_SOURCE_BONUSES = {
    "tafsir": {
        # Tafsir queries are handled by the structural Quran→Tafsir bridge below.
        # Do not blanket-boost every Ibn Kathir/Quran unit.
    },
    "hadith": {
        "bukhari": 0.024,
    },
    "fatwa": {
        "ibn_baz": 0.012,
        "ibn_uthaymeen": 0.012,
    },
    "quran": {
        "quran": 0.024,
        "ibn_kathir": 0.010,
    },
}

SOURCE_KEYS = {
    "quran",
    "bukhari",
    "ibn_baz",
    "ibn_uthaymeen",
    "ibn_kathir",
}

SOURCE_LABELS = {
    "quran": "Qur'an",
    "bukhari": "Sahih al-Bukhari",
    "ibn_baz": "Ibn Baz",
    "ibn_uthaymeen": "Ibn Uthaymeen",
    "ibn_kathir": "Tafsir Ibn Kathir",
}

SOURCE_ALIASES = {
    "quran": [
        "القران",
        "القرآن",
        "المصحف",
    ],
    "bukhari": [
        "البخاري",
        "صحيح البخاري",
    ],
    "ibn_baz": [
        "ابن باز",
        "بن باز",
    ],
    "ibn_uthaymeen": [
        "ابن عثيمين",
        "ابن العثيمين",
        "العثيمين",
    ],
    "ibn_kathir": [
        "ابن كثير",
        "تفسير ابن كثير",
    ],
}

AR_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
NON_WORD = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)


@dataclass
class SearchResult:
    rank: int
    global_index: int
    global_unit_id: str
    source: str
    source_name: str
    source_type: str
    unit_id: str
    parent_id: str | None
    title: str | None
    reference: str | None
    url: str | None
    text: str
    fused_score: float
    dense_score: float
    dense_rank: int | None
    bm25_score: float
    bm25_rank: int | None
    metadata: dict[str, Any]


def normalize_ar(text: str) -> str:
    text = str(text or "")
    text = AR_DIACRITICS.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
        .replace("ـ", "")
    )
    text = NON_WORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return [tok for tok in normalize_ar(text).split() if len(tok) > 1]


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"{path}: line {line_no} is not a JSON object."
                )

            rows.append(obj)

    return rows


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=int)

    k = min(k, len(scores))

    if k <= 0:
        return np.array([], dtype=int)

    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def detect_explicit_source(query: str) -> str | None:
    nq = normalize_ar(query)

    matches = []

    for source, aliases in SOURCE_ALIASES.items():
        for alias in aliases:
            na = normalize_ar(alias)
            if na and na in nq:
                matches.append((len(na), source))

    if not matches:
        return None

    # Most specific alias wins.
    matches.sort(reverse=True)
    return matches[0][1]


def detect_query_intent(query: str) -> str | None:
    nq = normalize_ar(query)

    tafsir_markers = [
        "تفسير قوله تعالى",
        "ما تفسير قوله تعالى",
        "ما معنى قوله تعالى",
        "تفسير الايه",
        "تفسير الاية",
        "تفسير ايه",
        "تفسير اية",
        "ما تفسير ايه",
        "ما تفسير اية",
        "معنى الايه",
        "معنى الاية",
        "معنى ايه",
        "معنى اية",
        "اشرح قوله تعالى",
    ]
    if any(normalize_ar(marker) in nq for marker in tafsir_markers):
        return "tafsir"

    hadith_markers = [
        "حديث",
        "قال النبي",
        "قال رسول الله",
        "صحيح البخاري",
        "ما صحة الحديث",
    ]
    if any(normalize_ar(marker) in nq for marker in hadith_markers):
        return "hadith"

    fatwa_markers = [
        "ما حكم",
        "حكم",
        "هل يجوز",
        "يجوز",
        "فتوى",
        "حلال",
        "حرام",
    ]
    if any(normalize_ar(marker) in nq for marker in fatwa_markers):
        return "fatwa"

    quran_markers = [
        "في القران",
        "في القرآن",
        "اية",
        "آية",
        "سورة",
    ]
    if any(normalize_ar(marker) in nq for marker in quran_markers):
        return "quran"

    return None


def focused_query_text(query: str) -> str:
    # Remove common question/source scaffolding from lexical matching so the
    # actual ayah/topic phrase carries BM25 and exact-substring signals.
    nq = normalize_ar(query)

    prefixes = [
        "ما تفسير قوله تعالى",
        "تفسير قوله تعالى",
        "ما معنى قوله تعالى",
        "معنى قوله تعالى",
        "اشرح قوله تعالى",
        "ما تفسير الايه",
        "ما تفسير الاية",
        "ما تفسير ايه",
        "ما تفسير اية",
        "تفسير الايه",
        "تفسير الاية",
        "تفسير ايه",
        "تفسير اية",
        "ما معنى الايه",
        "ما معنى الاية",
        "ما معنى ايه",
        "ما معنى اية",
        "معنى الايه",
        "معنى الاية",
        "معنى ايه",
        "معنى اية",
        "ما نص الايه",
        "ما نص الاية",
        "ما نص ايه",
        "ما نص اية",
        "نص الايه",
        "نص الاية",
        "نص ايه",
        "نص اية",
        "ما حكم",
        "ما هو حكم",
        "ما صحة حديث",
        "ما صحة الحديث",
    ]

    for phrase in prefixes:
        np = normalize_ar(phrase)
        if nq.startswith(np):
            nq = nq[len(np):].strip()
            break

    # Remove trailing punctuation BEFORE checking source-location suffixes.
    nq = nq.rstrip("؟?!.,،؛;: ")

    suffixes = [
        "في القران",
        "من القران",
        "بالقران",
        "في المصحف",
        "من المصحف",
    ]

    changed = True
    while changed:
        changed = False

        nq = nq.rstrip("؟?!.,،؛;: ")

        for phrase in suffixes:
            np = normalize_ar(phrase)

            if nq.endswith(np):
                nq = nq[:-len(np)].strip()
                nq = nq.rstrip("؟?!.,،؛;: ")
                changed = True
                break

    return nq or normalize_ar(query).rstrip("؟?!.,،؛;: ")


def lexical_document(unit: dict) -> str:
    metadata = unit.get("metadata") or {}

    parts = [
        unit.get("title"),
        unit.get("reference"),
        metadata.get("category"),
        metadata.get("subcategory"),
        metadata.get("book_name"),
        metadata.get("chapter_name"),
        metadata.get("scope"),
        " ".join(metadata.get("scope_topics") or [])
        if isinstance(metadata.get("scope_topics"), list)
        else metadata.get("scope_topics"),
        unit.get("text"),
    ]

    return "\n".join(
        str(x).strip()
        for x in parts
        if x is not None and str(x).strip()
    )


def build_bm25(corpus_tokens: list[list[str]]):
    n_docs = len(corpus_tokens)
    doc_lens = np.asarray(
        [len(tokens) for tokens in corpus_tokens],
        dtype=np.float32,
    )
    avgdl = float(doc_lens.mean()) if n_docs else 1.0

    postings = defaultdict(list)
    df = Counter()

    for doc_id, tokens in enumerate(corpus_tokens):
        tf = Counter(tokens)

        for term, freq in tf.items():
            df[term] += 1
            postings[term].append((doc_id, freq))

    idf = {
        term: math.log(
            1.0 + (n_docs - freq + 0.5) / (freq + 0.5)
        )
        for term, freq in df.items()
    }

    return doc_lens, avgdl, postings, idf


def bm25_scores(
    query_tokens: list[str],
    n_docs: int,
    doc_lens: np.ndarray,
    avgdl: float,
    postings,
    idf,
) -> np.ndarray:
    scores = np.zeros(n_docs, dtype=np.float32)

    if not query_tokens:
        return scores

    for term, qfreq in Counter(query_tokens).items():
        posting = postings.get(term)

        if not posting:
            continue

        term_idf = idf[term]

        for doc_id, tf in posting:
            denom = tf + BM25_K1 * (
                1.0
                - BM25_B
                + BM25_B
                * float(doc_lens[doc_id])
                / max(avgdl, 1e-9)
            )

            score = (
                term_idf
                * (tf * (BM25_K1 + 1.0))
                / denom
            )

            scores[doc_id] += float(
                score * (1.0 + 0.10 * (qfreq - 1))
            )

    return scores


def rrf_contribution(rank: int, weight: float) -> float:
    return weight / (RRF_K + rank)


def exact_substring_bonus(query: str, unit: dict) -> float:
    nq = normalize_ar(query)
    focused = focused_query_text(query)

    doc = normalize_ar(
        "\n".join(
            [
                str(unit.get("title") or ""),
                str(unit.get("text") or ""),
            ]
        )
    )

    bonus = 0.0

    if len(nq) >= 8 and nq in doc:
        bonus += EXACT_QUERY_SUBSTRING_BONUS

    if (
        focused
        and focused != nq
        and len(focused) >= 6
        and focused in doc
    ):
        bonus += FOCUSED_EXACT_SUBSTRING_BONUS

    return bonus


def title_coverage_bonus(
    query_tokens: list[str],
    unit: dict,
) -> float:
    title = unit.get("title")

    if not title:
        return 0.0

    qset = set(query_tokens)

    if not qset:
        return 0.0

    title_set = set(tokenize(title))

    if not title_set:
        return 0.0

    coverage = len(qset & title_set) / max(1, len(qset))

    return TITLE_TOKEN_COVERAGE_BONUS * coverage


def dedup_key(unit: dict) -> str:
    source = str(unit.get("source") or "")
    parent = unit.get("parent_id")

    if parent is not None and str(parent).strip():
        return f"{source}::parent::{parent}"

    # For Quran, each ayah/unit is naturally its own parent.
    return str(unit.get("global_unit_id") or unit.get("global_index"))


def unit_ayah_keys(unit: dict) -> set[str]:
    keys: set[str] = set()
    metadata = unit.get("metadata") or {}

    for field in (
        "ayah_key",
        "primary_ayah_key",
        "ayah_keys",
        "matched_scope_ayah_keys",
    ):
        value = metadata.get(field)

        if isinstance(value, list):
            for item in value:
                s = str(item or "").strip()
                if s:
                    keys.add(s)
        elif value is not None:
            s = str(value).strip()
            if s:
                keys.add(s)

    unit_id = str(unit.get("unit_id") or "").strip()
    if unit.get("source") == "quran" and unit_id.startswith("quran:"):
        keys.add(unit_id.split("quran:", 1)[1])

    return keys


def preferred_sources_for_intent(intent: str | None) -> tuple[str, ...]:
    if intent == "tafsir":
        return ("ibn_kathir", "quran")
    if intent == "hadith":
        return ("bukhari",)
    if intent == "fatwa":
        return ("ibn_baz", "ibn_uthaymeen")
    if intent == "quran":
        return ("quran", "ibn_kathir")
    return ()


class UnifiedRetriever:
    def __init__(self) -> None:
        for required in (
            UNITS_PATH,
            EMBEDDINGS_PATH,
            BUILD_VALIDATION_PATH,
        ):
            if not required.exists():
                raise FileNotFoundError(
                    f"Missing required unified artifact: {required}"
                )

        validation = json.loads(
            BUILD_VALIDATION_PATH.read_text(encoding="utf-8")
        )

        if validation.get("status") != "PASS":
            raise RuntimeError(
                "Unified build validation is not PASS."
            )

        print("Loading unified units...")
        self.units = load_jsonl(UNITS_PATH)

        print("Loading unified embeddings...")
        self.embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r",
        )

        if self.embeddings.ndim != 2:
            raise RuntimeError(
                f"Expected 2D embeddings, got {self.embeddings.shape}"
            )

        if len(self.units) != self.embeddings.shape[0]:
            raise RuntimeError(
                "Unified unit/embedding count mismatch: "
                f"{len(self.units)} vs {self.embeddings.shape[0]}"
            )

        self.sources = np.asarray(
            [str(u.get("source") or "") for u in self.units],
            dtype=object,
        )

        self.source_indices = {
            source: np.flatnonzero(self.sources == source)
            for source in SOURCE_KEYS
        }

        self.ibn_kathir_by_ayah: dict[str, list[int]] = defaultdict(list)
        for idx in self.source_indices.get("ibn_kathir", []):
            i = int(idx)
            for ayah_key in unit_ayah_keys(self.units[i]):
                self.ibn_kathir_by_ayah[ayah_key].append(i)

        # Fast exact-phrase Quran anchoring for queries that quote part of an ayah.
        self.quran_normalized_text: dict[int, str] = {}
        for idx in self.source_indices.get("quran", []):
            i = int(idx)
            self.quran_normalized_text[i] = normalize_ar(
                self.units[i].get("text") or ""
            )

        print("Building global BM25 index in memory...")
        self.lexical_docs = [
            lexical_document(unit)
            for unit in self.units
        ]
        self.corpus_tokens = [
            tokenize(doc)
            for doc in self.lexical_docs
        ]
        (
            self.doc_lens,
            self.avgdl,
            self.postings,
            self.idf,
        ) = build_bm25(self.corpus_tokens)

        print(f"Loading query model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        print(
            f"READY: {len(self.units)} units, "
            f"{self.embeddings.shape[1]} dimensions, "
            f"{len(self.idf)} lexical terms."
        )
        print()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source_filter: str | None = None,
        max_per_source: int = DEFAULT_MAX_PER_SOURCE,
        auto_source_filter: bool = True,
    ) -> list[SearchResult]:
        query = str(query or "").strip()

        if not query:
            return []

        explicit_source = None

        if source_filter is not None:
            source_filter = source_filter.strip()

            if source_filter not in SOURCE_KEYS:
                raise ValueError(
                    "Invalid source_filter. "
                    f"Allowed: {sorted(SOURCE_KEYS)}"
                )

            explicit_source = source_filter

        elif auto_source_filter:
            explicit_source = detect_explicit_source(query)

        query_intent = detect_query_intent(query)
        focused_text = focused_query_text(query)

        qvec = self.model.encode(
            [QUERY_PREFIX + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        dense_scores = np.asarray(
            self.embeddings @ qvec,
            dtype=np.float32,
        )

        query_tokens = tokenize(focused_text)

        lexical_scores = bm25_scores(
            query_tokens,
            len(self.units),
            self.doc_lens,
            self.avgdl,
            self.postings,
            self.idf,
        )

        allowed_mask = np.ones(
            len(self.units),
            dtype=bool,
        )

        if explicit_source is not None:
            allowed_mask = self.sources == explicit_source

        allowed_indices = np.flatnonzero(allowed_mask)

        if len(allowed_indices) == 0:
            return []

        # Dense candidates inside the allowed source universe.
        allowed_dense = dense_scores[allowed_indices]
        dense_local_idx = top_indices(
            allowed_dense,
            min(DENSE_CANDIDATES, len(allowed_indices)),
        )
        dense_idx = allowed_indices[dense_local_idx]

        # BM25 candidates inside the same allowed universe.
        allowed_bm25 = lexical_scores[allowed_indices]
        bm25_local_idx = top_indices(
            allowed_bm25,
            min(BM25_CANDIDATES, len(allowed_indices)),
        )
        bm25_idx = allowed_indices[bm25_local_idx]

        fused: dict[int, float] = defaultdict(float)
        dense_rank_by_idx: dict[int, int] = {}
        bm25_rank_by_idx: dict[int, int] = {}

        for rank, idx in enumerate(dense_idx, start=1):
            i = int(idx)
            dense_rank_by_idx[i] = rank
            fused[i] += rrf_contribution(
                rank,
                DENSE_WEIGHT,
            )

        for rank, idx in enumerate(bm25_idx, start=1):
            i = int(idx)

            # Skip zero-score tail documents; otherwise arbitrary BM25 zeros
            # could contribute rank points.
            if float(lexical_scores[i]) <= 0.0:
                continue

            bm25_rank_by_idx[i] = rank
            fused[i] += rrf_contribution(
                rank,
                BM25_WEIGHT,
            )

        # Make sure clear query intents are represented in the candidate pool,
        # even when another larger source dominates the global top-N.
        if explicit_source is None and query_intent in {"hadith", "fatwa"}:
            for preferred_source in preferred_sources_for_intent(query_intent):
                src_idx = self.source_indices.get(preferred_source)
                if src_idx is None or len(src_idx) == 0:
                    continue

                dense_local = top_indices(
                    dense_scores[src_idx],
                    min(40, len(src_idx)),
                )
                for local_rank, pos in enumerate(dense_local, start=1):
                    i = int(src_idx[int(pos)])
                    if i not in dense_rank_by_idx:
                        dense_rank_by_idx[i] = DENSE_CANDIDATES + local_rank
                    fused[i] += 0.35 * rrf_contribution(
                        local_rank,
                        DENSE_WEIGHT,
                    )

                bm25_local = top_indices(
                    lexical_scores[src_idx],
                    min(40, len(src_idx)),
                )
                for local_rank, pos in enumerate(bm25_local, start=1):
                    i = int(src_idx[int(pos)])
                    if float(lexical_scores[i]) <= 0.0:
                        continue
                    if i not in bm25_rank_by_idx:
                        bm25_rank_by_idx[i] = BM25_CANDIDATES + local_rank
                    fused[i] += 0.35 * rrf_contribution(
                        local_rank,
                        BM25_WEIGHT,
                    )

        # Tafsir bridge:
        # If a Quran verse is the strongest lexical anchor for an explicit tafsir
        # question, boost Ibn Kathir chunks that are structurally linked to that ayah.
        quran_anchor_key = None
        quran_anchor_bm25 = 0.0

        if explicit_source is None and query_intent == "tafsir":
            quran_idx = self.source_indices.get("quran")

            if quran_idx is not None and len(quran_idx):
                quran_scores = lexical_scores[quran_idx]
                best_local = int(np.argmax(quran_scores))
                best_global = int(quran_idx[best_local])
                best_score = float(lexical_scores[best_global])

                if best_score >= QURAN_ANCHOR_MIN_BM25:
                    keys = unit_ayah_keys(self.units[best_global])
                    if keys:
                        quran_anchor_key = sorted(keys)[0]
                        quran_anchor_bm25 = best_score

                        # Keep the exact Quran verse near the top for citation/context.
                        fused[best_global] += QURAN_EXACT_ANCHOR_BONUS

                        # Boost only Ibn Kathir chunks structurally linked to that verse.
                        for i in self.ibn_kathir_by_ayah.get(
                            quran_anchor_key,
                            [],
                        ):
                            fused[i] += QURAN_TO_TAFSIR_BRIDGE_BONUS

        # Exact Quran phrase anchoring:
        # If the user explicitly asks for Quran text / quotes a verse fragment,
        # any Quran unit containing that focused phrase is forcibly included and
        # receives a strong lexical anchor. This is deterministic text matching,
        # not semantic score tuning.
        focused_norm = focused_query_text(query)

        if (
            explicit_source == "quran"
            or query_intent == "quran"
        ) and len(focused_norm) >= 6:
            for i, qtext in self.quran_normalized_text.items():
                if focused_norm in qtext:
                    fused[i] += QURAN_EXACT_PHRASE_ANCHOR_BONUS

        # Add small retrieval-safe bonuses after rank fusion.
        for i in list(fused.keys()):
            unit = self.units[i]

            fused[i] += exact_substring_bonus(
                query,
                unit,
            )

            fused[i] += title_coverage_bonus(
                query_tokens,
                unit,
            )

            if query_intent is not None:
                source = str(unit.get("source") or "")
                fused[i] += INTENT_SOURCE_BONUSES.get(
                    query_intent,
                    {},
                ).get(source, 0.0)

        ranked_indices = sorted(
            fused,
            key=lambda i: (
                fused[i],
                float(dense_scores[i]),
                float(lexical_scores[i]),
            ),
            reverse=True,
        )

        results = []
        seen_parents = set()
        per_source = Counter()

        # If user explicitly requested a source, do not artificially cap it at 3.
        effective_source_cap = (
            max(top_k, max_per_source)
            if explicit_source is not None
            else max_per_source
        )

        for idx in ranked_indices:
            unit = self.units[idx]
            source = str(unit.get("source") or "")

            key = dedup_key(unit)

            if key in seen_parents:
                continue

            if per_source[source] >= effective_source_cap:
                continue

            seen_parents.add(key)
            per_source[source] += 1

            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    global_index=int(unit.get("global_index", idx)),
                    global_unit_id=str(
                        unit.get("global_unit_id") or ""
                    ),
                    source=source,
                    source_name=str(
                        unit.get("source_name")
                        or SOURCE_LABELS.get(source, source)
                    ),
                    source_type=str(
                        unit.get("source_type") or ""
                    ),
                    unit_id=str(unit.get("unit_id") or ""),
                    parent_id=(
                        str(unit.get("parent_id"))
                        if unit.get("parent_id") is not None
                        else None
                    ),
                    title=(
                        str(unit.get("title"))
                        if unit.get("title") is not None
                        else None
                    ),
                    reference=(
                        str(unit.get("reference"))
                        if unit.get("reference") is not None
                        else None
                    ),
                    url=(
                        str(unit.get("url"))
                        if unit.get("url") is not None
                        else None
                    ),
                    text=str(unit.get("text") or ""),
                    fused_score=float(fused[idx]),
                    dense_score=float(dense_scores[idx]),
                    dense_rank=dense_rank_by_idx.get(idx),
                    bm25_score=float(lexical_scores[idx]),
                    bm25_rank=bm25_rank_by_idx.get(idx),
                    metadata=unit.get("metadata") or {},
                )
            )

            if len(results) >= top_k:
                break

        return results


def print_results(
    query: str,
    results: list[SearchResult],
) -> None:
    print()
    print("=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    if not results:
        print("No results.")
        return

    for result in results:
        print()
        print(
            f"[{result.rank}] "
            f"{result.source_name} "
            f"| unit={result.unit_id}"
        )

        if result.title:
            print(f"    title:     {result.title}")

        if result.reference:
            print(f"    reference: {result.reference}")

        if result.url:
            print(f"    url:       {result.url}")

        print(
            f"    fused:    {result.fused_score:.6f} "
            f"| dense: {result.dense_score:.6f} "
            f"(rank={result.dense_rank}) "
            f"| bm25: {result.bm25_score:.6f} "
            f"(rank={result.bm25_rank})"
        )

        preview = re.sub(
            r"\s+",
            " ",
            result.text,
        ).strip()

        if len(preview) > 550:
            preview = preview[:550].rstrip() + "..."

        print(f"    text:      {preview}")

    print()
    print("=" * 100)


def result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "global_index": result.global_index,
        "global_unit_id": result.global_unit_id,
        "source": result.source,
        "source_name": result.source_name,
        "source_type": result.source_type,
        "unit_id": result.unit_id,
        "parent_id": result.parent_id,
        "title": result.title,
        "reference": result.reference,
        "url": result.url,
        "text": result.text,
        "score": result.fused_score,
        "dense_score": result.dense_score,
        "dense_rank": result.dense_rank,
        "bm25_score": result.bm25_score,
        "bm25_rank": result.bm25_rank,
        "metadata": result.metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified five-source Islamic RAG retriever V7."
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Optional one-shot query. If omitted, interactive mode starts.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--source",
        choices=sorted(SOURCE_KEYS),
        default=None,
        help="Optional hard source filter.",
    )

    parser.add_argument(
        "--max-per-source",
        type=int,
        default=DEFAULT_MAX_PER_SOURCE,
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON for one-shot mode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("UNIFIED RETRIEVER V7 — ACTIVE FIVE")
    print("Dense E5 + Focused BM25 + RRF + Robust Routing + Exact Quran Phrase Anchors V2")
    print("=" * 100)

    retriever = UnifiedRetriever()

    if args.query:
        results = retriever.search(
            args.query,
            top_k=max(1, args.top_k),
            source_filter=args.source,
            max_per_source=max(1, args.max_per_source),
        )

        if args.json:
            payload = {
                "query": args.query,
                "source_filter": args.source,
                "results": [
                    result_to_dict(r)
                    for r in results
                ],
            }
            print(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print_results(
                args.query,
                results,
            )

        return

    print("Interactive mode.")
    print("Type a question and press Enter.")
    print("Type 'exit' to close.")
    print()

    while True:
        try:
            query = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "q",
            "خروج",
        }:
            break

        try:
            results = retriever.search(
                query,
                top_k=max(1, args.top_k),
                source_filter=args.source,
                max_per_source=max(1, args.max_per_source),
            )

            print_results(
                query,
                results,
            )

        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
