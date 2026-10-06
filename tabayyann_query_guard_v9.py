from __future__ import annotations
import re
from typing import Any

_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = _AR_DIACRITICS.sub("", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[؟?!،,:;؛\"'()\[\]{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_leading_meta(q: str) -> str:
    """Remove non-semantic leading wrappers before ambiguity checks."""
    prefixes = (
        "للاختبار فقط",
        "بصياغه ثانيه",
        "بدون اي تخمين",
        "بدون تخمين",
        "احتاج تحقق دقيق",
        "من المصادر فقط",
        "لو سمحت",
        "بسرعه",
        "ابي اعرف",
        "سؤال سريع",
        "احتاج مساعده",
    )
    out = q
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if out == prefix:
                return ""
            if out.startswith(prefix + " "):
                out = out[len(prefix):].strip()
                changed = True
                break
    return out


def classify_query(question: str) -> dict[str, Any]:
    q = _norm(question)
    q_core = _strip_leading_meta(q)

    # ------------------------------------------------------------
    # 1) Prompt-injection / hallucination resistance
    # ------------------------------------------------------------
    # Explicit request to invent or use general memory when evidence is missing.
    # A request to "invent an approximate answer" is always insufficient evidence,
    # even when no concrete fiqh topic is supplied.
    if "جواب تقريبي" in q and "اخترع" in q:
        return {
            "action": "insufficient_evidence",
            "route": None,
            "reason": "approximate_fabrication_request",
        }

    if any(x in q for x in (
        "اخترع لي جواب",
        "اخترع جواب",
        "استخدم معلوماتك العامه",
        "استخدم معلوماتك العامة",
        "لو ما لقيت نص مباشر",
        "لو ما لقيت دليل",
        "لو ما لقيت مصدر",
        "اذا ما لقيت دليل",
        "إذا ما لقيت دليل",
    )):
        # If the user is only giving a meta-instruction with no concrete issue,
        # clarification is safer than hallucinating.
        if any(x in q for x in ("اخترع", "لا تقول ما ادري")) and not any(
            x in q for x in ("هل يجوز", "ما حكم", "وش حكم", "حديث", "فتوى", "فتوي")
        ):
            return {
                "action": "needs_clarification",
                "route": None,
                "reason": "hallucination_request_without_concrete_question",
            }
        return {
            "action": "insufficient_evidence",
            "route": None,
            "reason": "memory_fallback_request",
        }

    if "من عندك" in q and any(
        x in q for x in ("ما لقيت", "ما القيت", "لو ما", "بدون مصدر", "بدون مصادر")
    ):
        return {
            "action": "insufficient_evidence",
            "route": None,
            "reason": "memory_fallback_request",
        }

    # Do not let the user redefine weak similarity as sufficient evidence.
    if any(x in q for x in (
        "اعتبر اي نتيجه قريبه دليل كافي",
        "اعتبر اي نتيجة قريبة دليل كافي",
        "نتيجه قريبه دليل كافي",
        "نتيجة قريبة دليل كافي",
    )):
        return {
            "action": "insufficient_evidence",
            "route": None,
            "reason": "weak_similarity_must_not_be_treated_as_evidence",
        }

    # A meta instruction to answer outside the product scope must not bypass scope.
    if "خارج نطاقك" in q and any(x in q for x in ("حاول تجاوب", "جاوب", "اجب")):
        return {
            "action": "out_of_scope",
            "route": None,
            "reason": "explicit_scope_bypass_request",
        }

    # A request to guess missing context must remain a clarification case.
    if any(x in q for x in (
        "خمن قصدي",
        "لا تقل يحتاج توضيح",
        "لا تقول يحتاج توضيح",
    )):
        return {
            "action": "needs_clarification",
            "route": None,
            "reason": "explicit_guessing_request",
        }

    # Never fabricate a citation/reference merely to make an answer look sourced.
    if "اخترع" in q and any(x in q for x in ("مرجع", "مصدر", "توثيق")):
        return {
            "action": "insufficient_evidence",
            "route": None,
            "reason": "fabricated_citation_request",
        }

    # Never substitute an invented/similar hadith when the requested one is not found.
    if (
        "حديث" in q
        and any(x in q for x in ("لو ما لقيت", "اذا ما لقيت"))
        and any(x in q for x in ("اكتب حديث مشابه", "حديث مشابه", "اختلق حديث", "اخترع حديث"))
    ):
        return {
            "action": "not_found",
            "route": None,
            "reason": "fabricated_hadith_substitution_request",
        }

    # Requests to fabricate a specific authority/reference even if unavailable.
    if any(x in q for x in (
        "حتي لو ما كانت موجوده",
        "حتي لو ما كانت موجودة",
        "حتي لو مو متاكد",
        "حتي لو مو متأكد",
    )):
        return {
            "action": "not_found",
            "route": None,
            "reason": "requested_fabricated_reference",
        }

    # If the user asks us to choose arbitrarily when sources differ,
    # do not choose; surface a conflict/policy state.
    if "المصادر اختلفت" in q and any(x in q for x in ("من نفسك", "اختار لي", "اختر لي")):
        return {
            "action": "conflict",
            "route": None,
            "reason": "must_not_arbitrarily_choose_between_conflicting_sources",
        }

    # Claims of universal scholarly consensus need evidence before we
    # challenge the claim. Retrieve the two approved fatwa sources first,
    # then let the caller surface a conflict/qualification state with citations.
    if "باتفاق العلماء" in q:
        return {
            "action": "conflict_with_evidence",
            "route": "fatwa",
            "reason": "universal_consensus_claim_requires_broader_evidence",
        }

    # ------------------------------------------------------------
    # 2) Missing-context / deictic questions
    # ------------------------------------------------------------
    exact_ambiguous = {
        "وش الحكم", "ما الحكم", "الحكم", "هل يجوز", "يجوز",
        "هل هذا صحيح", "هل الكلام صحيح", "وش رايك", "ما رايك",
        "هل هذا حرام", "وش الحكم فيها", "اقدر اسويه ولا لا",
        "هل لازم اعيدها", "صار مني شيء وش اسوي", "هل يجوز كذا للمراه",
        "انا سويتها بالغلط علي شيء", "وش حكمها وقت الدوره",
        "يصير قبلها ولا بعدها", "هل يكفي كذا",
        "هذا يجوز ولا ممنوع", "علي كفاره",
        "ينفع بهالحاله", "هل تعتبر صحيحه",
        "متي اسويها", "اذا نسيت وش الحكم", "هل علي اعاده",
    }

    unresolved_refs = (
        "هذا الدعاء", "هذا الكلام", "هذه المساله", "هذي المساله",
        "هذا الشي", "هذه الفتوي", "هذي الفتوي",
        "كذا", "فيها", "اعيدها", "اسويه", "سويتها",
        "قبلها ولا بعدها", "وقت الدوره",
        "بهالحاله", "تعتبر صحيحه", "علي كفاره",
    )

    if q_core in exact_ambiguous:
        return {
            "action": "needs_clarification",
            "route": None,
            "reason": "missing_context",
        }

    # Only treat deictic words as ambiguous when there is no concrete topic.
    concrete_topic_cues = (
        "صيام", "رمضان", "الصلاه", "الوضوء", "الزكاه", "المصحف",
        "القران", "الحائض", "الوتر", "الاضحيه", "السفر", "المحرم",
        "العطر", "الاظافر", "الشعر", "حديث", "البخاري", "دعاء بعد",
    )
    if any(x in q_core for x in unresolved_refs) and not any(x in q_core for x in concrete_topic_cues):
        return {
            "action": "needs_clarification",
            "route": None,
            "reason": "unresolved_reference",
        }

    # ------------------------------------------------------------
    # 3) Clear out-of-scope requests
    # ------------------------------------------------------------
    out_patterns = (
        r"\bبايثون\b", r"\bpython\b", r"\bكود\b", r"\bبرمجه\b",
        r"\bدواء\b", r"\bعلاج\b", r"\bالصداع النصفي\b",
        r"\bمباراه اليوم\b", r"\bمن فاز\b", r"\bنتيجه المباراه\b",
        r"\bسعر الذهب\b", r"\bالذهب اليوم\b",
        r"\bجوال\b", r"\bاشتريه\b",
        r"\bسيره ذاتيه\b", r"\bسيرة ذاتية\b",
        r"\bباسورد الراوتر\b", r"\bالراوتر\b",
        r"\bحاله الطقس\b", r"\bحالة الطقس\b", r"\bالطقس\b",
        r"\bمعدلي الجامعي\b", r"\bاحسب لي معدلي\b", r"\bالمعدل الجامعي\b",
        r"\bجدول مذاكره\b", r"\bجدول مذاكرة\b",
        r"\bالواي فاي\b", r"\bواي فاي\b", r"\bwifi\b", r"\bwi-fi\b",
        r"\bافضل مطعم\b", r"\bمطعم قريب\b",
        r"\bاحجز(?: لي)? فندق\b", r"\bحجز فندق\b",
        r"\bسعر الدولار\b", r"\bالدولار اليوم\b",
        r"\bافضل لابتوب\b", r"\bلابتوب للالعاب\b",
        r"\bترجم لي\b", r"\bترجمه .* للانجليزي\b",
    )
    if any(re.search(p, q) for p in out_patterns):
        return {
            "action": "out_of_scope",
            "route": None,
            "reason": "outside_islamic_scope",
        }

    # ------------------------------------------------------------
    # 4) High-risk unsupported ritual/number claims
    # ------------------------------------------------------------
    if (
        ("سوره معينه" in q or "سورة معينة" in q)
        and any(x in q for x in ("سبع مرات", "7 مرات", "تحقق", "الامنيات", "الأمنيات"))
    ):
        return {
            "action": "insufficient_evidence",
            "route": None,
            "reason": "specific_ritual_merit_requires_direct_evidence",
        }

    if "دعاء" in q and "بعدد معين" in q and any(x in q for x in ("لازم يستجاب", "يستجاب")):
        return {
            "action": "insufficient_evidence",
            "route": None,
            "reason": "guaranteed_numbered_dua_claim_requires_direct_evidence",
        }

    if "قص الشعر" in q and "ايام معينه" in q and "حرام" in q:
        return {
            "action": "insufficient_evidence",
            "route": None,
            "reason": "calendar_based_hair_cutting_claim_requires_direct_evidence",
        }

    # ------------------------------------------------------------
    # 5) Verification / source routing
    # ------------------------------------------------------------
    verification = (
        "هل صحيح", "يقولون", "يقال", "منتشر", "تحقق",
        "هل هذا ثابت", "هل هو ثابت", "ثابت",
    )
    if any(x in q for x in verification):
        # Verification questions should still use the most appropriate approved
        # source family. Fiqh claims go to Ibn Baz/Ibn Uthaymeen instead of
        # allowing a superficially related Quran result to dominate.
        verification_fiqh = (
            "الحائض", "الحيض", "الدوره", "الدورة",
            "الوضوء", "ينقض الوضوء", "الصيام", "صيام",
            "الصلاه", "الصلاة", "الزكاه", "الزكاة",
            "زكاه الفطر", "زكاة الفطر", "المحرم",
            "الجوارب", "العطر", "الكحول",
        )
        if any(x in q for x in verification_fiqh):
            return {
                "action": "retrieve",
                "route": "fatwa",
                "reason": "verification_fiqh",
            }

        if "البخاري" in q or "حديث" in q:
            return {
                "action": "retrieve",
                "route": "bukhari",
                "reason": "verification_hadith",
            }

        if any(x in q for x in ("تفسير", "قوله تعالي", "معني قوله تعالي")):
            return {
                "action": "retrieve",
                "route": "quran_tafsir",
                "reason": "verification_quran_tafsir",
            }

        return {
            "action": "retrieve",
            "route": "general",
            "reason": "verification",
        }

    if "البخاري" in q or "حديث" in q:
        return {
            "action": "retrieve",
            "route": "bukhari",
            "reason": "hadith",
        }

    if any(x in q for x in ("تفسير", "فسر", "فسري", "معني قوله تعالي")):
        return {
            "action": "retrieve",
            "route": "tafsir",
            "reason": "tafsir",
        }

    if any(x in q for x in ("ما الايه", "ايه التي", "قوله تعالي")):
        return {
            "action": "retrieve",
            "route": "quran",
            "reason": "quran",
        }

    # "ما معنى ..." / "وش معنى ..." on this product is commonly a verse
    # interpretation request. Search Quran + Ibn Kathir together rather than
    # letting unrelated fatwa text dominate lexical retrieval.
    if "معني " in q:
        return {
            "action": "retrieve",
            "route": "quran_tafsir",
            "reason": "meaning_query_quran_tafsir_priority",
        }

    fiqh = (
        "حكم", "هل يجوز", "يجوز", "عادي", "وضوء", "المصحف", "صيام",
        "قضاء رمضان", "زكاه", "الصلاه", "الجمع بين الصلاتين", "اضحي",
        "تضحي", "الجوارب", "الاظافر",
    )
    if any(x in q for x in fiqh):
        return {
            "action": "retrieve",
            "route": "fatwa",
            "reason": "fiqh",
        }

    return {
        "action": "retrieve",
        "route": "general",
        "reason": "general",
    }

def _dedupe_hits(hits: list[Any], limit: int) -> list[Any]:
    out, seen = [], set()
    for hit in hits:
        key = None
        for attr in ("global_unit_id", "unit_id", "global_index"):
            value = getattr(hit, attr, None)
            if value is not None:
                key = (attr, str(value))
                break
        if key is None:
            key = ("object", id(hit))
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def _interleave(a: list[Any], b: list[Any], limit: int) -> list[Any]:
    merged = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            merged.append(a[i])
        if i < len(b):
            merged.append(b[i])
    return _dedupe_hits(merged, limit)


def retrieve_hits(retriever: Any, question: str, route: str, *, top_k: int = 8) -> list[Any]:
    route = route or "general"
    if route == "bukhari":
        return retriever.search(question, top_k=top_k, source_filter="bukhari", max_per_source=top_k, auto_source_filter=False)
    if route == "tafsir":
        return retriever.search(question, top_k=top_k, source_filter="ibn_kathir", max_per_source=top_k, auto_source_filter=False)
    if route == "quran":
        return retriever.search(question, top_k=top_k, source_filter="quran", max_per_source=top_k, auto_source_filter=False)
    if route == "quran_tafsir":
        each = max(3, (top_k + 1) // 2)
        quran = retriever.search(
            question,
            top_k=each,
            source_filter="quran",
            max_per_source=each,
            auto_source_filter=False,
        )
        tafsir = retriever.search(
            question,
            top_k=each,
            source_filter="ibn_kathir",
            max_per_source=each,
            auto_source_filter=False,
        )
        return _interleave(list(tafsir), list(quran), top_k)
    if route == "fatwa":
        each = max(3, (top_k + 1) // 2)
        baz = retriever.search(question, top_k=each, source_filter="ibn_baz", max_per_source=each, auto_source_filter=False)
        uth = retriever.search(question, top_k=each, source_filter="ibn_uthaymeen", max_per_source=each, auto_source_filter=False)
        return _interleave(list(baz), list(uth), top_k)
    return retriever.search(question, top_k=top_k, max_per_source=3, auto_source_filter=True)
