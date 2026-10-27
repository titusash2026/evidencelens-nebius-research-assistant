"""Stage 3: extract auditable, outcome-specific evidence passages.

This stage deliberately extracts from the retrieved title/abstract only.  It
does not infer effect sizes, rank treatments, or generate a clinical answer.
"""
from __future__ import annotations

import re
from typing import Any

from pico_coverage import outcome_present


EFFECT_TERMS = (
    "reduced", "reduce", "reduction", "prevented", "prevent", "lower", "decrease",
    "decreased", "decrease", "risk", "incidence", "hazard ratio", "relative risk",
    "odds ratio", "effective", "efficacy",
)
COMPARATOR_TERMS = (
    "compared with", "compared to", "versus", " vs ", "placebo", "control group",
    "usual care", "head to head", "head-to-head",
)
RANKING_TERMS = (
    "network meta analysis", "network meta-analysis", "network meta", "ranked",
    "ranking", "most effective", "superior to", "head to head", "head-to-head",
    "sucra", "p score", "p-score", "league table", "surface under the cumulative ranking",
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _sentences(text: str) -> list[str]:
    """Split conservatively and retain the original source wording for audit."""
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned) if part.strip()]


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _passage_from_sentence(article: dict[str, Any], sentence: str, requested_outcomes: list[str]) -> dict[str, Any] | None:
    normalised = _normalise(sentence)
    outcome_matches = [outcome for outcome in requested_outcomes if outcome_present(normalised, outcome)]
    # A numeric/effect phrase is required so a background sentence about fracture
    # epidemiology is never offered as proof of a treatment effect.
    if not outcome_matches or not _has_any(normalised, EFFECT_TERMS):
        return None
    return {
        "article_id": article["article_id"],
        "title": article.get("title", ""),
        "url": article.get("url", "") or next(iter(article.get("urls", [])), ""),
        "identifiers": {
            key: article.get(key, "") for key in ("doi", "pmid", "pmcid") if article.get(key)
        },
        "source_providers": article.get("providers", [article.get("provider", "")]),
        "passage": sentence,
        "outcomes": outcome_matches,
        "effect_language_detected": [term for term in EFFECT_TERMS if term in normalised],
        "comparator_language_detected": [term for term in COMPARATOR_TERMS if term in normalised],
        "ranking_language_detected": [term for term in RANKING_TERMS if term in normalised],
        "source_scope": "title_or_abstract_only",
    }


def extract_evidence_passages(articles: list[dict[str, Any]], requested_outcomes: list[str]) -> list[dict[str, Any]]:
    """Return maximum two evidence passages per direct PICO candidate article."""
    passages: list[dict[str, Any]] = []
    for article in articles:
        if article.get("pico_screening", {}).get("status") != "direct_candidate":
            continue
        selected = 0
        for sentence in _sentences(article.get("abstract", "")):
            item = _passage_from_sentence(article, sentence, requested_outcomes)
            if item:
                passages.append(item)
                selected += 1
            if selected == 2:
                break
    return passages


def targeted_retrieval_plan(profile: dict[str, Any], coverage: dict[str, Any]) -> list[dict[str, str]]:
    """Create query suggestions; it does not silently run more searches."""
    condition_terms = [term for term in profile.get("domain_terms", []) if term not in {
        "older", "adult", "adults", "treatment", "treatments", "therapy", "therapies",
        "prevention", "prevent", "effective", "effectively", "fracture", "fractures",
        "hip", "vertebral", "which", "most", "reduce", "risk", "compare", "compared",
    }]
    condition = " AND ".join(condition_terms[:3]) or "clinical condition"
    outcomes = profile.get("outcomes_requested", [])
    outcome_query = " OR ".join(f'"{outcome}"' for outcome in outcomes) or "clinical outcomes"
    plans: list[dict[str, str]] = []
    for failed in coverage.get("failed_checks", []):
        if failed == "comparative_evidence":
            plans.append({
                "reason": "Comparative evidence was not established by retrieved candidate abstracts.",
                "query": f'({condition}) AND ({outcome_query}) AND ("network meta-analysis" OR "systematic review" OR "head-to-head" OR comparative)',
            })
        elif failed == "requested_outcomes":
            missing = coverage["checks"]["requested_outcomes"].get("missing", [])
            plans.append({
                "reason": "At least one requested outcome has no direct candidate coverage.",
                "query": f'({condition}) AND ({" OR ".join(f"\"{item}\"" for item in missing)}) AND (trial OR "systematic review")',
            })
        elif failed == "direct_candidate_articles":
            plans.append({
                "reason": "Too few direct candidates were retrieved.",
                "query": f'({condition}) AND ({outcome_query}) AND (trial OR "systematic review" OR guideline)',
            })
    return plans


def build_evidence_packet(pico_report: dict[str, Any]) -> dict[str, Any]:
    """Add extractive evidence and release controls to a Stage 2 audit."""
    if pico_report.get("stage") != "pico_and_coverage":
        raise ValueError("Expected a Stage 2 PICO-and-coverage audit")
    profile = pico_report.get("pico_profile", {})
    coverage = pico_report.get("coverage", {})
    articles = pico_report.get("articles")
    if not isinstance(articles, list):
        raise ValueError("PICO audit has no article list")
    passages = extract_evidence_passages(articles, profile.get("outcomes_requested", []))
    passage_outcomes = {outcome for item in passages for outcome in item["outcomes"]}
    requested_outcomes = set(profile.get("outcomes_requested", []))
    comparative_passages = [item for item in passages if item["comparator_language_detected"]]
    ranking_passages = [item for item in passages if item["ranking_language_detected"]]
    missing = sorted(requested_outcomes - passage_outcomes)
    checks = {
        "upstream_coverage_ready": {
            "pass": coverage.get("coverage_status") == "READY_FOR_EVIDENCE_EXTRACTION",
            "observed": coverage.get("coverage_status"),
        },
        "outcome_specific_passages": {
            "pass": bool(passages) and not missing,
            "requested": sorted(requested_outcomes), "found": sorted(passage_outcomes), "missing": missing,
        },
        "comparative_passages": {
            "pass": bool(comparative_passages) if profile.get("comparison_requested") else True,
            "required": bool(profile.get("comparison_requested")), "observed": len(comparative_passages),
        },
        # A question asking "most effective" needs comparative ranking evidence,
        # not merely separate treatment-vs-placebo trials.
        "ranking_evidence": {
            "pass": bool(ranking_passages) if profile.get("comparison_requested") else True,
            "required": bool(profile.get("comparison_requested")), "observed": len(ranking_passages),
        },
    }
    failed = [name for name, value in checks.items() if not value["pass"]]
    updated = dict(pico_report)
    updated["stage"] = "evidence_extraction"
    updated["evidence_packet"] = {
        "status": "READY_FOR_CLAIM_VERIFICATION" if not failed else "INCOMPLETE",
        "checks": checks,
        "failed_checks": failed,
        "passage_count": len(passages),
        "passages": passages,
        "targeted_retrieval_plan": targeted_retrieval_plan(profile, coverage),
        "limitations": [
            "Passages are extracted from retrieved titles and abstracts; full text and source tables have not been reviewed.",
            "An extracted passage is a citation candidate, not a verified clinical claim.",
            "No clinical answer is released at Stage 3. Stage 4 must verify each proposed claim against these passages and source context.",
        ],
    }
    updated["clinical_answer_allowed"] = False
    return updated
