"""Stage 3.9: one bounded retrieval round driven by Stage 4.2 coverage gaps.

This stage deliberately runs only after a verified claim set fails the
question-answer coverage gate.  It never turns a failed gate into an answer:
it retrieves, re-screens, extracts, and lets Stage 4 verify again.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from claim_verification import TREATMENT_ENTITY_TERMS
from evidence_extraction import build_evidence_packet
from evidence_hub import ProviderDataError, ProviderHTTPError, deduplicate_records, search_europe_pmc
from fulltext_evidence import enrich_with_open_access_fulltext
from pico_coverage import add_pico_and_coverage
from targeted_retrieval import _source_records


RANKING_TERMS = (
    "network meta-analysis", "network meta analysis", "sucra", "p-score",
    "p score", "league table", "head-to-head", "head to head", "comparative",
)
POPULATION_TERMS = ("older adults", "elderly", "aged", "postmenopausal")
IGNORED_DOMAIN_TERMS = {
    "older", "adult", "adults", "treatment", "treatments", "therapy", "therapies",
    "prevention", "prevent", "effective", "effectively", "fracture", "fractures",
    "hip", "vertebral", "which", "most", "reduce", "risk", "compare", "compared",
}


def build_coverage_gap_query(profile: dict[str, Any], failed_checks: list[str]) -> str:
    """Build one reproducible query which addresses the missing answer elements."""
    condition = " AND ".join(
        term for term in profile.get("domain_terms", []) if term not in IGNORED_DOMAIN_TERMS
    ) or "osteoporosis"
    outcomes = profile.get("outcomes_requested", [])
    # Both outcomes are deliberately required for a question that asks for both.
    outcome_query = " AND ".join(f'"{outcome}"' for outcome in outcomes) or "fracture"
    population_query = " OR ".join(f'"{term}"' for term in POPULATION_TERMS)
    treatments = " OR ".join(f'"{term}"' for term in TREATMENT_ENTITY_TERMS)
    study_designs = (
        '"network meta-analysis" OR "network meta analysis" OR "head-to-head" '
        'OR "head to head" OR guideline OR "systematic review"'
    )
    # Keep the exact gate failures in the audit even though the bounded query is
    # intentionally comprehensive: this makes the recovery decision reviewable.
    _ = failed_checks
    return (
        f"({condition}) AND ({outcome_query}) AND ({population_query}) AND "
        f"({treatments}) AND ({study_designs}) AND OPEN_ACCESS:y"
    )


def _priority_ids(
    articles: list[dict[str, Any]], new_pmcids: set[str], requested_outcomes: set[str],
) -> set[str]:
    """Prioritize full text likely to close all coverage gaps, without trusting it."""
    priority: set[str] = set()
    for article in articles:
        text = " ".join([article.get("title", ""), article.get("abstract", "")]).lower()
        has_outcomes = all(outcome in text for outcome in requested_outcomes)
        has_population = any(term in text for term in POPULATION_TERMS)
        has_treatment = any(term in text for term in TREATMENT_ENTITY_TERMS)
        has_comparison = any(term in text for term in RANKING_TERMS)
        if (
            str(article.get("pmcid", "")).upper() in new_pmcids
            and has_outcomes and has_population and has_treatment and has_comparison
        ):
            priority.add(article["article_id"])
            screening = dict(article.get("pico_screening", {}))
            screening.update({
                "status": "direct_candidate",
                "reason": "Stage 3.9 priority: OA source matches coverage-gap retrieval criteria; full-text verification pending.",
                "stage_3_9_priority": True,
            })
            article["pico_screening"] = screening
    return priority


async def retrieve_coverage_gap_evidence(
    report: dict[str, Any],
    limit: int = 20,
    budget_seconds: float = 12.0,
    fulltext_timeout_seconds: float = 12.0,
) -> dict[str, Any]:
    """Retrieve one OA round addressing Stage 4.2 failures, then rebuild Stage 3."""
    if report.get("stage") not in {"evidence_extraction", "fulltext_evidence"}:
        raise ValueError("Expected a Stage 3 evidence audit with Stage 4.2 coverage")
    if not 1 <= limit <= 30 or not 1 <= budget_seconds <= 30:
        raise ValueError("Coverage-gap retrieval limit and budget must be 1..30")
    verification = report.get("claim_verification", {})
    answer_coverage = verification.get("question_answer_coverage", {})
    failed_checks = answer_coverage.get("failed_checks", [])
    if not isinstance(failed_checks, list):
        raise ValueError("Stage 4.2 coverage gate has invalid failed checks")
    question = report.get("research_question", "")
    profile = report.get("pico_profile", {})
    articles = report.get("articles")
    if not isinstance(question, str) or not question.strip() or not isinstance(articles, list):
        raise ValueError("Coverage-gap retrieval requires a research question and articles")

    if not failed_checks:
        updated = dict(report)
        updated["coverage_gap_retrieval_round"] = {
            "attempted": False,
            "reason": "Stage 4.2 already covers the requested conclusion.",
            "failed_checks_triggering_retrieval": [],
            "new_unique_articles_after_merge": 0,
        }
        return updated

    query = build_coverage_gap_query(profile, failed_checks)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(budget_seconds), follow_redirects=False,
            headers={"User-Agent": "EvidenceLens-Stage3.9/1.0"},
        ) as client:
            async with asyncio.timeout(budget_seconds):
                result = await search_europe_pmc(client, query, limit)
    except (TimeoutError, httpx.HTTPError, ProviderHTTPError, ProviderDataError,
            ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        updated = dict(report)
        updated["coverage_gap_retrieval_round"] = {
            "attempted": True, "query": query, "status": "error",
            "reason": type(exc).__name__,
            "failed_checks_triggering_retrieval": failed_checks,
            "new_unique_articles_after_merge": 0,
        }
        updated["clinical_answer_allowed"] = False
        return updated

    raw_records = _source_records(articles) + result["records"]
    merged, audit = deduplicate_records(raw_records)
    rescreen_input = {
        "schema_version": report.get("schema_version", "1.0"), "stage": "retrieval_only",
        "query": report.get("query", ""), "created_at": report.get("created_at", ""),
        "elapsed_seconds": report.get("elapsed_seconds", 0), "connector_budget_seconds": budget_seconds,
        "per_provider_limit": limit, "retrieval_check": "PASS", "raw_record_count": len(raw_records),
        "unique_article_count": len(merged), "duplicate_records_merged": len(raw_records) - len(merged),
        "articles_with_abstract": sum(bool(article.get("abstract")) for article in merged),
        "providers": report.get("providers", []), "deduplication_audit": audit, "articles": merged,
        "clinical_answer_allowed": False, "coverage_assessed": False,
    }
    rescreened = add_pico_and_coverage(rescreen_input, question)
    requested_outcomes = set(profile.get("outcomes_requested", []))
    new_pmcids = {str(record.get("pmcid", "")).upper() for record in result["records"]}
    priority_article_ids = _priority_ids(rescreened["articles"], new_pmcids, requested_outcomes)
    extracted = build_evidence_packet(rescreened)
    enriched = await enrich_with_open_access_fulltext(
        extracted, fulltext_timeout_seconds, priority_article_ids,
    )
    for field in (
        "targeted_retrieval_round", "open_access_retrieval_round", "ranking_retrieval_round",
    ):
        if field in report:
            enriched[field] = report[field]
    enriched["coverage_gap_retrieval_round"] = {
        "attempted": True, "query": query, "status": "ok",
        "failed_checks_triggering_retrieval": failed_checks,
        "total_hits": result["total_hits"], "records_returned": len(result["records"]),
        "new_unique_articles_after_merge": len(merged) - len(articles),
        "priority_fulltext_article_ids": sorted(priority_article_ids),
    }
    enriched["clinical_answer_allowed"] = False
    return enriched
