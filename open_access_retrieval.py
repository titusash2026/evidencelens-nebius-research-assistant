"""Stage 3.7: targeted Europe PMC open-access evidence retrieval."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from evidence_extraction import build_evidence_packet
from evidence_hub import ProviderDataError, ProviderHTTPError, deduplicate_records, search_europe_pmc
from fulltext_evidence import enrich_with_open_access_fulltext
from pico_coverage import add_pico_and_coverage
from targeted_retrieval import _source_records


def build_open_access_query(profile: dict[str, Any]) -> str:
    ignored = {
        "older", "adult", "adults", "treatment", "treatments", "therapy", "therapies",
        "prevention", "prevent", "effective", "effectively", "fracture", "fractures",
        "hip", "vertebral", "which", "most", "reduce", "risk", "compare", "compared",
    }
    condition = " AND ".join(
        word for word in profile.get("domain_terms", []) if word not in ignored
    ) or "clinical condition"
    outcomes = profile.get("outcomes_requested", [])
    outcome_query = " OR ".join(f'"{outcome}"' for outcome in outcomes) or "fracture"
    return (
        f"({condition}) AND ({outcome_query}) AND "
        '("network meta-analysis" OR "network meta analysis" OR '
        '"systematic review" OR guideline) AND '
        '("older adults" OR elderly OR aged OR postmenopausal) AND OPEN_ACCESS:y'
    )


async def retrieve_open_access_evidence(
    report: dict[str, Any],
    limit: int = 20,
    budget_seconds: float = 12.0,
    fulltext_timeout_seconds: float = 12.0,
) -> dict[str, Any]:
    """Retrieve one auditable OA review/guideline page, then re-screen it."""
    if report.get("stage") not in {"evidence_extraction", "fulltext_evidence"}:
        raise ValueError("Expected a Stage 3 or Stage 3.6 evidence audit")
    if not 1 <= limit <= 30 or not 1 <= budget_seconds <= 30:
        raise ValueError("Open-access retrieval limit must be 1..30 and budget must be 1..30 seconds")
    question = report.get("research_question", "")
    profile = report.get("pico_profile", {})
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Evidence audit has no research question")
    articles = report.get("articles")
    if not isinstance(articles, list):
        raise ValueError("Evidence audit has no article list")
    query = build_open_access_query(profile)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(budget_seconds), follow_redirects=False,
            headers={"User-Agent": "EvidenceLens-Stage3.7/1.0"},
        ) as client:
            async with asyncio.timeout(budget_seconds):
                result = await search_europe_pmc(client, query, limit)
    except (TimeoutError, httpx.HTTPError, ProviderHTTPError, ProviderDataError,
            ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        updated = dict(report)
        updated["open_access_retrieval_round"] = {
            "query": query, "status": "error", "reason": type(exc).__name__,
        }
        updated["clinical_answer_allowed"] = False
        return updated
    raw_records = _source_records(articles) + result["records"]
    merged_articles, deduplication_audit = deduplicate_records(raw_records)
    rescreen_input = {
        "schema_version": report.get("schema_version", "1.0"), "stage": "retrieval_only",
        "query": report.get("query", ""), "created_at": report.get("created_at", ""),
        "elapsed_seconds": report.get("elapsed_seconds", 0), "connector_budget_seconds": budget_seconds,
        "per_provider_limit": limit, "retrieval_check": "PASS", "raw_record_count": len(raw_records),
        "unique_article_count": len(merged_articles), "duplicate_records_merged": len(raw_records) - len(merged_articles),
        "articles_with_abstract": sum(bool(article.get("abstract")) for article in merged_articles),
        "providers": report.get("providers", []), "deduplication_audit": deduplication_audit,
        "articles": merged_articles, "clinical_answer_allowed": False, "coverage_assessed": False,
    }
    rescreened = add_pico_and_coverage(rescreen_input, question)
    extracted = build_evidence_packet(rescreened)
    enriched = await enrich_with_open_access_fulltext(extracted, fulltext_timeout_seconds)
    enriched["open_access_retrieval_round"] = {
        "query": query, "status": "ok", "total_hits": result["total_hits"],
        "records_returned": len(result["records"]),
        "new_unique_articles_after_merge": len(merged_articles) - len(articles),
    }
    # Preserve earlier stages after the report is rebuilt and re-screened.
    if "targeted_retrieval_round" in report:
        enriched["targeted_retrieval_round"] = report["targeted_retrieval_round"]
    enriched["clinical_answer_allowed"] = False
    return enriched
