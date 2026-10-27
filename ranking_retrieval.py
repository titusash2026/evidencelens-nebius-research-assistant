"""Stage 3.8: open-access comparative-ranking retrieval."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from evidence_extraction import build_evidence_packet
from evidence_hub import ProviderDataError, ProviderHTTPError, deduplicate_records, search_europe_pmc
from fulltext_evidence import enrich_with_open_access_fulltext
from pico_coverage import add_pico_and_coverage
from targeted_retrieval import _source_records


def build_ranking_query(profile: dict[str, Any]) -> str:
    ignored = {"older", "adult", "adults", "treatment", "treatments", "therapy", "therapies", "prevention", "prevent", "effective", "effectively", "fracture", "fractures", "hip", "vertebral", "which", "most", "reduce", "risk", "compare", "compared"}
    condition = " AND ".join(word for word in profile.get("domain_terms", []) if word not in ignored) or "clinical condition"
    outcomes = profile.get("outcomes_requested", [])
    outcome_query = " AND ".join(f'"{outcome}"' for outcome in outcomes) or "fracture"
    return (
        f"({condition}) AND ({outcome_query}) AND "
        '("network meta-analysis" OR "network meta analysis" OR SUCRA OR '
        '"P-score" OR "league table" OR "head-to-head" OR "head to head") '
        "AND OPEN_ACCESS:y"
    )


async def retrieve_ranking_evidence(report: dict[str, Any], limit: int = 20, budget_seconds: float = 12.0, fulltext_timeout_seconds: float = 12.0) -> dict[str, Any]:
    if report.get("stage") not in {"evidence_extraction", "fulltext_evidence"}:
        raise ValueError("Expected a Stage 3 evidence audit")
    if not 1 <= limit <= 30 or not 1 <= budget_seconds <= 30:
        raise ValueError("Ranking retrieval limit must be 1..30 and budget must be 1..30 seconds")
    question, profile, articles = report.get("research_question", ""), report.get("pico_profile", {}), report.get("articles")
    if not isinstance(question, str) or not question.strip() or not isinstance(articles, list):
        raise ValueError("Evidence audit is missing its question or articles")
    query = build_ranking_query(profile)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(budget_seconds), follow_redirects=False, headers={"User-Agent": "EvidenceLens-Stage3.8/1.0"}) as client:
            async with asyncio.timeout(budget_seconds):
                result = await search_europe_pmc(client, query, limit)
    except (TimeoutError, httpx.HTTPError, ProviderHTTPError, ProviderDataError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        updated = dict(report)
        updated["ranking_retrieval_round"] = {"query": query, "status": "error", "reason": type(exc).__name__}
        updated["clinical_answer_allowed"] = False
        return updated
    raw_records = _source_records(articles) + result["records"]
    merged, audit = deduplicate_records(raw_records)
    base = {
        "schema_version": report.get("schema_version", "1.0"), "stage": "retrieval_only", "query": report.get("query", ""),
        "created_at": report.get("created_at", ""), "elapsed_seconds": report.get("elapsed_seconds", 0),
        "connector_budget_seconds": budget_seconds, "per_provider_limit": limit, "retrieval_check": "PASS",
        "raw_record_count": len(raw_records), "unique_article_count": len(merged), "duplicate_records_merged": len(raw_records) - len(merged),
        "articles_with_abstract": sum(bool(article.get("abstract")) for article in merged), "providers": report.get("providers", []),
        "deduplication_audit": audit, "articles": merged, "clinical_answer_allowed": False, "coverage_assessed": False,
    }
    rescreened = add_pico_and_coverage(base, question)
    ranking_markers = (
        "network meta analysis", "network meta-analysis", "sucra", "p score",
        "p-score", "league table", "head to head", "head-to-head",
    )
    requested_outcomes = set(profile.get("outcomes_requested", []))
    priority_article_ids: set[str] = set()
    new_pmcids = {str(record.get("pmcid", "")).upper() for record in result["records"]}
    for article in rescreened["articles"]:
        text = " ".join([article.get("title", ""), article.get("abstract", "")]).lower()
        has_both_outcomes = all(outcome in text for outcome in requested_outcomes)
        has_ranking_marker = any(marker in text for marker in ranking_markers)
        if article.get("pmcid") in new_pmcids and has_both_outcomes and has_ranking_marker:
            priority_article_ids.add(article["article_id"])
            screening = dict(article.get("pico_screening", {}))
            screening.update({
                "status": "direct_candidate",
                "reason": "Stage 3.8 priority candidate: open-access ranking study with both requested outcomes; full-text verification pending.",
                "stage_3_8_priority": True,
            })
            article["pico_screening"] = screening
    extracted = build_evidence_packet(rescreened)
    enriched = await enrich_with_open_access_fulltext(
        extracted, fulltext_timeout_seconds, priority_article_ids,
    )
    for field in ("targeted_retrieval_round", "open_access_retrieval_round"):
        if field in report:
            enriched[field] = report[field]
    enriched["ranking_retrieval_round"] = {
        "query": query, "status": "ok", "total_hits": result["total_hits"], "records_returned": len(result["records"]),
        "new_unique_articles_after_merge": len(merged) - len(articles),
        "priority_fulltext_article_ids": sorted(priority_article_ids),
    }
    enriched["clinical_answer_allowed"] = False
    return enriched
