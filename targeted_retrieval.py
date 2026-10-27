"""Stage 3.5: one transparent, bounded follow-up retrieval round."""
from __future__ import annotations

import asyncio
from typing import Any

from evidence_hub import deduplicate_records, run_hub
from evidence_extraction import build_evidence_packet
from pico_coverage import add_pico_and_coverage


def _source_records(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover original provider records so deduplication is rerun safely."""
    records: list[dict[str, Any]] = []
    for article in articles:
        source_records = article.get("source_records")
        if not isinstance(source_records, list):
            raise ValueError("An article is missing its preserved source records")
        records.extend(source_records)
    return records


async def run_targeted_retrieval(
    evidence_report: dict[str, Any],
    limit: int = 15,
    budget_seconds: float = 8.0,
) -> dict[str, Any]:
    """Run at most one controlled targeted-retrieval round, then rescreen.

    The suggested queries were recorded in the Stage 3 JSON before any network
    call. This avoids hidden retries and prevents an unbounded agent loop.
    """
    if evidence_report.get("stage") != "evidence_extraction":
        raise ValueError("Expected a Stage 3 evidence-extraction audit")
    question = evidence_report.get("research_question", "")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Stage 3 audit has no research question")

    packet = evidence_report.get("evidence_packet", {})
    plans = packet.get("targeted_retrieval_plan", [])
    if not isinstance(plans, list):
        raise ValueError("Stage 3 audit has an invalid targeted retrieval plan")
    # One follow-up round gives a predictable latency and audit trail.
    plans = plans[:1]

    if not plans:
        updated = dict(evidence_report)
        updated["targeted_retrieval_round"] = {
            "attempted": False,
            "maximum_queries": 1,
            "runs": [],
            "new_unique_articles_after_merge": 0,
            "previous_evidence_packet_status": packet.get("status"),
        }
        return updated

    original_articles = evidence_report.get("articles")
    if not isinstance(original_articles, list):
        raise ValueError("Stage 3 audit has no article list")
    raw_records = _source_records(original_articles)
    round_audit: list[dict[str, Any]] = []

    for plan in plans:
        query = plan.get("query") if isinstance(plan, dict) else None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Targeted retrieval plan has no valid query")
        result = await run_hub(query, limit=limit, budget_seconds=budget_seconds)
        newly_retrieved = _source_records(result["articles"])
        raw_records.extend(newly_retrieved)
        round_audit.append({
            "reason": plan.get("reason", ""),
            "query": query,
            "retrieval_check": result["retrieval_check"],
            "raw_records_returned": result["raw_record_count"],
            "unique_articles_returned": result["unique_article_count"],
            "providers": result["providers"],
        })

    merged_articles, deduplication_audit = deduplicate_records(raw_records)
    all_followups_passed = bool(round_audit) and all(
        item["retrieval_check"] == "PASS" for item in round_audit
    )

    # Build a fresh Stage-1-shaped report, then re-run PICO and extraction.
    rescreen_input = {
        "schema_version": evidence_report.get("schema_version", "1.0"),
        "stage": "retrieval_only",
        "query": evidence_report.get("query", ""),
        "created_at": evidence_report.get("created_at", ""),
        "elapsed_seconds": evidence_report.get("elapsed_seconds", 0),
        "connector_budget_seconds": budget_seconds,
        "per_provider_limit": limit,
        "retrieval_check": "PASS" if all_followups_passed else "REVIEW",
        "raw_record_count": len(raw_records),
        "unique_article_count": len(merged_articles),
        "duplicate_records_merged": len(raw_records) - len(merged_articles),
        "articles_with_abstract": sum(bool(article.get("abstract")) for article in merged_articles),
        "providers": evidence_report.get("providers", []),
        "deduplication_audit": deduplication_audit,
        "articles": merged_articles,
        "clinical_answer_allowed": False,
        "coverage_assessed": False,
    }

    rescreened = add_pico_and_coverage(rescreen_input, question)
    updated = build_evidence_packet(rescreened)
    updated["targeted_retrieval_round"] = {
        "attempted": True,
        "maximum_queries": 1,
        "runs": round_audit,
        "new_unique_articles_after_merge": len(merged_articles) - len(original_articles),
        "previous_evidence_packet_status": packet.get("status"),
    }
    updated["clinical_answer_allowed"] = False
    return updated
