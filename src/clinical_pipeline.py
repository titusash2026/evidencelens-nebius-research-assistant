"""Streamlit bridge for EvidenceLens' clinically gated retrieval route.

This module keeps the clinical pipeline separate from the web/Tavily research
workflow.  It orchestrates the bounded stages already implemented in the root
modules and returns one JSON-serialisable audit for the Streamlit UI.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from claim_verification import build_evidence_gap_report, run_claim_verification
from coverage_gap_retrieval import retrieve_coverage_gap_evidence
from evidence_extraction import build_evidence_packet
from evidence_hub import run_hub
from fulltext_evidence import enrich_with_open_access_fulltext
from open_access_retrieval import retrieve_open_access_evidence
from pico_coverage import add_pico_and_coverage
from ranking_retrieval import retrieve_ranking_evidence
from targeted_retrieval import run_targeted_retrieval


ProgressCallback = Callable[[str], None]


class ClinicalPipelineError(RuntimeError):
    """A safe, user-displayable error from the clinical evidence route."""


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _validate(question: str, query: str, limit: int, budget_seconds: float) -> None:
    if not question.strip():
        raise ClinicalPipelineError("Enter a clinical research question before running the pipeline.")
    if not query.strip():
        raise ClinicalPipelineError("Enter a PubMed/Europe PMC search query before running the pipeline.")
    if len(question) > 2_000 or len(query) > 1_500:
        raise ClinicalPipelineError("The question or search query is too long for this bounded retrieval run.")
    if not 1 <= limit <= 30:
        raise ClinicalPipelineError("Records per source must be between 1 and 30.")
    if not 1 <= budget_seconds <= 30:
        raise ClinicalPipelineError("Connector budget must be between 1 and 30 seconds.")


def run_clinical_evidence_pipeline(
    *,
    question: str,
    query: str,
    limit: int = 15,
    budget_seconds: float = 12.0,
    model_timeout_seconds: float = 90.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run Stages 1–4.3 once, including the single allowed recovery retrieval.

    A Stage 3.9 round is permitted only after Stage 4.2 identifies answer
    coverage gaps.  Even then, a clinical conclusion remains blocked unless the
    re-verified claim set covers the user's requested conclusion.
    """
    question, query = question.strip(), query.strip()
    _validate(question, query, limit, budget_seconds)
    started_at = time.perf_counter()
    fulltext_timeout = min(30.0, max(8.0, budget_seconds))

    try:
        _emit(progress, "Stage 1/4.3 · retrieving PubMed and Europe PMC records")
        report = asyncio.run(run_hub(query, limit, budget_seconds))

        _emit(progress, "Stage 2/4.3 · screening candidate records against the question")
        report = add_pico_and_coverage(report, question)

        _emit(progress, "Stage 3 · extracting outcome-specific evidence passages")
        report = build_evidence_packet(report)

        _emit(progress, "Stage 3.5 · running one comparative follow-up search")
        report = asyncio.run(run_targeted_retrieval(report, limit, budget_seconds))

        _emit(progress, "Stage 3.6 · checking open-access full text")
        report = asyncio.run(enrich_with_open_access_fulltext(report, fulltext_timeout))

        _emit(progress, "Stage 3.7 · retrieving open-access reviews and guidelines")
        report = asyncio.run(
            retrieve_open_access_evidence(report, min(20, limit + 5), budget_seconds, fulltext_timeout)
        )

        _emit(progress, "Stage 3.8 · retrieving comparative and ranking evidence")
        report = asyncio.run(
            retrieve_ranking_evidence(report, min(20, limit + 5), budget_seconds, fulltext_timeout)
        )

        _emit(progress, "Stage 4 · drafting and verifying source-grounded claims with Nemotron")
        report = asyncio.run(run_claim_verification(report, model_timeout_seconds))
        initial_verification = report.get("claim_verification", {})
        initial_coverage = initial_verification.get("question_answer_coverage", {})

        # One auditable recovery attempt only.  A model/API failure is not a
        # coverage failure and must never trigger arbitrary further retrieval.
        if initial_verification.get("status") != "VERIFIED":
            report["coverage_gap_retrieval_round"] = {
                "attempted": False,
                "reason": "Stage 3.9 was not run because Stage 4 claim verification did not complete.",
                "verification_status": initial_verification.get("status", "NOT_RUN"),
                "failed_checks_triggering_retrieval": [],
                "new_unique_articles_after_merge": 0,
            }
        elif initial_coverage.get("answer_ready"):
            report["coverage_gap_retrieval_round"] = {
                "attempted": False,
                "reason": "Stage 4.2 already covers the requested conclusion.",
                "failed_checks_triggering_retrieval": [],
                "new_unique_articles_after_merge": 0,
            }
        else:
            _emit(progress, "Stage 3.9 · retrieving evidence targeted to verified coverage gaps")
            report = asyncio.run(
                retrieve_coverage_gap_evidence(report, min(20, limit + 5), budget_seconds, fulltext_timeout)
            )
            if report.get("coverage_gap_retrieval_round", {}).get("status") == "ok":
                report["initial_claim_verification"] = initial_verification
                _emit(progress, "Stage 4.3 · re-verifying the rebuilt evidence packet once")
                report = asyncio.run(run_claim_verification(report, model_timeout_seconds))

        report["evidence_gap_report"] = build_evidence_gap_report(report)
        report["clinical_answer_allowed"] = bool(report.get("clinical_answer_allowed"))
        report["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        report["streamlit_workflow"] = "clinical_evidence_pipeline"
        return report
    except ValueError as error:
        raise ClinicalPipelineError(str(error)) from error
    except RuntimeError as error:
        raise ClinicalPipelineError(f"Clinical evidence pipeline stopped safely: {error}") from error


def save_clinical_audit(report: dict[str, Any], output_dir: Path = Path("outputs")) -> Path:
    """Persist one UI run without overwriting a previous evidence audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir / f"clinical_evidence_audit_{stamp}_{uuid4().hex[:8]}.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    return destination


def clinical_stage_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact, display-safe stage outcomes for the Streamlit table."""
    coverage = report.get("coverage", {})
    packet = report.get("evidence_packet", {})
    verification = report.get("claim_verification", {})
    answer_coverage = verification.get("question_answer_coverage", {})
    gap_round = report.get("coverage_gap_retrieval_round", {})
    gap_report = report.get("evidence_gap_report", {})
    return [
        {"stage": "1 · Retrieval", "status": report.get("retrieval_check", "NOT_RUN"), "detail": f"{report.get('unique_article_count', 0)} unique articles"},
        {"stage": "2 · PICO screening", "status": coverage.get("coverage_status", "NOT_RUN"), "detail": f"{coverage.get('direct_candidate_count', 0)} direct candidates"},
        {"stage": "3 · Evidence extraction", "status": packet.get("status", "NOT_RUN"), "detail": f"{packet.get('passage_count', 0)} passages"},
        {"stage": "3.9 · Gap retrieval", "status": "ATTEMPTED" if gap_round.get("attempted") else "NOT_REQUIRED / NOT_RUN", "detail": gap_round.get("reason", ", ".join(gap_round.get("failed_checks_triggering_retrieval", [])) or "No gaps targeted")},
        {"stage": "4 · Claim verification", "status": verification.get("status", "NOT_RUN"), "detail": "Nemotron called" if verification.get("model_called") else verification.get("reason", "Not called")},
        {"stage": "4.2 · Answer coverage", "status": answer_coverage.get("status", "NOT_RUN"), "detail": ", ".join(answer_coverage.get("failed_checks", [])) or "All requested elements covered"},
        {"stage": "4.3 · Release decision", "status": gap_report.get("status", "NOT_RUN"), "detail": "Clinical answer released" if gap_report.get("answer_released") else gap_report.get("safe_display_text", "Clinical conclusion withheld")},
    ]


def build_clinical_report_markdown(report: dict[str, Any]) -> str:
    """Create a download that never promotes a blocked conclusion as an answer."""
    verification = report.get("claim_verification", {})
    gap_report = report.get("evidence_gap_report", {})
    lines = [
        "# EvidenceLens Clinical Evidence Audit",
        "",
        f"**Research question:** {report.get('research_question', 'Not recorded')}",
        f"**Search query:** {report.get('query', 'Not recorded')}",
        f"**Clinical answer release:** {'ALLOWED' if report.get('clinical_answer_allowed') else 'BLOCKED'}",
        "",
        "## Release decision",
        gap_report.get("message", "No release decision was recorded."),
        "",
    ]
    if report.get("clinical_answer_allowed"):
        lines.append("## Verified, question-covered evidence statements")
        for item in verification.get("verified_evidence_statements", []):
            lines.append(f"- {item.get('claim', '').strip()} ({', '.join(item.get('evidence_ids', []))})")
    else:
        lines.extend([
            "## Evidence gap",
            gap_report.get("safe_display_text", "No clinical conclusion can be displayed."),
            "",
            "Missing answer elements: " + (", ".join(gap_report.get("missing_answer_elements", [])) or "not recorded"),
        ])
    lines.extend([
        "",
        "## Safety note",
        "This audit is a retrieval and claim-verification aid, not patient-specific medical advice or a substitute for clinical expert review.",
    ])
    return "\n".join(lines) + "\n"
