"""Stage 3.6: bounded open-access full-text retrieval and passage extraction."""
from __future__ import annotations

import asyncio
import re
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from evidence_extraction import COMPARATOR_TERMS, EFFECT_TERMS, RANKING_TERMS
from pico_coverage import outcome_present


EUROPE_PMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
MAX_ARTICLES = 4
MAX_RESPONSE_BYTES = 4_000_000
MAX_PASSAGES_PER_ARTICLE = 3


def _normalise(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_text(element: ET.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if part.strip()]


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _section_blocks(xml: bytes) -> list[tuple[str, str]]:
    # Europe PMC JATS documents commonly include a harmless DOCTYPE header.
    # Entity declarations remain forbidden because ElementTree must never expand
    # attacker-controlled XML entities.
    if b"<!ENTITY" in xml.upper():
        raise ValueError("Full-text XML entity declarations are not accepted")
    xml = re.sub(br"<!DOCTYPE[^>]*(?:\[[^\]]*\]\s*)?>", b"", xml, flags=re.IGNORECASE)
    root = ET.fromstring(xml)
    blocks: list[tuple[str, str]] = []
    for element in root.iter():
        kind = _local_name(element.tag)
        if kind not in {"sec", "table-wrap"}:
            continue
        title = ""
        for child in element:
            if _local_name(child.tag) in {"title", "label", "caption"}:
                title = _element_text(child)
                break
        text = _element_text(element)
        if text:
            blocks.append((title, text))
    return blocks or [("full text", _element_text(root))]


def extract_fulltext_passages(
    article: dict[str, Any],
    xml: bytes,
    requested_outcomes: list[str],
) -> list[dict[str, Any]]:
    """Extract short, auditable windows around comparative outcome evidence."""
    passages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section_title, block in _section_blocks(xml):
        sentences = _sentences(block)
        for index in range(len(sentences)):
            window = " ".join(sentences[max(0, index - 1): min(len(sentences), index + 2)])
            normalised = _normalise(window)
            outcomes = [outcome for outcome in requested_outcomes if outcome_present(normalised, outcome)]
            # Ranking tables often contain SUCRA/P-score values without an
            # effect verb; retain them only when a requested outcome is also present.
            if not outcomes or not (_has_any(normalised, EFFECT_TERMS) or _has_any(normalised, RANKING_TERMS)):
                continue
            if not (_has_any(normalised, COMPARATOR_TERMS) or _has_any(normalised, RANKING_TERMS)):
                continue
            key = _normalise(window)
            if key in seen:
                continue
            seen.add(key)
            passages.append({
                "article_id": article["article_id"],
                "title": article.get("title", ""),
                "url": f"https://europepmc.org/articles/{article['pmcid']}",
                "identifiers": {key: article.get(key) for key in ("doi", "pmid", "pmcid") if article.get(key)},
                "source_providers": ["europe_pmc_fulltext"],
                "passage": window,
                "section": section_title or "untitled section",
                "outcomes": outcomes,
                "effect_language_detected": [term for term in EFFECT_TERMS if term in normalised],
                "comparator_language_detected": [term for term in COMPARATOR_TERMS if term in normalised],
                "ranking_language_detected": [term for term in RANKING_TERMS if term in normalised],
                "source_scope": "open_access_full_text",
            })
            if len(passages) >= MAX_PASSAGES_PER_ARTICLE:
                break
        if len(passages) >= MAX_PASSAGES_PER_ARTICLE:
            break
    return passages


async def _fetch_xml(client: httpx.AsyncClient, pmcid: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async with client.stream("GET", EUROPE_PMC_FULLTEXT.format(pmcid=pmcid)) as response:
        if response.status_code == 404:
            raise FileNotFoundError("Open-access full text is unavailable")
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise RuntimeError("Full-text response exceeds the size limit")
            chunks.append(chunk)
    return b"".join(chunks)


async def enrich_with_open_access_fulltext(
    report: dict[str, Any],
    timeout_seconds: float = 12.0,
    priority_article_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Fetch up to four PMC full texts, then recompute Stage-3 evidence checks."""
    if report.get("stage") != "evidence_extraction":
        raise ValueError("Expected a Stage 3 evidence-extraction audit")
    if not 1 <= timeout_seconds <= 30:
        raise ValueError("Full-text timeout must be between 1 and 30 seconds")
    packet = report.get("evidence_packet", {})
    profile = report.get("pico_profile", {})
    candidates = [
        article for article in report.get("articles", [])
        if article.get("pico_screening", {}).get("status") == "direct_candidate"
        and re.fullmatch(r"PMC\d+", str(article.get("pmcid", "")))
    ]
    priority_article_ids = priority_article_ids or set()
    candidates.sort(key=lambda article: article["article_id"] not in priority_article_ids)
    candidates = candidates[:MAX_ARTICLES]
    attempts: list[dict[str, Any]] = []
    fulltext_passages: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), follow_redirects=False) as client:
        for article in candidates:
            try:
                xml = await _fetch_xml(client, article["pmcid"])
                extracted = extract_fulltext_passages(article, xml, profile.get("outcomes_requested", []))
                fulltext_passages.extend(extracted)
                attempts.append({"article_id": article["article_id"], "pmcid": article["pmcid"], "status": "ok", "passages_extracted": len(extracted)})
            except FileNotFoundError as exc:
                attempts.append({"article_id": article["article_id"], "pmcid": article["pmcid"], "status": "unavailable", "reason": str(exc)})
            except (httpx.HTTPError, RuntimeError, ET.ParseError, ValueError) as exc:
                attempts.append({"article_id": article["article_id"], "pmcid": article["pmcid"], "status": "error", "reason": type(exc).__name__})
    original_passages = packet.get("passages", [])
    all_passages = original_passages + fulltext_passages
    outcomes = {outcome for item in all_passages for outcome in item.get("outcomes", [])}
    requested = set(profile.get("outcomes_requested", []))
    comparative = [item for item in all_passages if item.get("comparator_language_detected")]
    ranking = [item for item in all_passages if item.get("ranking_language_detected")]
    checks = dict(packet.get("checks", {}))
    checks["outcome_specific_passages"] = {"pass": bool(all_passages) and requested.issubset(outcomes), "requested": sorted(requested), "found": sorted(outcomes), "missing": sorted(requested - outcomes)}
    checks["comparative_passages"] = {"pass": bool(comparative) if profile.get("comparison_requested") else True, "required": bool(profile.get("comparison_requested")), "observed": len(comparative)}
    checks["ranking_evidence"] = {"pass": bool(ranking) if profile.get("comparison_requested") else True, "required": bool(profile.get("comparison_requested")), "observed": len(ranking)}
    failed = [name for name, item in checks.items() if not item["pass"]]
    updated = dict(report)
    updated["stage"] = "fulltext_evidence"
    updated["evidence_packet"] = {
        **packet,
        "status": "READY_FOR_CLAIM_VERIFICATION" if not failed else "INCOMPLETE",
        "checks": checks,
        "failed_checks": failed,
        "passage_count": len(all_passages),
        "passages": all_passages,
        "fulltext_retrieval": {
            "source": "Europe PMC open-access fullTextXML",
            "candidate_articles_with_pmcid": len(candidates),
            "attempts": attempts,
            "fulltext_passage_count": len(fulltext_passages),
        },
    }
    updated["clinical_answer_allowed"] = False
    return updated
