"""Transparent quality ranking and provider-diversity selection."""

from __future__ import annotations

from copy import deepcopy
import re
from urllib.parse import urlparse


DOMAIN_AUTHORITY_SCORES = {
    "pubmed.ncbi.nlm.nih.gov": 45,
    "pmc.ncbi.nlm.nih.gov": 45,
    "ncbi.nlm.nih.gov": 43,
    "who.int": 48,
    "nih.gov": 48,
    "cdc.gov": 47,
    "uspstf.org": 47,
    "nice.org.uk": 47,
    "cochranelibrary.com": 46,
    "nature.com": 45,
    "thelancet.com": 45,
    "nejm.org": 45,
    "jamanetwork.com": 45,
    "diabetesjournals.org": 44,
    "rsna.org": 42,
    "radiologyinfo.org": 40,
}


DOCUMENT_TYPE_SCORES = {
    "Systematic review or meta-analysis": (16, "high-level evidence synthesis"),
    "Randomized controlled trial": (15, "randomized trial evidence"),
    "Clinical guideline": (14, "clinical guideline evidence"),
    "Observational cohort study": (10, "observational study evidence"),
    "Modelling study": (4, "modelling evidence"),
    "Evidence briefing or device record": (2, "evidence briefing"),
    "Journal literature record": (6, "journal literature record"),
    "Unclassified evidence record": (0, "unclassified evidence type"),
}


def _normalise_domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def source_family(url: str) -> str:
    """Group related websites so PubMed and PMC are not treated as independent."""
    domain = _normalise_domain(url)

    if domain.endswith("ncbi.nlm.nih.gov"):
        return "NCBI (PubMed/PMC)"
    if domain.endswith("who.int"):
        return "World Health Organization"
    if domain.endswith("nih.gov"):
        return "National Institutes of Health"
    if domain.endswith("cdc.gov"):
        return "Centers for Disease Control"
    if domain.endswith("uspstf.org"):
        return "USPSTF"
    if domain.endswith("nice.org.uk"):
        return "NICE"
    if domain.endswith("cochranelibrary.com"):
        return "Cochrane"
    if domain.endswith("nature.com"):
        return "Nature Portfolio"
    if domain.endswith("thelancet.com"):
        return "The Lancet"
    if domain.endswith("nejm.org"):
        return "NEJM"
    if domain.endswith("jamanetwork.com"):
        return "JAMA Network"
    if domain.endswith("diabetesjournals.org"):
        return "American Diabetes Association"
    if domain.endswith("rsna.org"):
        return "RSNA"
    if domain.endswith("radiologyinfo.org"):
        return "RadiologyInfo"

    return domain or "Unknown provider"


def _authority_score(url: str) -> tuple[int, str]:
    domain = _normalise_domain(url)

    for trusted_domain, score in DOMAIN_AUTHORITY_SCORES.items():
        if domain == trusted_domain or domain.endswith(f".{trusted_domain}"):
            return score, f"trusted provider: {source_family(url)}"

    return 8, "provider is not in the trusted-source list"


def _tokenise(text: str) -> set[str]:
    ignored_words = {
        "the", "and", "for", "with", "that", "this", "from",
        "what", "are", "was", "were", "into", "about", "evidence",
    }

    return {
        word.lower()
        for word in re.findall(r"[A-Za-z]{3,}", text or "")
        if word.lower() not in ignored_words
    }


def _question_match_score(source: dict, question: str) -> tuple[int, str]:
    question_tokens = _tokenise(question)
    source_text = " ".join(
        [
            str(source.get("title", "")),
            str(source.get("content", "")),
            str(source.get("raw_content", "")),
        ]
    )
    source_tokens = _tokenise(source_text)

    if not question_tokens:
        return 0, "no question-keyword match available"

    overlap_ratio = len(question_tokens & source_tokens) / len(question_tokens)
    score = round(min(overlap_ratio, 1.0) * 20)

    if score >= 15:
        return score, "strong match with the research question"
    if score >= 8:
        return score, "moderate match with the research question"
    return score, "limited match with the research question"


def _tavily_relevance_score(source: dict) -> tuple[int, str]:
    try:
        raw_score = float(source.get("score", 0))
    except (TypeError, ValueError):
        raw_score = 0.0

    score = round(max(0.0, min(raw_score, 1.0)) * 35)

    if score >= 26:
        return score, "high retrieval relevance"
    if score >= 13:
        return score, "moderate retrieval relevance"
    return score, "low or unavailable retrieval relevance"


def quality_label(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 55:
        return "Moderate"
    return "Low"


def _document_type_score(source: dict) -> tuple[int, str]:
    document_type = str(
        source.get("document_type", "Unclassified evidence record")
    ).strip()

    return DOCUMENT_TYPE_SCORES.get(
        document_type,
        (3, f"other evidence type: {document_type or 'not supplied'}"),
    )

def rank_sources(sources: list[dict], question: str) -> list[dict]:
    """Score every retrieved source before diversity selection."""
    ranked_sources = []

    for source in sources:
        ranked_source = deepcopy(source)
        url = str(ranked_source.get("url", ""))

        authority_points, authority_reason = _authority_score(url)
        tavily_points, tavily_reason = _tavily_relevance_score(ranked_source)
        match_points, match_reason = _question_match_score(
            ranked_source,
            question,
        )
        document_type_points, document_type_reason = _document_type_score(
            ranked_source
        )

        total_score = min(
            100,
            authority_points
            + tavily_points
            + match_points
            + document_type_points,
        )

        ranked_source["quality_score"] = total_score
        ranked_source["quality_label"] = quality_label(total_score)
        ranked_source["source_family"] = source_family(url)
        ranked_source["ranking_reason"] = (
            f"{authority_reason}; {tavily_reason}; {match_reason}; "
            f"{document_type_reason}."
        )
        ranked_sources.append(ranked_source)

    return sorted(
        ranked_sources,
        key=lambda item: item["quality_score"],
        reverse=True,
    )

def select_diverse_sources(
    ranked_sources: list[dict],
    limit: int = 10,
    max_per_family: int = 2,
) -> list[dict]:
    """
    Select varied sources:
    - First pass: best source from each independent provider.
    - Second pass: add more sources, never exceeding two per provider.
    """
    selected = []
    family_counts: dict[str, int] = {}

    # Prioritize one strong source from each provider.
    for source in ranked_sources:
        family = source["source_family"]
        if family_counts.get(family, 0) == 0:
            selected.append(source)
            family_counts[family] = 1

            if len(selected) == limit:
                return selected

    # Add further high-quality sources while preventing dominance by one provider.
    for source in ranked_sources:
        if source in selected:
            continue

        family = source["source_family"]
        if family_counts.get(family, 0) >= max_per_family:
            continue

        selected.append(source)
        family_counts[family] = family_counts.get(family, 0) + 1

        if len(selected) == limit:
            break

    return selected


def source_diversity_summary(sources: list[dict]) -> str:
    families = sorted({source.get("source_family", "Unknown provider") for source in sources})
    return f"{len(sources)} sources across {len(families)} independent providers: {', '.join(families)}"