from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
PMID_PATTERN = re.compile(r"\bPMID\s*[:#]?\s*(\d{5,9})\b", re.IGNORECASE)
PUBMED_URL_PATTERN = re.compile(
    r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})",
    re.IGNORECASE,
)


def _normalise_title(title: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(title or "").lower())
    return " ".join(cleaned.split())


def _canonical_url(url: Any) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""

    parts = urlsplit(raw_url)
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/").lower()

    return urlunsplit(
        (parts.scheme.lower() or "https", host, path, "", "")
    )


def _extract_doi(source: dict[str, Any]) -> str:
    text = " ".join(
        str(source.get(field, ""))
        for field in ("doi", "url", "title", "content", "raw_content")
    )
    match = DOI_PATTERN.search(text)
    return match.group(0).lower().rstrip(".,;)") if match else ""


def _extract_pmid(source: dict[str, Any]) -> str:
    direct_value = str(source.get("pmid", "")).strip()
    if direct_value.isdigit():
        return direct_value

    text = " ".join(
        str(source.get(field, ""))
        for field in ("url", "title", "content", "raw_content")
    )

    url_match = PUBMED_URL_PATTERN.search(text)
    if url_match:
        return url_match.group(1)

    pmid_match = PMID_PATTERN.search(text)
    return pmid_match.group(1) if pmid_match else ""


def _publication_date(source: dict[str, Any]) -> str:
    for field in (
        "published_date",
        "publication_date",
        "published",
        "date",
        "updated_date",
    ):
        value = str(source.get(field, "")).strip()
        if value:
            return value

    return "Not supplied by retrieval provider"


def _document_type(source: dict[str, Any]) -> str:
    """Classify evidence using title-first, study-specific signals."""
    title = str(source.get("title", "")).strip().casefold()
    supplied_type = str(
        source.get("bibliographic_document_type", "")
    ).strip().casefold()

    title_and_metadata = f"{title} {supplied_type}"

    # Specific research designs must take priority over generic guidance words.
    if (
        "systematic review" in title_and_metadata
        or "meta-analysis" in title_and_metadata
        or "meta analysis" in title_and_metadata
    ):
        return "Systematic review or meta-analysis"

    if (
        "randomized controlled trial" in title_and_metadata
        or "randomised controlled trial" in title_and_metadata
        or "randomized trial" in title_and_metadata
        or "randomised trial" in title_and_metadata
    ):
        return "Randomized controlled trial"

    if (
        "cohort study" in title_and_metadata
        or "prospective study" in title_and_metadata
        or "prospective cohort" in title_and_metadata
    ):
        return "Observational cohort study"

    if (
        "microsimulation" in title_and_metadata
        or "modelling study" in title_and_metadata
        or "modeling study" in title_and_metadata
    ):
        return "Modelling study"

    if (
        "guideline" in title_and_metadata
        or "practice guidance" in title_and_metadata
        or "recommendation" in title_and_metadata
    ):
        return "Clinical guideline"

    if (
        "evidence briefing" in title_and_metadata
        or "technology appraisal" in title_and_metadata
        or "device" in title_and_metadata
    ):
        return "Evidence briefing or device record"

    if supplied_type and supplied_type not in {
        "pubmed literature record",
        "journal literature record",
        "article",
        "literature record",
    }:
        return supplied_type.title()

    url = str(source.get("url", "")).casefold()
    if "pubmed" in url or "pmc" in url or "journal" in url:
        return "Journal literature record"

    return "Unclassified evidence record"


def _version_identity(
    source: dict[str, Any],
    document_type: str,
    publication_date: str,
) -> str:
    explicit_version = str(source.get("version", "")).strip()
    if explicit_version:
        return f"Explicit source version: {explicit_version}"

    text = " ".join(
        str(source.get(field, ""))
        for field in ("title", "content", "raw_content", "url")
    )

    version_match = re.search(
        r"\bversion\s*(\d+(?:\.\d+){0,3})\b",
        text,
        re.IGNORECASE,
    )
    if version_match:
        return f"Extracted version: {version_match.group(1)}"

    if document_type == "Clinical guideline":
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        if year_match:
            return f"Guideline year label: {year_match.group(0)}"

    if publication_date != "Not supplied by retrieval provider":
        return f"Retrieved publication/update date: {publication_date}"

    return "Version/update date not supplied"


def _stable_document_id(
    doi: str,
    canonical_url: str,
    normalised_title: str,
) -> tuple[str, str, str]:
    if doi:
        return f"doi:{doi}", "DOI", "High"

    if canonical_url:
        digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
        return f"url:{digest}", "Canonical URL", "Moderate"

    if normalised_title:
        digest = hashlib.sha256(normalised_title.encode("utf-8")).hexdigest()[:16]
        return f"title:{digest}", "Normalised title", "Low"

    return "source:unidentified", "No stable identifier", "Low"


def enrich_source_identity(source: dict[str, Any]) -> dict[str, Any]:
    """Add local document provenance without changing evidence decisions."""
    enriched = dict(source)

    doi = _extract_doi(enriched)
    pmid = _extract_pmid(enriched)
    canonical_url = _canonical_url(enriched.get("url", ""))
    normalised_title = _normalise_title(enriched.get("title", ""))

    document_id, identity_method, identity_confidence = _stable_document_id(
        doi=doi,
        canonical_url=canonical_url,
        normalised_title=normalised_title,
    )

    source_domain = urlsplit(canonical_url).netloc or "Unknown publisher"
    publication_date = _publication_date(enriched)
    document_type = _document_type(enriched)

    enriched.update(
        {
            "document_id": document_id,
            "document_title": str(
                enriched.get("title", "Untitled source")
            ).strip()
            or "Untitled source",
            "doi": doi,
            "pmid": pmid,
            "canonical_url": canonical_url,
            "source_domain": source_domain,
            "publisher": str(
                enriched.get("publisher", source_domain)
            ).strip()
            or source_domain,
            "publication_date": publication_date,
            "version_identity": _version_identity(
                enriched,
                document_type,
                publication_date,
            ),
            "document_type": document_type,
            "identity_method": identity_method,
            "identity_confidence": identity_confidence,
            "bibliographic_lookup_status": str(
                enriched.get("bibliographic_lookup_status", "not_requested")
            ),
            "bibliographic_metadata_source": str(
                enriched.get("bibliographic_metadata_source", "")
            ),
        }
    )

    return enriched


def enrich_source_identities(
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    enriched_sources = [enrich_source_identity(source) for source in sources]

    audit = {
        "total_sources": len(enriched_sources),
        "doi_identified": sum(bool(source["doi"]) for source in enriched_sources),
        "canonical_urls": sum(
            bool(source["canonical_url"]) for source in enriched_sources
        ),
        "publication_dates": sum(
            source["publication_date"]
            != "Not supplied by retrieval provider"
            for source in enriched_sources
        ),
        "high_confidence": sum(
            source["identity_confidence"] == "High"
            for source in enriched_sources
        ),
    }

    return enriched_sources, audit


def format_identity_summary(audit: dict[str, int]) -> str:
    return (
        "Stage 05 identity audit: "
        f"{audit['total_sources']} sources · "
        f"{audit['doi_identified']} DOI identifiers · "
        f"{audit['canonical_urls']} canonical URLs · "
        f"{audit['publication_dates']} dates supplied · "
        f"{audit['high_confidence']} high-confidence identities."
    )


def _fetch_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "EvidenceLens/1.0 metadata resolver",
        },
    )

    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _title_similarity(left: str, right: str) -> float:
    left_normalised = _normalise_title(left)
    right_normalised = _normalise_title(right)

    if not left_normalised or not right_normalised:
        return 0.0

    sequence_score = SequenceMatcher(
        None,
        left_normalised,
        right_normalised,
    ).ratio()

    left_tokens = set(left_normalised.split())
    right_tokens = set(right_normalised.split())
    token_score = len(left_tokens & right_tokens) / max(len(left_tokens), 1)

    return max(sequence_score, token_score)


def _crossref_date(message: dict[str, Any]) -> str:
    for field in (
        "published-print",
        "published-online",
        "published",
        "issued",
    ):
        parts = (
            message.get(field, {})
            .get("date-parts", [[]])[0]
        )
        if parts:
            return "-".join(str(part) for part in parts)

    return ""


def _crossref_lookup_by_title(
    title: str,
    timeout_seconds: float,
) -> dict[str, str]:
    params = urlencode(
        {
            "query.title": title,
            "rows": 1,
            "select": (
                "DOI,title,published,published-online,published-print,"
                "issued,container-title,publisher,type,URL"
            ),
        }
    )
    payload = _fetch_json(
        f"https://api.crossref.org/works?{params}",
        timeout_seconds,
    )

    items = payload.get("message", {}).get("items", [])
    if not items:
        return {}

    candidate = items[0]
    candidate_title = " ".join(candidate.get("title", []))

    # Do not attach a DOI unless title matching is reasonably strong.
    if _title_similarity(title, candidate_title) < 0.78:
        return {}

    return {
        "doi": str(candidate.get("DOI", "")).lower(),
        "publication_date": _crossref_date(candidate),
        "publisher": str(candidate.get("publisher", "")).strip(),
        "bibliographic_document_type": str(candidate.get("type", "")).strip(),
        "canonical_url": _canonical_url(candidate.get("URL", "")),
    }


def _pubmed_lookup(
    pmid: str,
    timeout_seconds: float,
) -> dict[str, str]:
    params = urlencode(
        {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
        }
    )
    payload = _fetch_json(
        (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            f"esummary.fcgi?{params}"
        ),
        timeout_seconds,
    )

    record = payload.get("result", {}).get(str(pmid), {})
    if not record:
        return {}

    doi = ""
    for article_id in record.get("articleids", []):
        if str(article_id.get("idtype", "")).lower() == "doi":
            doi = str(article_id.get("value", "")).lower()
            break

    return {
        "pmid": str(pmid),
        "resolved_title": str(record.get("title", "")).strip(),
        "doi": doi,
        "publication_date": str(
            record.get("pubdate")
            or record.get("sortpubdate")
            or ""
        ).strip(),
        "publisher": str(record.get("source", "")).strip(),
        "bibliographic_document_type": "PubMed literature record",
    }


def _find_pubmed_pmid_by_title(
    title: str,
    timeout_seconds: float,
) -> str:
    """Find a PubMed record only when its returned title closely matches.

    Tavily often returns a publisher or society URL instead of a PubMed URL.
    This bounded, title-matched lookup lets the app retrieve the public PubMed
    abstract without guessing an identifier or bypassing publisher access rules.
    """
    normalised_title = _normalise_title(title)
    if len(normalised_title) < 20:
        return ""

    params = urlencode(
        {
            "db": "pubmed",
            "term": f'"{title}"[Title]',
            "retmode": "json",
            "retmax": 3,
        }
    )
    payload = _fetch_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + params,
        timeout_seconds,
    )
    candidate_ids = payload.get("esearchresult", {}).get("idlist", [])
    if not isinstance(candidate_ids, list):
        return ""

    for candidate_id in candidate_ids[:3]:
        candidate_pmid = str(candidate_id).strip()
        if not candidate_pmid.isdigit():
            continue
        metadata = _pubmed_lookup(candidate_pmid, timeout_seconds)
        if _title_similarity(title, metadata.get("resolved_title", "")) >= 0.82:
            return candidate_pmid

    return ""

def _fetch_pubmed_abstract(
    pmid: str,
    timeout_seconds: float,
) -> str:
    """Fetch the publicly available PubMed abstract for one PMID."""
    params = urlencode(
        {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
    )

    request = Request(
        (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            f"efetch.fcgi?{params}"
        ),
        headers={
            "Accept": "application/xml",
            "User-Agent": "EvidenceLens/1.0 evidence-text resolver",
        },
    )

    with urlopen(request, timeout=timeout_seconds) as response:
        root = ElementTree.fromstring(response.read())

    sections = []

    for abstract_node in root.findall(".//Abstract/AbstractText"):
        text = " ".join("".join(abstract_node.itertext()).split())

        if not text:
            continue

        label = str(abstract_node.attrib.get("Label", "")).strip()

        if label:
            sections.append(f"{label}: {text}")
        else:
            sections.append(text)

    return "\n".join(sections).strip()


def _fetch_pubmed_related_pmids(
    pmid: str,
    timeout_seconds: float,
    limit: int,
) -> list[str]:
    """Return a bounded set of PubMed related-article identifiers."""
    params = urlencode(
        {
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "cmd": "neighbor_score",
            "retmode": "xml",
        }
    )
    request = Request(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?" + params,
        headers={
            "Accept": "application/xml",
            "User-Agent": "EvidenceLens/1.0 related-evidence resolver",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        root = ElementTree.fromstring(response.read())

    related_pmids: list[str] = []
    for node in root.findall(".//LinkSetDb/Link/Id"):
        candidate = str(node.text or "").strip()
        if candidate.isdigit() and candidate != pmid and candidate not in related_pmids:
            related_pmids.append(candidate)
        if len(related_pmids) >= limit:
            break
    return related_pmids


def expand_related_pubmed_sources(
    sources: list[dict[str, Any]],
    max_seed_sources: int = 3,
    max_related_sources: int = 6,
    timeout_seconds: float = 4.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build a small, auditable evidence graph from trusted seed papers.

    Each added paper must be a PubMed related article with a public abstract.
    This intentionally does *not* crawl arbitrary web links or publisher pages.
    """
    existing_pmids = {
        str(source.get("pmid", "")).strip()
        for source in sources
        if str(source.get("pmid", "")).strip().isdigit()
    }
    expanded_sources: list[dict[str, Any]] = []
    audit = {
        "seed_sources_considered": 0,
        "seed_pmids_resolved": 0,
        "related_pmids_discovered": 0,
        "related_sources_added": 0,
        "related_sources_skipped_no_abstract": 0,
        "related_lookup_failed_safe": 0,
    }

    ranked_seeds = sorted(
        (dict(source) for source in sources),
        key=lambda source: int(source.get("quality_score", 0) or 0),
        reverse=True,
    )
    for seed in ranked_seeds[:max_seed_sources]:
        if len(expanded_sources) >= max_related_sources:
            break
        audit["seed_sources_considered"] += 1
        seed_pmid = str(seed.get("pmid") or _extract_pmid(seed)).strip()
        if not seed_pmid:
            try:
                seed_pmid = _find_pubmed_pmid_by_title(
                    str(seed.get("document_title") or seed.get("title") or ""),
                    timeout_seconds,
                )
            except Exception:
                audit["related_lookup_failed_safe"] += 1
                continue
        if not seed_pmid:
            continue

        audit["seed_pmids_resolved"] += 1
        try:
            remaining = max_related_sources - len(expanded_sources)
            related_pmids = _fetch_pubmed_related_pmids(
                seed_pmid,
                timeout_seconds,
                remaining,
            )
            audit["related_pmids_discovered"] += len(related_pmids)
        except Exception:
            audit["related_lookup_failed_safe"] += 1
            continue

        for related_pmid in related_pmids:
            if related_pmid in existing_pmids:
                continue
            try:
                metadata = _pubmed_lookup(related_pmid, timeout_seconds)
                abstract = _fetch_pubmed_abstract(related_pmid, timeout_seconds)
            except Exception:
                audit["related_lookup_failed_safe"] += 1
                continue

            if not abstract:
                audit["related_sources_skipped_no_abstract"] += 1
                continue

            existing_pmids.add(related_pmid)
            expanded_sources.append(
                {
                    "title": metadata.get("resolved_title") or f"PubMed {related_pmid}",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{related_pmid}/",
                    "pmid": related_pmid,
                    "doi": metadata.get("doi", ""),
                    "publisher": metadata.get("publisher", "PubMed"),
                    "publication_date": metadata.get("publication_date", ""),
                    "bibliographic_document_type": metadata.get(
                        "bibliographic_document_type", "PubMed literature record"
                    ),
                    "raw_content": f"PubMed abstract:\n{abstract}",
                    "content": abstract,
                    "evidence_text_status": "pubmed_related_abstract",
                    "evidence_access_gap": "",
                    "retrieval_bucket": "PubMed related-evidence expansion",
                    "evidence_relation": (
                        f"PubMed related article linked to seed PMID {seed_pmid}."
                    ),
                    "related_to_pmid": seed_pmid,
                    "score": 0.80,
                }
            )
            audit["related_sources_added"] += 1
            if len(expanded_sources) >= max_related_sources:
                break

    return expanded_sources, audit


def format_related_evidence_summary(audit: dict[str, int]) -> str:
    return (
        "Related-evidence graph: "
        f"{audit['seed_pmids_resolved']} seed PMID(s) · "
        f"{audit['related_pmids_discovered']} related record(s) discovered · "
        f"{audit['related_sources_added']} public abstract(s) added · "
        f"{audit['related_lookup_failed_safe']} failure-safe lookup(s)."
    )


def enrich_source_evidence_text(
    sources: list[dict[str, Any]],
    max_abstract_lookups: int = 8,
    timeout_seconds: float = 4.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Preserve Tavily text and append an official PubMed abstract when available.

    This is bounded, failure-safe, and does not bypass publisher access controls.
    """
    enriched_sources = []
    lookup_count = 0
    identifier_lookup_count = 0

    audit = {
        "selected_sources": len(sources),
        "pubmed_abstract_attempted": 0,
        "pubmed_abstract_enriched": 0,
        "pubmed_title_match_attempted": 0,
        "pubmed_title_match_resolved": 0,
        "retrieval_excerpt_only": 0,
        "access_gaps_recorded": 0,
        "failed_safe": 0,
        "tavily_raw_content_fallback": 0,
    }

    for source in sources:
        current = dict(source)

        existing_text = str(
            current.get("raw_content")
            or current.get("content")
            or ""
        ).strip()

        pmid = str(current.get("pmid") or _extract_pmid(current)).strip()

        # Resolve an official PubMed identifier from the article title when
        # retrieval returned a journal, guideline, or society landing page.
        if not pmid and identifier_lookup_count < max_abstract_lookups:
            identifier_lookup_count += 1
            audit["pubmed_title_match_attempted"] += 1
            try:
                pmid = _find_pubmed_pmid_by_title(
                    str(current.get("document_title") or current.get("title") or ""),
                    timeout_seconds,
                )
            except Exception as error:
                current["pubmed_title_match_error"] = (
                    f"{type(error).__name__}: {error}"
                )[:160]

            if pmid:
                current["pmid"] = pmid
                audit["pubmed_title_match_resolved"] += 1

        if not pmid:
            current["evidence_text_status"] = "retrieval_excerpt_only"
            current["evidence_access_gap"] = (
                "No title-matched PubMed identifier was available for abstract enrichment."
            )
            audit["retrieval_excerpt_only"] += 1
            audit["access_gaps_recorded"] += 1
            enriched_sources.append(current)
            continue

        if lookup_count >= max_abstract_lookups:
            current["evidence_text_status"] = "not_requested_limit"
            current["evidence_access_gap"] = (
                "PubMed abstract enrichment was not attempted because the "
                "bounded lookup limit was reached."
            )
            audit["access_gaps_recorded"] += 1
            enriched_sources.append(current)
            continue

        lookup_count += 1
        audit["pubmed_abstract_attempted"] += 1

        try:
            abstract_text = _fetch_pubmed_abstract(pmid, timeout_seconds)

            if not abstract_text:
                current["evidence_text_status"] = "pubmed_abstract_unavailable"
                current["evidence_access_gap"] = (
                    "No PubMed abstract text was available for this record."
                )
                audit["access_gaps_recorded"] += 1
                enriched_sources.append(current)
                continue

            if abstract_text.casefold() not in existing_text.casefold():
                sections = [section for section in (existing_text,) if section]
                sections.append(f"PubMed abstract:\n{abstract_text}")
                current["raw_content"] = "\n\n".join(sections)

            current["evidence_text_status"] = "pubmed_abstract_enriched"
            current["evidence_access_gap"] = ""
            audit["pubmed_abstract_enriched"] += 1
            enriched_sources.append(current)

        except Exception as error:
            current["evidence_text_error"] = (
                f"{type(error).__name__}: {error}"
            )[:160]

            if len(existing_text) >= 300:
                current["raw_content"] = existing_text
                current["evidence_text_status"] = "tavily_raw_content_fallback"
                current["evidence_access_gap"] = (
                    "PubMed abstract retrieval failed; retained retrieved source "
                    "text for exact-quote verification."
                )
                audit["tavily_raw_content_fallback"] += 1
                audit["access_gaps_recorded"] += 1
                enriched_sources.append(current)
                continue

            current["evidence_text_status"] = "pubmed_abstract_fetch_failed_safe"
            current["evidence_access_gap"] = (
                f"PubMed abstract retrieval failed safely: "
                f"{type(error).__name__}"
            )
            audit["failed_safe"] += 1
            audit["access_gaps_recorded"] += 1
            enriched_sources.append(current)

    return enriched_sources, audit


def format_evidence_text_summary(audit: dict[str, int]) -> str:
    return (
        "Stage 06 evidence-text enrichment: "
        f"{audit['pubmed_abstract_enriched']} PubMed abstract(s) added · "
        f"{audit.get('pubmed_title_match_resolved', 0)} PMID(s) resolved from title · "
        f"{audit['tavily_raw_content_fallback']} retrieved-text fallback(s) used · "
        f"{audit['retrieval_excerpt_only']} excerpt-only source(s) · "
        f"{audit['access_gaps_recorded']} access/extraction gap(s) recorded."
    )


def _merge_external_metadata(
    source: dict[str, Any],
    metadata: dict[str, str],
    metadata_source: str,
) -> dict[str, Any]:
    merged = dict(source)

    for field_name in ("doi", "pmid"):
        if metadata.get(field_name):
            merged[field_name] = metadata[field_name]

    if (
        metadata.get("publication_date")
        and merged.get("publication_date")
        in ("", "Not supplied by retrieval provider")
    ):
        merged["publication_date"] = metadata["publication_date"]

    if metadata.get("publisher"):
        merged["publisher"] = metadata["publisher"]

    if (
        metadata.get("bibliographic_document_type")
        and merged.get("document_type") == "Unclassified evidence record"
    ):
        merged["bibliographic_document_type"] = (
            metadata["bibliographic_document_type"]
        )

    merged["bibliographic_lookup_status"] = "resolved"
    merged["bibliographic_metadata_source"] = metadata_source
    return merged


def enrich_bibliographic_metadata(
    sources: list[dict[str, Any]],
    max_external_lookups: int = 3,
    timeout_seconds: float = 3.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Resolve missing DOI/PMID metadata for only the final selected sources.

    Lookups are bounded and best-effort. A lookup failure changes only the
    source's metadata status; it never removes a source or blocks synthesis.
    """
    enriched_sources: list[dict[str, Any]] = []
    lookup_count = 0
    audit = {
        "selected_sources": len(sources),
        "lookups_attempted": 0,
        "resolved": 0,
        "failed_safe": 0,
        "pubmed_resolved": 0,
        "crossref_resolved": 0,
    }

    for source in sources:
        current = enrich_source_identity(source)
        pmid = current.get("pmid", "")
        doi = current.get("doi", "")
        title = current.get("document_title", "")

        should_lookup = bool(pmid) or (not doi and len(title) >= 20)

        if not should_lookup:
            current["bibliographic_lookup_status"] = "not_needed"
            enriched_sources.append(current)
            continue

        if lookup_count >= max_external_lookups:
            current["bibliographic_lookup_status"] = "not_requested_limit"
            enriched_sources.append(current)
            continue

        lookup_count += 1
        audit["lookups_attempted"] += 1

        try:
            if pmid:
                metadata = _pubmed_lookup(pmid, timeout_seconds)
                metadata_source = "NCBI PubMed ESummary"
                source_name = "pubmed"
            else:
                metadata = _crossref_lookup_by_title(title, timeout_seconds)
                metadata_source = "Crossref REST API"
                source_name = "crossref"

            if metadata:
                current = _merge_external_metadata(
                    current,
                    metadata,
                    metadata_source,
                )
                audit["resolved"] += 1
                audit[f"{source_name}_resolved"] += 1
            else:
                current["bibliographic_lookup_status"] = "no_verified_match"

        except Exception as error:
            current["bibliographic_lookup_status"] = "lookup_failed_safe"
            current["bibliographic_lookup_error"] = (
                f"{type(error).__name__}: {error}"
            )[:160]
            audit["failed_safe"] += 1

        enriched_sources.append(current)

    return enriched_sources, audit


def format_bibliographic_summary(audit: dict[str, int]) -> str:
    return (
        "Stage 05 metadata resolution: "
        f"{audit['lookups_attempted']} bounded lookup(s) · "
        f"{audit['resolved']} resolved · "
        f"{audit['pubmed_resolved']} PubMed · "
        f"{audit['crossref_resolved']} Crossref · "
        f"{audit['failed_safe']} failure-safe."
    )


def build_identity_registry_csv(sources: list[dict[str, Any]]) -> str:
    """Create a downloadable source-provenance registry."""
    fieldnames = [
        "document_id",
        "document_title",
        "document_type",
        "doi",
        "pmid",
        "canonical_url",
        "publisher",
        "source_domain",
        "publication_date",
        "version_identity",
        "retrieval_bucket",
        "identity_method",
        "identity_confidence",
        "bibliographic_lookup_status",
        "bibliographic_metadata_source",
        "quality_score",
        "quality_label",
    ]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    for source in sources:
        writer.writerow(
            {
                field_name: source.get(field_name, "")
                for field_name in fieldnames
            }
        )

    return buffer.getvalue()
