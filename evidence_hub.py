"""Stage 1: literature retrieval and conservative record deduplication.

No PICO decision, medical answer, or inference API call is made here.
Python >= 3.11. HTTPX handles HTTP; asyncio bounds each connector operation.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
import re
import time
from typing import Any
from xml.etree import ElementTree as ET

import httpx

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MAX_RESPONSE_BYTES = 4_000_000
IDENTIFIERS = ("doi", "pmid", "pmcid")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderHTTPError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class ProviderDataError(Exception):
    pass


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "br", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "div", "li"}:
            self.parts.append("\n")


def plain_text(value: Any) -> str:
    parser = PlainText()
    parser.feed(str(value or ""))
    return " ".join("".join(parser.parts).split())


def xml_text(node) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def normalize_doi(value: Any) -> str:
    # Only called on dedicated bibliographic ID fields. Never scan abstracts.
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text)
    return text if re.fullmatch(r"10\.\d{4,9}/\S+", text) else ""


def canonical_ids(record: dict) -> dict[str, str]:
    pmid = str(record.get("pmid") or "").strip()
    pmcid = str(record.get("pmcid") or "").strip().upper()
    return {
        "doi": normalize_doi(record.get("doi")),
        "pmid": pmid if pmid.isdigit() else "",
        "pmcid": pmcid if re.fullmatch(r"PMC\d+", pmcid) else "",
    }


def parse_pubmed_xml(content: bytes) -> list[dict]:
    if b"<!ENTITY" in content.upper():
        raise ProviderDataError("Entity declarations are not accepted")
    root = ET.fromstring(content)
    if root.tag != "PubmedArticleSet" or root.find(".//ERROR") is not None:
        raise ProviderDataError("Unexpected PubMed response")
    records = []
    for item in root.findall("PubmedArticle"):
        citation = item.find("MedlineCitation")
        article = item.find("MedlineCitation/Article")
        if citation is None or article is None:
            raise ProviderDataError("Missing PubMed article data")
        pmid = xml_text(citation.find("PMID"))
        if not pmid.isdigit():
            raise ProviderDataError("Missing PubMed identifier")
        ids = {
            elem.get("IdType", "").lower(): xml_text(elem)
            for elem in item.findall("PubmedData/ArticleIdList/ArticleId")
        }
        for elem in article.findall("ELocationID"):
            if elem.get("EIdType") == "doi" and elem.get("ValidYN") != "N":
                ids.setdefault("doi", xml_text(elem))
        paragraphs = []
        for elem in article.findall("Abstract/AbstractText"):
            label = elem.get("Label", "")
            paragraphs.append((label + ": " if label else "") + xml_text(elem))
        date = article.find("Journal/JournalIssue/PubDate")
        published = ""
        if date is not None:
            published = xml_text(date.find("MedlineDate")) or "-".join(
                xml_text(date.find(field)) for field in ("Year", "Month", "Day")
                if xml_text(date.find(field))
            )
        records.append({
            "record_id": f"pubmed:{pmid}", "provider": "pubmed",
            "title": xml_text(article.find("ArticleTitle")),
            "abstract": "\n".join(paragraphs), "pmid": pmid,
            "doi": normalize_doi(ids.get("doi")), "pmcid": ids.get("pmc", "").upper(),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "published": published,
            "publication_types": [xml_text(e) for e in article.findall("PublicationTypeList/PublicationType")],
            "publication_notices": [
                {"relation": e.get("RefType", ""), "related_pmid": xml_text(e.find("PMID"))}
                for e in citation.findall("CommentsCorrectionsList/CommentsCorrections")
            ],
            "retrieved_at": utc_now(), "content_level": "abstract_or_metadata",
        })
    return records


def parse_europe_pmc(payload: dict) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("resultList"), dict):
        raise ProviderDataError("Unexpected Europe PMC response")
    items = payload.get("resultList", {}).get("result")
    if not isinstance(items, list) or "hitCount" not in payload:
        raise ProviderDataError("Unexpected Europe PMC response")
    records = []
    for item in items:
        if not isinstance(item, dict):
            raise ProviderDataError("Unexpected Europe PMC record")
        types = item.get("pubTypeList") or {}
        if not isinstance(types, dict) or not isinstance(types.get("pubType", []), list):
            raise ProviderDataError("Unexpected Europe PMC publication types")
        source, identifier = str(item.get("source", "")), str(item.get("id", ""))
        if not source or not identifier or not re.fullmatch(r"[A-Za-z0-9_.-]+", source + identifier):
            raise ProviderDataError("Missing or invalid Europe PMC identifier")
        pmid = str(item.get("pmid") or (identifier if source == "MED" else ""))
        records.append({
            "record_id": f"europe_pmc:{source}:{identifier}", "provider": "europe_pmc",
            "title": plain_text(item.get("title")),
            "abstract": plain_text(item.get("abstractText")),
            "doi": normalize_doi(item.get("doi")), "pmid": pmid,
            "pmcid": str(item.get("pmcid") or (identifier if source == "PMC" else "")).upper(),
            "url": f"https://europepmc.org/article/{source}/{identifier}",
            "published": str(item.get("firstPublicationDate") or item.get("pubYear") or ""),
            "publication_types": types.get("pubType", []),
            "publication_notices": [],
            "retrieved_at": utc_now(), "content_level": "abstract_or_metadata",
        })
    return records


def deduplicate_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge only shared authoritative article IDs, preserving original records.

    ID conflicts remain separate and are audited. No title-only merges or
    trial-level merging: distinct reports of one trial must remain available.
    """
    groups: list[list[dict]] = []
    audit = []
    for original in records:
        record = deepcopy(original)
        record.update(canonical_ids(record))
        record_keys = {(k, record[k]) for k in IDENTIFIERS if record[k]}
        matches = [
            index for index, group in enumerate(groups)
            if record_keys & {(k, r[k]) for r in group for k in IDENTIFIERS if r[k]}
        ]
        combined = [record] + [r for index in matches for r in groups[index]]
        conflicts = [k for k in IDENTIFIERS if len({r[k] for r in combined if r[k]}) > 1]
        if matches and conflicts:
            audit.append({"action": "kept_separate_identifier_conflict", "record_id": record["record_id"],
                          "conflicting_fields": conflicts})
            groups.append([record])
        elif matches:
            audit.append({"action": "merged_duplicate_records", "record_id": record["record_id"],
                          "matched_records": [r["record_id"] for i in matches for r in groups[i]]})
            for index in reversed(matches):
                groups.pop(index)
            groups.append(combined)
        else:
            groups.append([record])
    unique = []
    for group in groups:
        # A display abstract is selected, but every source version is preserved.
        richest = max(group, key=lambda r: len(r.get("abstract", "")))
        ids = {k: next((r[k] for r in group if r[k]), "") for k in IDENTIFIERS}
        identity = json.dumps(ids, sort_keys=True) if any(ids.values()) else richest["record_id"]
        unique.append({
            "article_id": "article_" + sha256(identity.encode()).hexdigest()[:16],
            **ids, "title": richest["title"], "abstract": richest["abstract"],
            "abstract_source_record_id": richest["record_id"],
            "urls": sorted({r["url"] for r in group}),
            "providers": sorted({r["provider"] for r in group}),
            "source_records": group,
            "pico_status": "not_screened", "verification_status": "not_verified",
        })
    return unique, audit


async def fetch_bytes(client: httpx.AsyncClient, url: str, params: dict) -> bytes:
    chunks = []
    size = 0
    async with client.stream("GET", url, params=params) as response:
        if response.status_code != 200:
            raise ProviderHTTPError(response.status_code)
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise ProviderDataError("Response exceeds the starter's size limit")
            chunks.append(chunk)
    return b"".join(chunks)


async def search_pubmed(client: httpx.AsyncClient, query: str, limit: int) -> dict:
    raw = await fetch_bytes(client, PUBMED_BASE + "esearch.fcgi", {
        "db": "pubmed", "term": query, "retmode": "json", "retmax": limit,
        "sort": "relevance", "tool": "EvidenceLensStage1",
    })
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("esearchresult"), dict):
        raise ProviderDataError("Unexpected PubMed search response")
    search = payload.get("esearchresult", {})
    if "count" not in search or not isinstance(search.get("idlist"), list) or search.get("ERROR"):
        raise ProviderDataError("Unexpected PubMed search response")
    total = int(search["count"])
    ids = search["idlist"]
    if total < 0 or not all(isinstance(value, str) and value.isdigit() for value in ids):
        raise ProviderDataError("Invalid PubMed count or identifiers")
    warnings = []
    if search.get("warninglist") or search.get("errorlist"):
        warnings.append("PubMed reported query translation warnings; inspect its search syntax.")
    if not ids:
        if total:
            raise ProviderDataError("PubMed reported matches but returned no IDs")
        return {"records": [], "total_hits": total, "warnings": warnings}
    # Two NCBI calls per run; avoid a burst between ESearch and EFetch.
    await asyncio.sleep(0.35)
    content = await fetch_bytes(client, PUBMED_BASE + "efetch.fcgi", {
        "db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "EvidenceLensStage1",
    })
    records = parse_pubmed_xml(content)
    requested = set(ids)
    if any(r["pmid"] not in requested for r in records):
        raise ProviderDataError("PubMed returned an unrequested article")
    if {r["pmid"] for r in records} != requested:
        warnings.append("Some requested PubMed records were not returned as journal articles.")
    return {"records": records, "total_hits": total, "warnings": warnings}


async def search_europe_pmc(client: httpx.AsyncClient, query: str, limit: int) -> dict:
    raw = await fetch_bytes(client, EUROPE_PMC_SEARCH, {
        "query": query, "format": "json", "resultType": "core", "pageSize": limit,
    })
    payload = json.loads(raw)
    records = parse_europe_pmc(payload)
    total = int(payload["hitCount"])
    if total and not records:
        raise ProviderDataError("Europe PMC reported matches but returned no records")
    return {"records": records, "total_hits": total, "warnings": []}


async def run_hub(query: str, limit: int = 15, budget_seconds: float = 8.0,
                  transport: httpx.AsyncBaseTransport | None = None) -> dict:
    if not query.strip() or len(query) > 1500:
        raise ValueError("Use a nonempty search query of at most 1500 characters")
    if not 1 <= limit <= 30 or not 0 < budget_seconds <= 30:
        raise ValueError("Limit must be 1..30 and connector budget must be >0 and <=30 seconds")
    start = time.perf_counter()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(min(5.0, budget_seconds)), transport=transport,
        follow_redirects=False, headers={"User-Agent": "EvidenceLens-Stage1/1.0"},
    ) as client:
        async def run_one(name, adapter):
            began = time.perf_counter()
            result = {"provider": name, "records": [], "total_hits": None, "warnings": []}
            try:
                async with asyncio.timeout(budget_seconds):
                    result.update(await adapter(client, query, limit))
                result["status"] = "partial" if result["warnings"] else ("ok" if result["records"] else "empty")
            except (TimeoutError, httpx.TimeoutException):
                result.update(status="timeout", error="Connector exceeded its request or operation budget")
            except ProviderHTTPError as exc:
                result.update(status="rate_limited" if exc.status_code == 429 else "error",
                              error=f"HTTP {exc.status_code}")
            except (httpx.HTTPError, ProviderDataError, ValueError, ET.ParseError, TypeError, KeyError) as exc:
                # Do not include URLs, headers, credentials, or response bodies in errors.
                result.update(status="error", error=type(exc).__name__)
            result["elapsed_seconds"] = round(time.perf_counter() - began, 3)
            result["returned_records"] = len(result["records"])
            return result
        providers = await asyncio.gather(
            run_one("pubmed", search_pubmed), run_one("europe_pmc", search_europe_pmc)
        )
    raw_records = [record for provider in providers for record in provider["records"]]
    articles, audit = deduplicate_records(raw_records)
    all_ok = all(p["status"] == "ok" for p in providers)
    conflicts = any(a["action"] == "kept_separate_identifier_conflict" for a in audit)
    gate = "PASS" if all_ok and articles and not conflicts else ("REVIEW" if articles else "NO_RECORDS")
    return {
        "schema_version": "1.0", "stage": "retrieval_only", "query": query,
        "created_at": utc_now(), "elapsed_seconds": round(time.perf_counter() - start, 3),
        "connector_budget_seconds": budget_seconds, "per_provider_limit": limit,
        "retrieval_check": gate,
        "raw_record_count": len(raw_records), "unique_article_count": len(articles),
        "duplicate_records_merged": len(raw_records) - len(articles),
        "articles_with_abstract": sum(bool(r["abstract"]) for r in articles),
        "providers": [{k: v for k, v in p.items() if k != "records"} for p in providers],
        "deduplication_audit": audit, "articles": articles,
        "clinical_answer_allowed": False,
        "coverage_assessed": False,
        "limitations": [
            "PASS means the two retrieval connectors returned records, not that a medical answer is correct.",
            "This is a capped first-page search; it is not a comprehensive evidence review.",
            "PICO screening, guideline retrieval, study-level linking and claim verification are not implemented in Stage 1.",
            "Abstracts and metadata are retrieved; full text is not fetched.",
            "PubMed and Europe PMC overlap and are not independent evidence confirmations.",
            "No Nemotron or other model was called.",
        ],
    }
