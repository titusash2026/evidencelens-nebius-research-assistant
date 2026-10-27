"""Free public biomedical retrieval for EvidenceLens.

This module is deliberately self-contained.  It retrieves bibliographic records
from public services and always returns the source schema expected by app.py.
Provider failures are recorded rather than allowed to stop a research run.
"""

from __future__ import annotations

import json
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

try:
    import truststore
except ImportError:  # pragma: no cover - standard SSL remains a safe fallback
    truststore = None


REQUEST_TIMEOUT_SECONDS = 15
USER_AGENT = "EvidenceLens/1.0 (academic biomedical evidence retrieval)"


def _ssl_context() -> ssl.SSLContext:
    if truststore is not None:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()


def _compact(value: object, limit: int = 2_000) -> str:
    """Return normalized, single-line text limited to a safe length."""
    return " ".join(str(value or "").split())[:limit]


def _request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS, context=SSL_CONTEXT) as response:
        payload = response.read().decode("utf-8", errors="replace")
    value = json.loads(payload)
    return value if isinstance(value, dict) else {}


def _request_xml(url: str) -> ElementTree.Element:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/xml"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS, context=SSL_CONTEXT) as response:
        payload = response.read()
    return ElementTree.fromstring(payload)


def _normalise_url(url: str) -> str:
    parts = urlsplit(str(url).strip())
    if not parts.scheme or not parts.netloc:
        return str(url).strip()
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def _source(
    *,
    title: object,
    url: str,
    bucket: str,
    abstract: object = "",
    authors: object = "",
    published_date: object = "",
    source_id: object = "",
    source_type: str = "Biomedical record",
    evidence_text_status: str = "abstract",
    evidence_role_hint: str = "",
) -> dict[str, Any]:
    clean_title = _compact(title, 500) or "Untitled source"
    text_limit = 12_000 if evidence_text_status == "full_text_results_conclusions" else 4_000
    clean_abstract = _compact(abstract, text_limit)
    return {
        "title": clean_title,
        "url": url,
        "content": clean_abstract or clean_title,
        "raw_content": clean_abstract or clean_title,
        "snippet": clean_abstract,
        "description": clean_abstract,
        "authors": _compact(authors, 1_000),
        "published_date": _compact(published_date, 100),
        "source_id": _compact(source_id, 100),
        "source_type": source_type,
        "retrieval_bucket": bucket,
        "retrieval_provider": "free_biomedical_api",
        "evidence_text_status": evidence_text_status,
        "evidence_role_hint": evidence_role_hint,
    }


def _pubmed_sources(query: str, limit: int) -> list[dict[str, Any]]:
    params = urlencode({"db": "pubmed", "term": query, "retmax": max(1, min(limit, 20)), "retmode": "json"})
    search = _request_json(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}")
    ids = search.get("esearchresult", {}).get("idlist", [])
    ids = [str(item) for item in ids if str(item).strip()]
    if not ids:
        return []

    summary_params = urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    summary = _request_json(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{summary_params}")
    records = summary.get("result", {})
    # ESummary does not contain an article abstract.  Fetch the MEDLINE XML so
    # Nemotron receives clinical evidence rather than an author name.
    fetch_params = urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
    abstract_by_pmid: dict[str, str] = {}
    try:
        xml_root = _request_xml(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{fetch_params}")
        for article in xml_root.findall(".//PubmedArticle"):
            pmid = _compact(article.findtext(".//PMID"))
            abstract_nodes = article.findall(".//Abstract/AbstractText")
            abstract = " ".join(_compact(" ".join(node.itertext()), 8_000) for node in abstract_nodes)
            if pmid and abstract:
                abstract_by_pmid[pmid] = _compact(abstract, 8_000)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, ElementTree.ParseError):
        # Keep the record discoverable, but explicitly mark the absent abstract.
        abstract_by_pmid = {}
    output: list[dict[str, Any]] = []
    for pmid in ids:
        item = records.get(pmid, {})
        if not isinstance(item, dict):
            continue
        authors = "; ".join(author.get("name", "") for author in item.get("authors", []) if isinstance(author, dict))
        output.append(_source(
            title=item.get("title", ""),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            bucket="Free PubMed fallback",
            abstract=abstract_by_pmid.get(pmid, ""),
            authors=authors,
            published_date=item.get("pubdate", ""),
            source_id=pmid,
            source_type=item.get("pubtype", "PubMed record"),
            evidence_text_status=("abstract" if abstract_by_pmid.get(pmid) else "metadata_only"),
            evidence_role_hint=("" if abstract_by_pmid.get(pmid) else "Supplementary"),
        ))
    return output


def _europe_pmc_sources(query: str, limit: int) -> list[dict[str, Any]]:
    params = urlencode({"query": query, "format": "json", "pageSize": max(1, min(limit, 25)), "resultType": "core"})
    payload = _request_json(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}")
    items = [item for item in payload.get("resultList", {}).get("result", []) if isinstance(item, dict)]

    def fulltext_for_pmcid(pmcid: str) -> str:
        root = _request_xml(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
        passages: list[str] = []
        for section in root.findall(".//sec"):
            heading = _compact(" ".join(node.text or "" for node in section.findall("./title")), 160)
            heading_key = heading.casefold()
            if any(word in heading_key for word in ("result", "conclusion", "discussion")):
                text = _compact(" ".join(section.itertext()), 10_000)
                if text:
                    passages.append(f"[{heading or 'Evidence passage'}] {text}")
        if not passages:
            passages.append(_compact(" ".join(root.itertext()), 10_000))
        return _compact(" ".join(passages), 12_000)

    fulltext_by_pmcid: dict[str, str] = {}
    pmcids = [
        _compact(item.get("pmcid")) for item in items
        if _compact(item.get("pmcid"))
    ][:4]
    if pmcids:
        with ThreadPoolExecutor(max_workers=min(4, len(pmcids))) as executor:
            futures = {executor.submit(fulltext_for_pmcid, pmcid): pmcid for pmcid in pmcids}
            for future in as_completed(futures):
                pmcid = futures[future]
                try:
                    fulltext_by_pmcid[pmcid] = future.result()
                except (HTTPError, URLError, TimeoutError, OSError, ValueError, ElementTree.ParseError):
                    continue

    output: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pmid = _compact(item.get("pmid"))
        pmcid = _compact(item.get("pmcid"))
        identifier = pmid or pmcid or _compact(item.get("id"))
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else f"https://europepmc.org/article/{item.get('source', 'MED')}/{identifier}"
        output.append(_source(
            title=item.get("title", ""),
            url=url,
            bucket="Free Europe PMC fallback",
            abstract=fulltext_by_pmcid.get(pmcid) or item.get("abstractText", ""),
            authors=item.get("authorString", ""),
            published_date=item.get("firstPublicationDate") or item.get("pubYear", ""),
            source_id=identifier,
            source_type=item.get("pubType", "Europe PMC record"),
            evidence_text_status=("full_text_results_conclusions" if fulltext_by_pmcid.get(pmcid) else ("abstract" if item.get("abstractText") else "metadata_only")),
            evidence_role_hint=("" if (fulltext_by_pmcid.get(pmcid) or item.get("abstractText")) else "Supplementary"),
        ))
    return output


def _openalex_sources(query: str, limit: int) -> list[dict[str, Any]]:
    params = urlencode({"search": query, "per-page": max(1, min(limit, 25))})
    payload = _request_json(f"https://api.openalex.org/works?{params}")
    output: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        authors = "; ".join(
            author.get("author", {}).get("display_name", "")
            for author in item.get("authorships", [])
            if isinstance(author, dict)
        )
        landing_url = item.get("doi") or item.get("id") or ""
        if landing_url and not str(landing_url).startswith("http"):
            landing_url = f"https://doi.org/{landing_url}"
        record = _source(
            title=item.get("title", ""),
            url=str(landing_url),
            bucket="Free OpenAlex fallback",
            abstract="",
            authors=authors,
            published_date=item.get("publication_date", ""),
            source_id=item.get("id", ""),
            source_type=item.get("type", "OpenAlex record"),
            evidence_text_status="metadata_only",
            evidence_role_hint="Supplementary",
        )
        record["doi"] = str(item.get("doi") or "")
        record["open_access_url"] = str((item.get("best_oa_location") or {}).get("landing_page_url") or "")
        output.append(record)
    return [record for record in output if record.get("url")]


def _clinicaltrials_sources(query: str, limit: int) -> list[dict[str, Any]]:
    params = urlencode({"query.term": query, "pageSize": max(1, min(limit, 20)), "format": "json"})
    payload = _request_json(f"https://clinicaltrials.gov/api/v2/studies?{params}")
    output: list[dict[str, Any]] = []
    for study in payload.get("studies", []):
        protocol = study.get("protocolSection", {}) if isinstance(study, dict) else {}
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        description = protocol.get("descriptionModule", {})
        results = study.get("resultsSection", {}) if isinstance(study, dict) else {}
        nct_id = _compact(identification.get("nctId"))
        if not nct_id:
            continue
        outcome_module = results.get("outcomeMeasuresModule", {}) if isinstance(results, dict) else {}
        outcome_measures = outcome_module.get("outcomeMeasures", []) if isinstance(outcome_module, dict) else []
        posted_results = _compact(json.dumps(outcome_measures, ensure_ascii=False), 8_000) if outcome_measures else ""
        output.append(_source(
            title=identification.get("briefTitle", ""),
            url=f"https://clinicaltrials.gov/study/{nct_id}",
            bucket="Free ClinicalTrials.gov fallback",
            abstract=posted_results or description.get("briefSummary", ""),
            published_date=status.get("studyFirstPostDateStruct", {}).get("date", ""),
            source_id=nct_id,
            source_type="Clinical trial registry record",
            evidence_text_status=("posted_registry_results" if posted_results else "registry_metadata"),
            evidence_role_hint=("" if posted_results else "Supplementary"),
        ))
    return output


Provider = Callable[[str, int], list[dict[str, Any]]]
PROVIDERS: tuple[tuple[str, Provider], ...] = (
    ("Free PubMed fallback", _pubmed_sources),
    ("Free Europe PMC fallback", _europe_pmc_sources),
    ("Free OpenAlex fallback", _openalex_sources),
    ("Free ClinicalTrials.gov fallback", _clinicaltrials_sources),
)


def retrieve_free_biomedical_sources(
    queries: list[str],
    per_source_limit: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """Retrieve, deduplicate, and safely report free public biomedical records."""
    unique_queries: list[str] = []
    for query in queries:
        cleaned = _compact(query, 2_000)
        if cleaned and cleaned not in unique_queries:
            unique_queries.append(cleaned)
        if len(unique_queries) >= 12:
            break

    bucket_counts = {name: 0 for name, _ in PROVIDERS}
    failures: list[str] = []
    source_map: dict[str, dict[str, Any]] = {}
    if not unique_queries:
        return [], bucket_counts, failures

    for query in unique_queries:
        with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as executor:
            futures = {
                executor.submit(provider, query, per_source_limit): name
                for name, provider in PROVIDERS
            }
            for future in as_completed(futures):
                provider_name = futures[future]
                try:
                    records = future.result()
                except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, ElementTree.ParseError) as error:
                    failures.append(f"{provider_name} ({type(error).__name__})")
                    continue
                except Exception as error:  # provider isolation is intentional
                    failures.append(f"{provider_name} ({type(error).__name__})")
                    continue

                for record in records:
                    url = _normalise_url(str(record.get("url", "")))
                    if not url or url in source_map:
                        continue
                    record["url"] = url
                    source_map[url] = record
                    bucket_counts[provider_name] += 1

    return list(source_map.values()), bucket_counts, failures
