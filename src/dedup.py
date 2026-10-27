from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _normalise_title(title: str) -> str:
    title = str(title).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split())


def _canonical_url(url: str) -> str:
    """Remove fragments, tracking parameters, www, and a trailing slash."""
    raw_url = str(url).strip()
    if not raw_url:
        return ""

    parts = urlsplit(raw_url)
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/").lower()

    return urlunsplit(
        (
            parts.scheme.lower() or "https",
            host,
            path,
            "",  # discard query parameters such as utm_source
            "",  # discard fragments
        )
    )


def _extract_doi(source: dict[str, Any]) -> str:
    text = " ".join(
        str(source.get(field, ""))
        for field in ("doi", "url", "title", "content", "raw_content")
    ).lower()

    match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text)
    return match.group(0).rstrip(".,;)") if match else ""


def _source_strength(source: dict[str, Any]) -> tuple[float, int, int]:
    """
    Prefer the source with an existing quality score; otherwise prefer
    the richer source text. This preserves the strongest duplicate.
    """
    score = 0.0

    for field in ("quality_score", "evidencelens_quality_score", "score"):
        try:
            score = max(score, float(source.get(field, 0)))
        except (TypeError, ValueError):
            pass

    text_length = len(
        str(source.get("raw_content", ""))
        or str(source.get("content", ""))
    )
    title_length = len(str(source.get("title", "")))

    return score, text_length, title_length


def _source_keys(source: dict[str, Any]) -> list[tuple[str, str]]:
    keys = []

    doi = _extract_doi(source)
    if doi:
        keys.append(("DOI", doi))

    canonical_url = _canonical_url(source.get("url", ""))
    if canonical_url:
        keys.append(("canonical URL", canonical_url))

    title = _normalise_title(source.get("title", ""))
    if len(title) >= 24:
        keys.append(("normalised title", title))

    return keys


def _audit_row(
    discarded: dict[str, Any],
    retained: dict[str, Any],
    match_method: str,
) -> dict[str, str]:
    return {
        "excluded_title": str(discarded.get("title", "Untitled source")),
        "excluded_url": str(discarded.get("url", "")),
        "retained_title": str(retained.get("title", "Untitled source")),
        "retained_url": str(retained.get("url", "")),
        "reason": f"Duplicate detected by matching {match_method}.",
    }


def deduplicate_sources(
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Keep one copy of each source, detected by DOI, canonical URL,
    or an exact normalised-title match.
    """
    unique_sources: list[dict[str, Any]] = []
    duplicate_audit: list[dict[str, str]] = []
    seen_keys: dict[tuple[str, str], int] = {}

    for source in sources:
        source_copy = dict(source)
        keys = _source_keys(source_copy)

        matching_keys = [key for key in keys if key in seen_keys]

        if not matching_keys:
            position = len(unique_sources)
            unique_sources.append(source_copy)

            for key in keys:
                seen_keys[key] = position
            continue

        match_method = matching_keys[0][0]
        position = seen_keys[matching_keys[0]]
        existing_source = unique_sources[position]

        if _source_strength(source_copy) > _source_strength(existing_source):
            unique_sources[position] = source_copy
            duplicate_audit.append(
                _audit_row(
                    discarded=existing_source,
                    retained=source_copy,
                    match_method=match_method,
                )
            )

            for key in keys:
                seen_keys[key] = position
        else:
            duplicate_audit.append(
                _audit_row(
                    discarded=source_copy,
                    retained=existing_source,
                    match_method=match_method,
                )
            )

    return unique_sources, duplicate_audit