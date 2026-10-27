"""Deterministic evidence packet construction for EvidenceLens."""
from __future__ import annotations

import re
from typing import Any


OUTCOME_PATTERNS: dict[str, tuple[str, ...]] = {
    "Clinical remission": ("clinical remission", "remission induction"),
    "Endoscopic remission": ("endoscopic remission", "endoscopic response", "mucosal healing"),
    "Corticosteroid-free remission": ("corticosteroid-free remission", "steroid-free remission", "steroid free remission"),
    "Colectomy": ("colectomy",),
    "Serious infection": ("serious infection",),
    "Venous thromboembolism": ("venous thromboembolism", "vte", "thrombosis"),
    "Malignancy": ("malignancy", "cancer"),
    "Treatment discontinuation": ("treatment discontinuation", "discontinuation", "withdrawal"),
    "Annual exacerbation rate": ("annual exacerbation", "exacerbation rate"),
    "Oral-corticosteroid reduction": ("oral-corticosteroid reduction", "oral corticosteroid reduction", "ocs reduction"),
    "Lung function": ("fev1", "lung function"),
    "Asthma-control score": ("asthma-control score", "asthma control score", "act score", "acq"),
    "Eosinophil reduction": ("eosinophil reduction", "eosinophil"),
    "Injection reactions": ("injection reaction", "injection-site reaction"),
    "Anaphylaxis": ("anaphylaxis",),
    "Long-term adverse events": ("long-term adverse", "long term adverse", "long-term safety", "long term safety"),
    "Cardiovascular death / mortality": (
        "cardiovascular death", "cv death", "cardiovascular mortality",
        "all-cause mortality", "all cause mortality", "mortality", "death",
    ),
    "Heart-failure events / hospitalization": (
        "heart failure hospitalization", "hf hospitalization", "hospitalisation",
        "hospitalization", "worsening heart failure", "hf event", "rehospital",
    ),
    "Symptoms / quality of life": (
        "kccq", "quality of life", "symptom", "health status", "nyha",
    ),
    "Exercise capacity": (
        "6-minute walk", "6 minute walk", "6mwd", "exercise capacity",
        "peak vo", "functional capacity", "walk distance",
    ),
    "Weight / metabolic outcomes": (
        "weight loss", "body weight", "weight reduction", "bmi", "adiposity",
    ),
    "Renal outcomes": ("egfr", "renal", "kidney", "albuminuria", "dialysis"),
    "Adverse events / tolerability": (
        "adverse event", "serious adverse", "safety", "discontinuation",
        "hyperkalemia", "hypotension", "gastrointestinal", "volume depletion",
    ),
}
NUMERIC_SIGNAL = re.compile(
    r"\b(?:hr|rr|or|md|mean difference|hazard ratio|risk ratio)\b|"
    r"\b\d+(?:\.\d+)?\s*%|95\s*%\s*ci|p\s*[<=>]", re.IGNORECASE
)


def source_text(source: dict[str, Any]) -> str:
    return str(source.get("raw_content") or source.get("content") or "").strip()


def evidence_text_status(source: dict[str, Any], text: str) -> str:
    supplied = str(source.get("evidence_text_status", "")).strip()
    if supplied:
        return supplied
    if len(text) >= 8_000:
        return "substantial retrieved text"
    if len(text) >= 1_000:
        return "abstract or extended excerpt"
    if text:
        return "short excerpt"
    return "metadata only — not eligible to support outcome claims"


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if len(s.strip()) >= 35]


def _matching_sentences(text: str, patterns: tuple[str, ...], limit: int = 3) -> list[str]:
    matches = []
    for sentence in _sentences(text):
        if any(pattern in sentence.casefold() for pattern in patterns):
            matches.append(sentence[:900])
        if len(matches) >= limit:
            break
    return matches


def _numeric_sentences(text: str, limit: int = 4) -> list[str]:
    return [sentence[:900] for sentence in _sentences(text) if NUMERIC_SIGNAL.search(sentence)][:limit]


def _patterns_for_question(question: str) -> dict[str, tuple[str, ...]]:
    question_text = question.casefold()
    requested = {
        label: patterns
        for label, patterns in OUTCOME_PATTERNS.items()
        if any(pattern in question_text for pattern in patterns)
    }
    return requested or OUTCOME_PATTERNS


def build_evidence_packet(question: str, sources: list[dict[str, Any]], *, chars_per_source: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    requested_patterns = _patterns_for_question(question)
    cards, coverage = [], {outcome: 0 for outcome in requested_patterns}
    for source_id, source in enumerate(sources, start=1):
        text = source_text(source)
        selected_text = text[:chars_per_source]
        outcome_quotes = {}
        for outcome, patterns in requested_patterns.items():
            quotes = _matching_sentences(selected_text, patterns)
            outcome_quotes[outcome] = quotes
            if quotes:
                coverage[outcome] += 1
        cards.append({
            "source_id": source_id,
            "title": str(source.get("title", "Untitled source")),
            "url": str(source.get("url", "")),
            "document_type": str(source.get("document_type", "Unclassified evidence record")),
            "evidence_role": str(source.get("pico_status", "Direct")),
            "population_relevance": str(source.get("pico_reason", "Not assessed")),
            "text_status": evidence_text_status(source, text),
            "outcome_quotes": outcome_quotes,
            "numeric_result_quotes": _numeric_sentences(selected_text),
            "retrieved_text": selected_text,
        })
    return cards, coverage


def format_evidence_packet(question: str, cards: list[dict[str, Any]], coverage: dict[str, int]) -> str:
    coverage_text = "\n".join(f"- {outcome}: {count} selected source(s) contain an outcome signal" for outcome, count in coverage.items())
    sections = []
    for card in cards:
        quote_groups = []
        for outcome, values in card["outcome_quotes"].items():
            if values:
                quote_groups.append(f"{outcome}:\n" + "\n".join(f"- {value}" for value in values))
        quote_block = "\n".join(quote_groups) or "No outcome-specific passage was extracted."
        numeric_block = "\n".join(f"- {value}" for value in card["numeric_result_quotes"]) or "No numerical-result passage was extracted."
        sections.append(
            f"[Source {card['source_id']}]\nTitle: {card['title']}\nURL: {card['url']}\n"
            f"Study type: {card['document_type']}\nEvidence role: {card['evidence_role']}\n"
            f"Population relevance: {card['population_relevance']}\nText completeness: {card['text_status']}\n"
            f"Outcome passages copied from this source:\n{quote_block}\n"
            f"Numerical-result passages copied from this source:\n{numeric_block}\n"
            f"Retrieved text for verification:\n{card['retrieved_text']}"
        )
    return f"Research question:\n{question}\n\nOutcome-coverage audit (availability, not proof):\n{coverage_text}\n\nSource cards:\n\n" + "\n\n".join(sections)
