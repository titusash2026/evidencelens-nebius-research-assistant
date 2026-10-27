"""Exact intervention matching for controller-selected direct evidence.

The PICO classifier determines general population/outcome relevance.  This
module adds a deterministic guard: a source cannot be Direct evidence for a
comparative question unless it actually studies at least one intervention the
user named.
"""

from __future__ import annotations

import re
from typing import Any


INTERVENTION_ALIASES: dict[str, tuple[str, ...]] = {
    "TNF inhibitors": ("tnf inhibitor", "anti-tnf", "adalimumab", "infliximab", "etanercept", "certolizumab"),
    "IL-17 inhibitors": ("il-17 inhibitor", "anti-il17", "anti-il-17", "secukinumab", "ixekizumab", "brodalumab", "bimekizumab"),
    "IL-23 inhibitors": ("il-23 inhibitor", "anti-il23", "anti-il-23", "guselkumab", "risankizumab", "tildrakizumab"),
    "Ustekinumab": ("ustekinumab",),
    "Methotrexate": ("methotrexate",),
    "Cyclosporine": ("cyclosporine", "ciclosporin"),
    "Apremilast": ("apremilast",),
    "Phototherapy": ("phototherapy", "narrowband uvb", "nb-uvb"),
    "SGLT2 inhibitors": ("sglt2 inhibitor", "empagliflozin", "dapagliflozin", "canagliflozin"),
    "GLP-1 receptor agonists": ("glp-1", "glp1", "semaglutide", "liraglutide", "dulaglutide"),
    "Dual GIP/GLP-1 agonists": ("gip/glp-1", "gip glp-1", "tirzepatide"),
    "Mineralocorticoid receptor antagonists": ("mineralocorticoid", "finerenone", "spironolactone", "eplerenone"),
    "ARNI therapy": ("arni", "sacubitril", "sacubitril/valsartan", "sacubitril valsartan"),
    "Diuretics": ("diuretic", "furosemide", "torsemide", "bumetanide"),
    "Structured exercise rehabilitation": ("exercise rehabilitation", "exercise training", "cardiac rehabilitation", "structured exercise"),
    "Nintedanib": ("nintedanib",),
    "Pirfenidone": ("pirfenidone",),
    "Lecanemab": ("lecanemab",),
    "Donanemab": ("donanemab",),
    "Mepolizumab": ("mepolizumab",),
    "Benralizumab": ("benralizumab",),
    "Dupilumab": ("dupilumab",),
    "Tezepelumab": ("tezepelumab",),
    "Omalizumab": ("omalizumab",),
    "Bronchial thermoplasty": ("bronchial thermoplasty",),
    "Long-term oral corticosteroids": (
        "long-term oral corticosteroid", "long term oral corticosteroid",
        "maintenance oral corticosteroid", "chronic oral corticosteroid",
        "oral corticosteroid", "oral glucocorticoid",
    ),
}


def _normalise(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("-", " ").casefold()).strip()


def requested_interventions(question: str) -> list[str]:
    """Return known named interventions in the order in which the user wrote them."""
    text = _normalise(question)
    matches: list[tuple[int, str]] = []
    for label, aliases in INTERVENTION_ALIASES.items():
        positions = [text.find(_normalise(alias)) for alias in aliases]
        positions = [position for position in positions if position >= 0]
        if positions:
            matches.append((min(positions), label))
    return [label for _, label in sorted(matches)]


def _source_text(source: dict[str, Any]) -> str:
    return _normalise(
        " ".join(
            str(source.get(field, ""))
            for field in ("title", "raw_content", "content", "snippet", "description")
        )
    )


def source_intervention_matches(source: dict[str, Any], interventions: list[str]) -> list[str]:
    text = _source_text(source)
    return [
        label
        for label in interventions
        if any(_normalise(alias) in text for alias in INTERVENTION_ALIASES.get(label, (label,)))
    ]


def build_treatment_queries(question: str) -> list[str]:
    """Build one focused biomedical-search lane per requested intervention."""
    interventions = requested_interventions(question)
    if not interventions:
        return []
    population_context = re.split(r"\bhow\s+do\b", question, maxsplit=1, flags=re.IGNORECASE)[0]
    population_context = " ".join(population_context.split())[:420]
    return [
        f"{treatment} {population_context} randomized trial systematic review efficacy safety"
        for treatment in interventions
    ]


def enforce_exact_intervention_match(
    question: str,
    direct_sources: list[dict[str, Any]],
    supplementary_sources: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Demote a PICO-relevant source when it studies none of the named treatments."""
    interventions = requested_interventions(question)
    if not interventions:
        return direct_sources, supplementary_sources, audit

    retained: list[dict[str, Any]] = []
    contextual = list(supplementary_sources)
    updated_audit = list(audit)
    for source in direct_sources:
        matched = source_intervention_matches(source, interventions)
        metadata_only = str(source.get("evidence_text_status", "")).casefold() in {
            "metadata_only", "retrieval_excerpt_only"
        }
        if matched and not metadata_only:
            kept = dict(source)
            kept["matched_interventions"] = matched
            retained.append(kept)
            continue

        moved = dict(source)
        moved["pico_status"] = "Supplementary"
        if metadata_only:
            reason = "Metadata-only record; it cannot establish a direct clinical outcome."
        else:
            reason = (
                "Out of scope for direct comparison: the source does not study any "
                f"requested intervention ({', '.join(interventions)})."
            )
        moved["pico_reason"] = reason
        contextual.append(moved)
        updated_audit.append(
            {
                "title": str(moved.get("title", "Untitled source")),
                "url": str(moved.get("url", "")),
                "role": "Supplementary evidence",
                "reason": reason,
            }
        )
    return retained, contextual, updated_audit
