"""Build a deterministic evidence-positioning contract before synthesis."""

from __future__ import annotations

import re
from typing import Any


OUTCOME_ALIASES: dict[str, tuple[str, ...]] = {
    "PASI-90": ("pasi-90", "pasi 90"),
    "PASI-100": ("pasi-100", "pasi 100"),
    "Dermatology quality of life": ("dermatology quality-of-life", "dermatology quality of life", "dlqi", "skindex"),
    "Psoriatic arthritis outcomes": ("psoriatic arthritis",),
    "Inflammatory bowel disease risk": ("inflammatory bowel disease", "ibd risk"),
    "Cardiovascular events": ("cardiovascular events", "cardiovascular event"),
    "Clinical remission": ("clinical remission", "remission induction"),
    "Endoscopic remission": ("endoscopic remission", "endoscopic response", "mucosal healing"),
    "Corticosteroid-free remission": ("corticosteroid-free remission", "steroid-free remission", "steroid free remission"),
    "Hospitalization": ("hospitalization", "hospitalisation", "admission"),
    "Colectomy": ("colectomy", "surgery"),
    "Serious infection": ("serious infection",),
    "Venous thromboembolism": ("venous thromboembolism", "vte", "thrombosis"),
    "Malignancy": ("malignancy", "cancer"),
    "Treatment discontinuation": ("treatment discontinuation", "discontinuation", "withdrawal"),
    "Annual exacerbation rate": ("annual exacerbation", "exacerbation rate"),
    "Oral-corticosteroid reduction": ("oral-corticosteroid reduction", "oral corticosteroid reduction", "ocs reduction"),
    "Lung function": ("fev1", "lung function"),
    "Asthma-control score": ("asthma-control score", "asthma control score", "act score", "acq"),
    "Quality of life": ("quality of life", "qol", "health status"),
    "Eosinophil reduction": ("eosinophil reduction", "eosinophil"),
    "Injection reactions": ("injection reaction", "injection-site reaction"),
    "Anaphylaxis": ("anaphylaxis",),
    "Long-term adverse events": ("long-term adverse", "long term adverse", "long-term safety", "long term safety"),
    "Cognitive decline": ("cognitive decline", "cognition"),
    "Functional independence": ("functional independence", "function"),
    "Amyloid clearance": ("amyloid clearance", "amyloid"),
    "ARIA-E": ("aria-e",),
    "ARIA-H": ("aria-h",),
    "Intracerebral hemorrhage": ("intracerebral hemorrhage", "intracerebral haemorrhage"),
    "Mortality": ("mortality", "death"),
    "FVC decline": ("fvc decline", "forced vital capacity"),
    "Acute exacerbations": ("acute exacerbation",),
    "Renal outcomes": ("renal", "kidney", "egfr", "albuminuria"),
    "Adverse events": ("adverse event", "safety", "toxicity"),
}

# These are high-value constraints where a broader population must never be
# represented as an exact population match.
CRITICAL_POPULATION_QUALIFIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("inadequate anti-TNF response", ("anti-tnf", "anti tnf", "tnf-inhibitor failure", "tnf inhibitor failure")),
    ("biomarker-confirmed amyloid pathology", ("amyloid-positive", "amyloid positive", "amyloid pathology", "biomarker-confirmed")),
    ("type 2 diabetes", ("type 2 diabetes", "t2d")),
    ("obesity", ("obesity", "obese")),
    ("severe uncontrolled asthma", ("severe asthma", "uncontrolled asthma")),
)


def _normalise(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("-", " ").casefold()).strip()


def _source_text(source: dict[str, Any]) -> str:
    return _normalise(" ".join(str(source.get(key, "")) for key in ("title", "raw_content", "content", "snippet", "description")))


def requested_outcomes(question: str) -> list[str]:
    text = _normalise(question)
    hits: list[tuple[int, str]] = []
    for label, aliases in OUTCOME_ALIASES.items():
        positions = [text.find(_normalise(alias)) for alias in aliases]
        positions = [position for position in positions if position >= 0]
        if positions:
            hits.append((min(positions), label))
    return [label for _, label in sorted(hits)]


def required_population_qualifiers(question: str) -> list[str]:
    text = _normalise(question)
    return [
        label for label, aliases in CRITICAL_POPULATION_QUALIFIERS
        if any(_normalise(alias) in text for alias in aliases)
    ]


def build_answer_contract(question: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """State the only evidence labels Nemotron may use in its conclusion."""
    qualifiers = required_population_qualifiers(question)
    outcomes = requested_outcomes(question)
    exact_direct, most_direct, indirect = [], [], []
    outcome_source_ids = {outcome: [] for outcome in outcomes}

    for source_id, source in enumerate(sources, start=1):
        text = _source_text(source)
        source_role = str(source.get("pico_status", "Supplementary"))
        missing_qualifiers = [
            qualifier
            for qualifier, aliases in CRITICAL_POPULATION_QUALIFIERS
            if qualifier in qualifiers and not any(_normalise(alias) in text for alias in aliases)
        ]
        for outcome in outcomes:
            aliases = OUTCOME_ALIASES[outcome]
            if any(_normalise(alias) in text for alias in aliases):
                outcome_source_ids[outcome].append(source_id)
        source_summary = {
            "source_id": source_id,
            "title": str(source.get("title", "Untitled source")),
            "missing_population_qualifiers": missing_qualifiers,
        }
        if source_role == "Direct" and not missing_qualifiers:
            exact_direct.append(source_summary)
        elif source_role == "Direct":
            most_direct.append(source_summary)
        else:
            indirect.append(source_summary)

    missing_outcomes = [outcome for outcome, ids in outcome_source_ids.items() if not ids]
    return {
        "requested_outcomes": outcomes,
        "required_population_qualifiers": qualifiers,
        "exact_direct": exact_direct,
        "most_direct": most_direct,
        "indirect": indirect,
        "outcome_source_ids": outcome_source_ids,
        "missing_outcomes": missing_outcomes,
    }


def format_answer_contract(contract: dict[str, Any]) -> str:
    def show(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "None"
        return "\n".join(
            f"- Source {row['source_id']}: {row['title']}"
            + (f" — missing target qualifier(s): {', '.join(row['missing_population_qualifiers'])}" if row["missing_population_qualifiers"] else "")
            for row in rows
        )

    outcome_status = "\n".join(
        f"- {outcome}: source IDs {ids or 'none'}"
        for outcome, ids in contract.get("outcome_source_ids", {}).items()
    ) or "- No explicit requested outcomes were parsed."
    return (
        "SYNTHESIS POSITIONING CONTRACT — Python generated; obey it exactly.\n"
        f"Exact Direct evidence sources:\n{show(contract.get('exact_direct', []))}\n"
        f"Most-direct evidence sources (related but not exact target population):\n{show(contract.get('most_direct', []))}\n"
        f"Indirect/context-only sources:\n{show(contract.get('indirect', []))}\n"
        f"Requested outcomes and source-signal availability:\n{outcome_status}\n"
        f"Outcomes with no keyword signal in the selected source text: {', '.join(contract.get('missing_outcomes', [])) or 'None identified by the availability scan.'}\n"
        "Required wording rules: A source listed as Most-direct must never be called Direct. "
        "For a missing outcome, write 'not found in the supplied evidence packet', not 'no evidence exists'. "
        "Do not mention any outcome that is not in the requested-outcomes list. "
        "One Source number must always refer to one and only one source-card title."
    )
