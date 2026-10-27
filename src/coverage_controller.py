"""Treatment-by-outcome evidence coverage gate for EvidenceLens."""
from __future__ import annotations

import re
from typing import Any


TREATMENT_ALIASES = {
    "SGLT2 inhibitors": ("sglt2 inhibitor", "empagliflozin", "dapagliflozin", "canagliflozin"),
    "GLP-1 receptor agonists": ("glp-1", "glp1", "semaglutide", "liraglutide", "dulaglutide"),
    "Dual GIP/GLP-1 agonists": ("gip/glp-1", "gip glp-1", "tirzepatide"),
    "Mineralocorticoid receptor antagonists": ("mineralocorticoid", "mra", "finerenone", "spironolactone", "eplerenone"),
    "ARNI therapy": ("arni", "sacubitril", "valsartan"),
    "Diuretics": ("diuretic", "furosemide", "torsemide", "bumetanide"),
    "Structured exercise rehabilitation": ("exercise rehabilitation", "exercise training", "cardiac rehabilitation", "structured exercise"),
    "Nintedanib": ("nintedanib",),
    "Pirfenidone": ("pirfenidone",),
    "Lung transplantation": ("lung transplant", "transplant referral"),
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
        "oral corticosteroid",
    ),
}

OUTCOME_ALIASES = {
    "Cardiovascular death / mortality": ("cardiovascular death", "cv death", "cardiovascular mortality", "mortality", "death"),
    "Hospitalization / disease events": ("hospitalization", "hospitalisation", "worsening heart failure", "acute exacerbation", "disease event", "rehospital"),
    "Symptoms / quality of life": ("symptom", "quality of life", "kccq", "health status", "functional independence"),
    "Exercise capacity / function": ("exercise capacity", "six-minute walk", "6-minute walk", "6mwd", "functional capacity", "physical function"),
    "Weight / metabolic outcomes": ("weight loss", "body weight", "weight reduction", "bmi", "adiposity", "metabolic"),
    "Renal outcomes": ("renal", "kidney", "egfr", "albuminuria"),
    "Adverse events / tolerability": ("adverse event", "safety", "hyperkalemia", "hypotension", "gastrointestinal", "discontinuation", "toxicity"),
    "Exacerbations": ("exacerbation", "asthma attack"),
    "Lung function": ("fev1", "forced expiratory volume"),
    "Oral-corticosteroid reduction": ("oral corticosteroid reduction", "ocs reduction", "steroid sparing"),
    "Asthma control": ("asthma control", "act score", "acq"),
    "Eosinophil reduction": ("eosinophil",),
    "Anaphylaxis / injection reactions": ("anaphylaxis", "injection reaction", "injection-site reaction"),
}


def _normalise(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("-", " ").casefold()).strip()


def _labels(text: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    return [label for label, terms in aliases.items() if any(_normalise(term) in text for term in terms)]


def _source_text(source: dict[str, Any]) -> str:
    return _normalise(" ".join(str(source.get(key, "")) for key in ("title", "raw_content", "content", "snippet")))


def _specification(question: str, plan: dict[str, Any]) -> tuple[list[str], list[str]]:
    question_text = _normalise(question)
    treatments = _labels(question_text, TREATMENT_ALIASES)
    outcomes = _labels(question_text, OUTCOME_ALIASES)
    for planned in plan.get("outcomes", []) if isinstance(plan.get("outcomes"), list) else []:
        for outcome in _labels(_normalise(planned), OUTCOME_ALIASES):
            if outcome not in outcomes:
                outcomes.append(outcome)
    return treatments, outcomes


def assess_evidence_coverage(question: str, plan: dict[str, Any], direct_sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure retrieval coverage only; strict verification still proves claims."""
    treatments, outcomes = _specification(question, plan)
    matrix = {treatment: {outcome: [] for outcome in outcomes} for treatment in treatments}
    source_matches: dict[int, dict[str, list[str]]] = {}
    for source_id, source in enumerate(direct_sources, start=1):
        text = _source_text(source)
        treatment_hits = _labels(text, {item: TREATMENT_ALIASES[item] for item in treatments})
        outcome_hits = _labels(text, {item: OUTCOME_ALIASES[item] for item in outcomes})
        source_matches[source_id] = {"treatments": treatment_hits, "outcomes": outcome_hits}
        for treatment in treatment_hits:
            for outcome in outcome_hits:
                matrix[treatment][outcome].append(source_id)
    covered_treatments = [t for t in treatments if any(matrix[t][o] for o in outcomes)]
    covered_outcomes = [o for o in outcomes if any(matrix[t][o] for t in treatments)]
    missing_treatments = [t for t in treatments if t not in covered_treatments]
    missing_outcomes = [o for o in outcomes if o not in covered_outcomes]
    treatment_ratio = len(covered_treatments) / len(treatments) if treatments else 1.0
    outcome_ratio = len(covered_outcomes) / len(outcomes) if outcomes else 1.0
    is_comparative = len(treatments) >= 2
    ready = len(direct_sources) >= 2 and (not is_comparative or treatment_ratio >= 0.80) and (not outcomes or outcome_ratio >= 0.60)
    reasons = []
    if len(direct_sources) < 2:
        reasons.append("Fewer than two direct sources survived PICO triage.")
    if is_comparative and treatment_ratio < 0.80:
        reasons.append("Direct evidence was not retrieved for enough named treatment categories.")
    if outcomes and outcome_ratio < 0.60:
        reasons.append("Direct evidence was not retrieved for enough requested outcome categories.")
    return {
        "ready_for_full_comparison": ready,
        "is_comparative_question": is_comparative,
        "direct_source_count": len(direct_sources),
        "treatments": treatments,
        "outcomes": outcomes,
        "matrix": matrix,
        "source_matches": source_matches,
        "covered_treatments": covered_treatments,
        "covered_outcomes": covered_outcomes,
        "missing_treatments": missing_treatments,
        "missing_outcomes": missing_outcomes,
        "treatment_coverage_ratio": round(treatment_ratio, 2),
        "outcome_coverage_ratio": round(outcome_ratio, 2),
        "reasons": reasons,
    }


def build_targeted_coverage_queries(question: str, coverage: dict[str, Any], *, max_queries: int = 12) -> list[str]:
    """Create bounded public-literature queries only for evidence gaps."""
    queries: list[str] = []
    outcomes = list(coverage.get("missing_outcomes", [])) or list(coverage.get("outcomes", []))[:2] or ["benefits harms"]
    for treatment in coverage.get("missing_treatments", []):
        queries.append(f"{question} {treatment} {' '.join(outcomes[:3])} systematic review randomized trial guideline")
        if len(queries) >= max_queries:
            return queries
    for outcome in coverage.get("missing_outcomes", []):
        queries.append(f"{question} {outcome} systematic review randomized trial")
        if len(queries) >= max_queries:
            break
    return queries


def coverage_summary(coverage: dict[str, Any]) -> str:
    state = "Ready" if coverage.get("ready_for_full_comparison") else "Incomplete"
    return (f"{state}: {coverage.get('direct_source_count', 0)} direct sources; "
            f"treatment coverage {coverage.get('treatment_coverage_ratio', 0):.0%}; "
            f"outcome coverage {coverage.get('outcome_coverage_ratio', 0):.0%}.")
