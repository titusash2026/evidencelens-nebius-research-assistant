"""Stage 2: transparent PICO candidate screening and coverage checks.

This module is deliberately conservative. It labels records as candidates for
manual/semantic review; it does not decide clinical truth or permit an answer.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "can", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "should", "that", "the", "their",
    "this", "to", "what", "which", "with", "would",
}

OUTCOME_PHRASES = {
    "hip fracture": ("hip fracture", "hip fractures"),
    "vertebral fracture": (
        "vertebral fracture", "vertebral fractures", "vertebra fracture",
        "spine fracture", "spinal fracture",
    ),
    "pain": ("pain",),
    "disability": ("disability", "functional limitation", "function"),
    "mortality": ("mortality", "death", "survival"),
    "hospital admission": ("hospital admission", "hospitalization", "hospitalisation"),
    "hba1c": ("hba1c", "hemoglobin a1c", "glycated hemoglobin"),
}

POPULATION_SYNONYMS = {
    "older adults": ("older adults", "older adult", "elderly", "aged", "older people"),
    "adults": ("adults", "adult"),
    "children": ("children", "child", "paediatric", "pediatric", "adolescent"),
    "postmenopausal women": ("postmenopausal", "post-menopausal"),
}

INTERVENTION_TERMS = (
    "alendronate", "risedronate", "zoledronate", "zoledronic acid", "ibandronate",
    "denosumab", "romosozumab", "teriparatide", "abaloparatide", "raloxifene",
    "hormone replacement", "estrogen", "oestrogen", "bisphosphonate", "treatment",
    "therapy", "therapies", "intervention", "exercise", "physiotherapy", "medication",
)


def normalise(text: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).split())


def contains_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def matched_phrases(text: str, phrases: tuple[str, ...] | list[str]) -> list[str]:
    return [phrase for phrase in phrases if contains_phrase(text, normalise(phrase))]


def outcome_present(text: str, label: str) -> bool:
    """Handle coordinated wording such as 'hip and vertebral fractures'."""
    if label in {"hip fracture", "vertebral fracture"}:
        location = "hip" if label == "hip fracture" else "vertebral"
        return contains_phrase(text, location) and bool(re.search(r"\bfractur(?:e|es)\b", text))
    return bool(matched_phrases(text, list(OUTCOME_PHRASES[label])))


def extract_question_profile(question: str) -> dict[str, Any]:
    """Return visible extraction cues, never a hidden or model-generated PICO."""
    text = normalise(question)
    tokens = [word for word in text.split() if len(word) >= 4 and word not in STOP_WORDS]
    token_counts = Counter(tokens)
    outcome_labels = [
        label for label, phrases in OUTCOME_PHRASES.items()
        if outcome_present(text, label)
    ]
    population_labels = [
        label for label, phrases in POPULATION_SYNONYMS.items()
        if matched_phrases(text, list(phrases))
    ]
    # Prefer the specific population phrase over a generic phrase nested in it.
    if "older adults" in population_labels and "adults" in population_labels:
        population_labels.remove("adults")
    intervention_labels = matched_phrases(text, list(INTERVENTION_TERMS))
    ranking = any(marker in text for marker in (
        "which treatment", "which treatments", "most effective", "best treatment",
        "compare", "compared", "versus", " vs ",
    ))
    prevention = any(marker in text for marker in ("prevent", "prevention", "reduce risk"))
    # Domain tokens remain auditable and are not treated as exact medical concepts.
    concepts = [word for word, _ in token_counts.most_common(12)]
    return {
        "question": question,
        "population_cues": population_labels,
        "intervention_cues": intervention_labels,
        "comparison_requested": ranking,
        "outcomes_requested": outcome_labels,
        "prevention_requested": prevention,
        "domain_terms": concepts,
        "extraction_method": "transparent rule-based candidate extraction",
    }


def _condition_terms(profile: dict[str, Any]) -> list[str]:
    excluded = {
        "older", "adult", "adults", "treatment", "treatments", "therapy", "therapies",
        "prevention", "prevent", "effective", "effectively", "fracture", "fractures",
        "hip", "vertebral", "which", "most", "reduce", "risk", "compare", "compared",
    }
    return [term for term in profile["domain_terms"] if term not in excluded]


def _source_text(article: dict[str, Any]) -> str:
    return normalise(" ".join([
        article.get("title", ""), article.get("abstract", ""),
        " ".join(article.get("publication_types", []) if isinstance(article.get("publication_types"), list) else []),
    ]))


def _article_types(article: dict[str, Any], text: str) -> list[str]:
    labels = []
    types = normalise(" ".join(article.get("publication_types", []) if isinstance(article.get("publication_types"), list) else []))
    combined = text + " " + types
    if "guideline" in combined or "practice guideline" in combined:
        labels.append("guideline")
    if "systematic review" in combined or "meta analysis" in combined or "meta-analysis" in combined:
        labels.append("systematic_review")
    if "randomized controlled trial" in combined or "randomised controlled trial" in combined:
        labels.append("randomized_trial")
    if not labels:
        labels.append("other_or_unclear")
    return labels


def _requested_population_match(text: str, profile: dict[str, Any]) -> tuple[bool, list[str]]:
    cues = profile["population_cues"]
    if not cues:
        return True, []
    matched = []
    for cue in cues:
        synonyms = POPULATION_SYNONYMS[cue]
        if matched_phrases(text, list(synonyms)):
            matched.append(cue)
    # A record needs to match one requested population group. Requiring every
    # overlapping cue would make "elderly" fail when a question says "older adults".
    return bool(matched), matched


def screen_articles(articles: list[dict[str, Any]], profile: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Classify candidates. A direct candidate still requires later verification."""
    condition_terms = _condition_terms(profile)
    screened, audit = [], []
    for article in articles:
        text = _source_text(article)
        condition_matches = [term for term in condition_terms if contains_phrase(text, term)]
        population_ok, population_matches = _requested_population_match(text, profile)
        outcome_matches = [
            label for label in profile["outcomes_requested"]
            if outcome_present(text, label)
        ]
        interventions = matched_phrases(text, list(INTERVENTION_TERMS))
        comparative = any(marker in text for marker in (
            " compared ", " versus ", " vs ", "randomized", "randomised", "controlled trial",
            "superior to", "noninferior", "non inferior", "network meta analysis", "network meta-analysis",
        ))
        has_abstract = bool(article.get("abstract", "").strip())
        # Condition + requested outcome provides the minimum candidate standard.
        # Where no condition term could be derived, do not pretend relevance is certain.
        direct_candidate = bool(condition_matches) and bool(outcome_matches) and population_ok and has_abstract
        score = (
            3 * len(condition_matches) + 2 * len(outcome_matches) +
            int(population_ok) + int(bool(interventions)) + int(comparative)
        )
        if direct_candidate:
            status, reason = "direct_candidate", "Matches condition candidate terms and at least one requested outcome."
        elif not has_abstract:
            status, reason = "unclear", "No abstract is available for automated candidate screening."
        elif not condition_matches:
            status, reason = "supplementary", "Does not match the derived condition candidate terms."
        elif not outcome_matches:
            status, reason = "supplementary", "Does not mention a requested outcome in title, abstract, or publication type."
        elif not population_ok:
            status, reason = "supplementary", "Does not match the population cue in its title, abstract, or publication type."
        else:
            status, reason = "unclear", "Rule-based screening could not classify the record confidently."
        enriched = dict(article)
        enriched["pico_screening"] = {
            "status": status, "reason": reason, "candidate_score": score,
            "condition_matches": condition_matches, "population_matches": population_matches,
            "outcome_matches": outcome_matches, "intervention_mentions": interventions,
            "comparative_language": comparative, "evidence_type_candidates": _article_types(article, text),
        }
        screened.append(enriched)
        audit.append({
            "article_id": article["article_id"], "title": article.get("title", ""),
            **enriched["pico_screening"],
        })
    return screened, audit


def assess_coverage(screened_articles: list[dict], profile: dict[str, Any]) -> dict[str, Any]:
    direct = [a for a in screened_articles if a["pico_screening"]["status"] == "direct_candidate"]
    direct_outcomes = {
        outcome for article in direct for outcome in article["pico_screening"]["outcome_matches"]
    }
    requested_outcomes = set(profile["outcomes_requested"])
    types = {
        evidence_type for article in direct
        for evidence_type in article["pico_screening"]["evidence_type_candidates"]
    }
    intervention_mentions = sorted({
        intervention for article in direct
        for intervention in article["pico_screening"]["intervention_mentions"]
    })
    has_comparative = any(article["pico_screening"]["comparative_language"] for article in direct)
    checks = {
        "direct_candidate_articles": {"pass": len(direct) >= 2, "observed": len(direct), "minimum": 2},
        "requested_outcomes": {
            "pass": requested_outcomes.issubset(direct_outcomes) if requested_outcomes else False,
            "requested": sorted(requested_outcomes), "found": sorted(direct_outcomes),
            "missing": sorted(requested_outcomes - direct_outcomes),
        },
        "comparative_evidence": {
            "pass": has_comparative if profile["comparison_requested"] else True,
            "required": profile["comparison_requested"], "observed": has_comparative,
        },
        "intervention_variety": {
            "pass": len(intervention_mentions) >= 2 if profile["comparison_requested"] else bool(intervention_mentions),
            "observed": intervention_mentions,
            "minimum": 2 if profile["comparison_requested"] else 1,
        },
        "evidence_type_diversity": {
            "pass": bool(types), "observed": sorted(types),
        },
    }
    failed = [name for name, check in checks.items() if not check["pass"]]
    return {
        "coverage_status": "READY_FOR_EVIDENCE_EXTRACTION" if not failed else "RETRIEVE_MORE",
        "checks": checks,
        "failed_checks": failed,
        "direct_candidate_article_ids": [article["article_id"] for article in direct],
        "direct_candidate_count": len(direct),
        "clinical_answer_allowed": False,
        "limitations": [
            "Coverage checks are retrieval controls, not proof of clinical completeness or correctness.",
            "PICO status is based on title, abstract, and publication metadata. It needs semantic review against full source passages.",
            "An intervention mention does not prove efficacy, safety, or superiority.",
            "A READY_FOR_EVIDENCE_EXTRACTION result cannot release an answer until claim verification is added.",
        ],
    }


def add_pico_and_coverage(retrieval_report: dict[str, Any], research_question: str) -> dict[str, Any]:
    if retrieval_report.get("stage") != "retrieval_only":
        raise ValueError("Expected a Stage 1 retrieval-only audit")
    if not str(research_question).strip():
        raise ValueError("A nonempty natural-language research question is required")
    articles = retrieval_report.get("articles")
    if not isinstance(articles, list):
        raise ValueError("Retrieval audit has no article list")
    profile = extract_question_profile(research_question)
    screened, audit = screen_articles(articles, profile)
    updated = dict(retrieval_report)
    updated["stage"] = "pico_and_coverage"
    updated["research_question"] = research_question
    updated["pico_profile"] = profile
    updated["pico_audit"] = audit
    updated["articles"] = screened
    updated["coverage"] = assess_coverage(screened, profile)
    updated["clinical_answer_allowed"] = False
    updated["coverage_assessed"] = True
    return updated
