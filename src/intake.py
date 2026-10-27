from __future__ import annotations

import json
import re
from typing import Any


MAX_PLANNING_ATTEMPTS = 2
MAX_SEARCH_QUERIES = 3


class ResearchPlanError(RuntimeError):
    """Raised when a Nemotron research plan is invalid or unavailable."""


def _clean_text(value: Any, field_name: str, max_length: int = 500) -> str:
    text = str(value or "").strip()

    if not text:
        raise ResearchPlanError(f"Missing required field: {field_name}.")

    return text[:max_length]


def _clean_string_list(
    value: Any,
    field_name: str,
    minimum: int,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list):
        raise ResearchPlanError(f"{field_name} must be a JSON list.")

    cleaned: list[str] = []

    for item in value:
        text = str(item or "").strip()
        if text and text not in cleaned:
            cleaned.append(text[:500])

    if not minimum <= len(cleaned) <= maximum:
        raise ResearchPlanError(
            f"{field_name} must contain {minimum} to {maximum} non-empty items."
        )

    return cleaned


def _extract_json_object(raw_response: Any) -> dict[str, Any]:
    """Accept one JSON object even if the model adds a code fence or preface."""
    text = str(raw_response or "").strip()

    if not text:
        raise ResearchPlanError("Nemotron returned an empty response.")

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    decoder = json.JSONDecoder()

    for start in (match.start() for match in re.finditer(r"\{", text)):
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    raise ResearchPlanError("Nemotron did not return a complete JSON object.")


def _normalise_plan(
    payload: dict[str, Any],
    question: str,
    settings: dict[str, str],
    planning_attempts: int,
) -> dict[str, Any]:
    """Validate and normalise only the fields Stage 03 needs from Nemotron."""
    plan_steps = _clean_string_list(payload.get("plan_steps"), "plan_steps", 3, 3)
    search_queries = _clean_string_list(
        payload.get("search_queries"),
        "search_queries",
        1,
        MAX_SEARCH_QUERIES,
    )
    preferred_source_types = _clean_string_list(
        payload.get("preferred_source_types"),
        "preferred_source_types",
        1,
        4,
    )
    verification_requirements = _clean_string_list(
        payload.get("verification_requirements"),
        "verification_requirements",
        3,
        5,
    )

    return {
        "topic": _clean_text(payload.get("topic", question), "topic"),
        "population": _clean_text(
            payload.get("population", "Specified in the research question"),
            "population",
        ),
        "outcomes": _clean_string_list(
            payload.get("outcomes", ["benefits", "harms", "evidence quality"]),
            "outcomes",
            1,
            5,
        ),
        "study_types": [
            settings.get("evidence_preference", "All available evidence")
        ],
        "publication_window": settings.get("publication_window", "All years"),
        "answer_mode": settings.get(
            "answer_format",
            "Cited narrative with comparison",
        ),
        "plan_steps": plan_steps,
        "search_queries": search_queries,
        "preferred_source_types": preferred_source_types,
        "verification_requirements": verification_requirements,
        "planning_mode": "nemotron_schema_validated",
        "planning_attempts": planning_attempts,
        "planning_errors": [],
    }


def _fallback_task(
    question: str,
    settings: dict[str, str],
    errors: list[str],
) -> dict[str, Any]:
    """Safe local plan that preserves the downstream evidence pipeline."""
    evidence_preference = settings.get(
        "evidence_preference",
        "All available evidence",
    )

    return {
        "topic": question,
        "population": "Specified in the research question",
        "outcomes": [
            "benefits",
            "harms",
            "evidence quality",
            "study limitations",
        ],
        "study_types": [evidence_preference],
        "publication_window": settings.get("publication_window", "All years"),
        "answer_mode": settings.get(
            "answer_format",
            "Cited narrative with comparison",
        ),
        "plan_steps": [
            "Identify high-quality reviews and guidelines relevant to the research question.",
            "Compare eligible studies for population, outcomes, benefits, harms, and follow-up.",
            "Check evidence limitations, PICO relevance, conflicts, and certainty before conclusion.",
        ],
        "search_queries": [
            question,
            f"{question} systematic review",
            f"{question} clinical guideline",
        ],
        "preferred_source_types": [
            "clinical guidelines",
            "systematic reviews",
            "randomized controlled trials",
            "prospective cohort studies",
        ],
        "verification_requirements": [
            "study design",
            "population",
            "outcome definition",
            "effect direction",
            "limitations",
        ],
        "planning_mode": "local_safe_fallback",
        "planning_attempts": MAX_PLANNING_ATTEMPTS,
        "planning_errors": errors,
    }


def build_research_task(
    client: Any,
    model: str,
    question: str,
    settings: dict[str, str],
) -> dict[str, Any]:
    """
    Generate and validate a bounded Nemotron research plan.

    If both attempts fail, return a deterministic local plan so retrieval,
    synthesis, verification, and safety controls remain available.
    """
    if client is None:
        return _fallback_task(
            question,
            settings,
            ["Planner client is unavailable; local safe fallback used."],
        )

    system_prompt = """
You are the Stage 03 research-planning component of EvidenceLens.

Create a focused research plan for the supplied question. Return exactly one
valid JSON object. Do not use Markdown, code fences, citations, explanations,
or any keys not requested.

Required JSON shape:
{
  "topic": "string",
  "population": "string",
  "outcomes": ["string"],
  "plan_steps": ["string", "string", "string"],
  "search_queries": ["string", "string", "string"],
  "preferred_source_types": ["string"],
  "verification_requirements": ["string", "string", "string"]
}

Rules:
- plan_steps must contain exactly 3 concise steps.
- search_queries must contain 1 to 3 focused, non-duplicated queries.
- Prefer guidelines and systematic reviews where relevant.
- The plan is metadata only, never medical advice.
""".strip()

    base_request = (
        f"Research question:\n{question}\n\n"
        f"Research settings:\n{json.dumps(settings, ensure_ascii=False)}"
    )

    errors: list[str] = []

    for attempt in range(1, MAX_PLANNING_ATTEMPTS + 1):
        repair_note = ""

        if errors:
            repair_note = (
                "\n\nThe earlier response did not pass validation: "
                f"{errors[-1]}\n"
                "Return a fresh complete JSON object only."
            )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": base_request + repair_note,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=900,
                timeout=30.0,
            )

            raw_response = response.choices[0].message.content
            payload = _extract_json_object(raw_response)

            return _normalise_plan(
                payload=payload,
                question=question,
                settings=settings,
                planning_attempts=attempt,
            )

        except Exception as error:
            errors.append(
                f"Attempt {attempt}: {type(error).__name__}: {str(error)[:220]}"
            )

    return _fallback_task(question, settings, errors)