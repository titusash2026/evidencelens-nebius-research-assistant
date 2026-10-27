"""Stage 4: Nemotron claim drafting plus deterministic evidence checks."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from pico_coverage import POPULATION_SYNONYMS, outcome_present


MODEL_ID = "nvidia/Nemotron-3-Ultra-550b-a55b"
DEFAULT_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1"
MAX_EVIDENCE_PASSAGES = 12
MAX_PASSAGE_CHARS = 1200
# JSON-object mode is supported by the selected Token Factory endpoint.  The
# detailed claim schema is enforced deterministically by _parse_json_response
# and verify_claims after the response is received.
CLAIM_RESPONSE_FORMAT = {"type": "json_object"}
# Prefer a concrete schema. Some OpenAI-compatible deployments decline
# json_schema requests, so _call_nemotron falls back once to json_object mode.
CLAIM_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidencelens_claims",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["claims"],
            "properties": {
                "claims": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["claim", "evidence_ids", "supporting_quote"],
                        "properties": {
                            "claim": {"type": "string", "minLength": 1},
                            "evidence_ids": {
                                "type": "array", "minItems": 1,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "supporting_quote": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
        },
    },
}
CLAIM_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
}
OUTCOME_PATTERNS = {
    "hip fracture": r"\bhip fractures?\b",
    "vertebral fracture": r"\bvertebral fractures?\b",
}
EVIDENCE_WIDE_PATTERN = r"\b(?:the )?(?:supplied|all|overall|available) evidence\b"
RANKING_CLAIM_PATTERN = r"\b(?:cannot rank|can not rank|most effective|best treatment|ranked|ranking)\b"
RANKING_SUPPORT_PATTERN = r"\b(?:rank(?:ed|ing)?|network meta[ -]?analysis|league table|sucra|p[ -]?score|head to head)\b"
COMPARATIVE_SUPPORT_PATTERN = (
    r"\b(?:rank(?:ed|ing)?|superior|more effective|better|compared|versus|vs|"
    r"head to head|network meta[ -]?analysis|league table|sucra|p[ -]?score|"
    r"relative risk|odds ratio|hazard ratio)\b"
)
LIMITATION_PATTERN = r"\b(?:limit(?:ed|ing|ation)?|insufficient|uncertain|absent|lack(?:ing)?|short follow up|cannot|unable)\b"
TREATMENT_ENTITY_TERMS = (
    "alendronate", "risedronate", "zoledronate", "zoledronic acid", "ibandronate",
    "denosumab", "romosozumab", "teriparatide", "abaloparatide", "raloxifene",
    "bisphosphonate", "antiresorptive", "anabolic", "hormone replacement",
    "estrogen", "oestrogen", "calcitonin",
)


class ModelResponseError(Exception):
    pass


def _normalise(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _evidence_sources(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    passages = packet.get("passages", [])
    if not isinstance(passages, list):
        raise ValueError("Evidence packet has an invalid passage list")
    sources = {}
    for index, item in enumerate(passages[:MAX_EVIDENCE_PASSAGES], start=1):
        if not isinstance(item, dict) or not str(item.get("passage", "")).strip():
            continue
        sources[f"E{index}"] = item
    return sources


def _model_messages(report: dict[str, Any], sources: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    evidence = []
    for source_id, item in sources.items():
        evidence.append({
            "id": source_id,
            "title": item.get("title", ""),
            "outcomes": item.get("outcomes", []),
            "passage": str(item["passage"])[:MAX_PASSAGE_CHARS],
        })
    question = report.get("research_question", "")
    return [
        {
            "role": "system",
            "content": (
                "You are EvidenceLens Stage 4. Use only the supplied evidence passages. "
                "Do not use external knowledge. Do not claim a treatment is most effective "
                "unless a supplied passage explicitly supports comparative ranking. "
                "Return exactly one JSON object, beginning with { and ending with }."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "research_question": question,
                "required_output": {
                    "claims": [{
                        "claim": "one source-grounded clinical evidence statement",
                        "evidence_ids": ["E1"],
                        "supporting_quote": "an exact quote from one selected evidence passage",
                    }],
                },
                "rules": [
                    "Every claim needs at least one evidence ID.",
                    "supporting_quote must be copied exactly from the cited passage.",
                    "If no passage supports a ranking, state that the supplied evidence cannot rank treatments.",
                    "Return no markdown and no fields other than claims.",
                    "Your complete response must have this shape: {\"claims\":[{\"claim\":\"...\",\"evidence_ids\":[\"E1\"],\"supporting_quote\":\"...\"}]}.",
                ],
                "evidence": evidence,
            }, ensure_ascii=False),
        },
    ]


def _parse_json_response(content: Any) -> dict[str, Any]:
    # Some OpenAI-compatible endpoints return content parts rather than one
    # plain string. Preserve only text parts before parsing.
    if isinstance(content, list):
        text = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    else:
        text = str(content or "").strip()

    # Nemotron is instructed to return JSON only, but a model can still wrap a
    # valid object in prose or a Markdown fence. Locate and decode one complete
    # JSON object without accepting Python-style or otherwise malformed data.
    decoder = json.JSONDecoder()
    payload: Any = None
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "claims" in candidate:
            payload = candidate
            break
    if payload is None:
        raise ModelResponseError("Nemotron did not return a JSON object containing claims")
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        raise ModelResponseError("Nemotron response has an invalid claim schema")
    if not 1 <= len(payload["claims"]) <= 8:
        raise ModelResponseError("Nemotron returned an invalid number of claims")
    return payload


def _repair_messages(
    report: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    rejected_decisions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Give Nemotron one auditable opportunity to correct rejected claims."""
    cited_ids = {
        source_id
        for item in rejected_decisions
        for source_id in item.get("evidence_ids", [])
        if source_id in sources
    }
    repair_evidence = [
        {
            "id": source_id,
            "title": item.get("title", ""),
            "outcomes": item.get("outcomes", []),
            "passage": str(item.get("passage", ""))[:MAX_PASSAGE_CHARS],
        }
        for source_id, item in sources.items()
        if source_id in cited_ids
    ]
    feedback = [
        {
            "claim": item.get("claim", ""),
            "evidence_ids": item.get("evidence_ids", []),
            "reason": item.get("reason", ""),
        }
        for item in rejected_decisions
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are EvidenceLens Stage 4 claim repair. Use only the supplied cited passages. "
                "Return exactly one JSON object and no Markdown."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
            "research_question": report.get("research_question", ""),
            "task": "Repair the previous response. Return a replacement claims object only.",
            "rejected_claims": feedback,
            "cited_evidence": repair_evidence,
            "repair_rules": [
                "Every claim must be narrowly scoped to what its cited passage states.",
                "For a single-study limitation, write 'In the cited study' rather than making a conclusion about all supplied evidence.",
                "Do not add a population, outcome, treatment ranking, or evidence-wide conclusion unless that concept appears in the cited passage.",
                "Never state that all supplied evidence cannot rank treatments unless the cited passage explicitly says that.",
                "Use at least three substantive terms that also occur in the cited passage.",
                "Use an exact supporting_quote copied from the cited passage.",
                "Return only {\"claims\":[...]} with one to eight claims.",
            ],
        }, ensure_ascii=False),
        },
    ]


async def _call_nemotron(messages: list[dict[str, str]], timeout_seconds: float) -> dict[str, Any]:
    api_key = os.getenv("NEBIUS_API_KEY") or os.getenv("NEBIUS_TOKEN_FACTORY_KEY")
    if not api_key:
        raise ModelResponseError("NEBIUS_API_KEY or NEBIUS_TOKEN_FACTORY_KEY is not set")
    base_url = os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    retry_messages = messages
    format_errors: list[str] = []
    # First request a schema-constrained response. If a model/deployment does
    # not accept that feature or still misses the schema, retry exactly once in
    # the widely supported JSON-object mode with an explicit corrective prompt.
    for attempt, response_format in enumerate((CLAIM_SCHEMA_RESPONSE_FORMAT, CLAIM_RESPONSE_FORMAT)):
        payload = {
            "model": os.getenv("NEBIUS_MODEL", MODEL_ID),
            "messages": retry_messages,
            "temperature": 0,
            "max_tokens": 1400,
            # JSON-object mode is enforced by Token Factory; local validation below
            # enforces the stricter EvidenceLens claim schema.
            "response_format": response_format,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds), follow_redirects=False) as client:
                response = await client.post(
                    base_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ModelResponseError(f"Nemotron request failed: {type(exc).__name__}") from exc
        if response.status_code != 200:
            if attempt == 0 and response.status_code in {400, 404, 422}:
                format_errors.append(f"json_schema request returned HTTP {response.status_code}")
                retry_messages = messages + [{
                    "role": "user",
                    "content": (
                        "Use JSON-object mode. Return only the required object with the exact key "
                        "'claims', containing one to eight claim objects."
                    ),
                }]
                continue
            raise ModelResponseError(f"Nemotron returned HTTP {response.status_code}")
        try:
            body = response.json()
            message = body["choices"][0]["message"]
            refusal = str(message.get("refusal") or "").strip()
            if refusal:
                raise ModelResponseError("Nemotron refused the structured-output request")
            content = message["content"]
        except (TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ModelResponseError("Nemotron response has an unexpected API schema") from exc
        try:
            return _parse_json_response(content)
        except ModelResponseError as exc:
            format_errors.append(str(exc))
            if attempt:
                detail = "; ".join(format_errors)
                raise ModelResponseError(f"Nemotron structured-output recovery failed: {detail}") from exc
            retry_messages = messages + [{
                "role": "user",
                "content": (
                    "Your previous response did not contain the required claims object. Return only one "
                    "valid JSON object with exactly this top-level key: claims. Each claims item must have "
                    "claim, evidence_ids, and supporting_quote. Do not include explanation, Markdown, or reasoning."
                ),
            }]
    raise AssertionError("Unreachable Nemotron format retry state")


def _claim_tokens(value: str) -> set[str]:
    return {word for word in _normalise(value).split() if len(word) >= 4 and word not in CLAIM_STOP_WORDS}


def _scope_violation(claim: str, quote: str) -> str | None:
    """Reject conclusions broader than the exact quote can establish."""
    claim_text = _normalise(claim)
    quote_text = _normalise(quote)
    if re.search(EVIDENCE_WIDE_PATTERN, claim_text):
        return "A single cited passage cannot support a corpus-wide evidence conclusion."
    for outcome, pattern in OUTCOME_PATTERNS.items():
        if re.search(pattern, claim_text) and not re.search(pattern, quote_text):
            return f'Claim names "{outcome}" but the exact supporting quote does not.'
    if re.search(RANKING_CLAIM_PATTERN, claim_text) and not re.search(RANKING_SUPPORT_PATTERN, quote_text):
        return "Claim makes a treatment-ranking conclusion without ranking language in the exact supporting quote."
    return None


def verify_claims(candidate: dict[str, Any], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify citations, exact quotes, and minimum lexical alignment."""
    decisions = []
    for index, item in enumerate(candidate["claims"], start=1):
        if not isinstance(item, dict):
            decisions.append({"claim_index": index, "status": "rejected", "reason": "Claim is not an object."})
            continue
        claim = str(item.get("claim", "")).strip()
        quote = str(item.get("supporting_quote", "")).strip()
        evidence_ids = item.get("evidence_ids")
        if not claim or not quote or not isinstance(evidence_ids, list) or not evidence_ids:
            decisions.append({"claim_index": index, "status": "rejected", "reason": "Claim, quote, or evidence IDs are missing."})
            continue
        valid_ids = [source_id for source_id in evidence_ids if source_id in sources]
        quoted_sources = [source_id for source_id in valid_ids if _normalise(quote) in _normalise(sources[source_id]["passage"])]
        claim_words = _claim_tokens(claim)
        supporting_words = set().union(*(_claim_tokens(sources[source_id]["passage"]) for source_id in quoted_sources)) if quoted_sources else set()
        if not quoted_sources:
            status, reason = "rejected", "Supporting quote does not exactly occur in a cited passage."
        elif scope_reason := _scope_violation(claim, quote):
            status, reason = "rejected", scope_reason
        elif len(claim_words & supporting_words) < 3:
            status, reason = "rejected", "Claim has insufficient lexical alignment with its cited quote."
        else:
            status, reason = "supported", "Cited evidence ID, exact quote, and lexical alignment passed."
        decisions.append({
            "claim_index": index, "claim": claim, "status": status, "reason": reason,
            "evidence_ids": valid_ids, "supporting_quote": quote,
        })
    return decisions


def _has_term(text: str, term: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None


def _condition_terms(profile: dict[str, Any]) -> list[str]:
    excluded = {
        "older", "adult", "adults", "treatment", "treatments", "therapy", "therapies",
        "prevention", "prevent", "effective", "effectively", "fracture", "fractures",
        "hip", "vertebral", "which", "most", "reduce", "risk", "compare", "compared",
    }
    return [term for term in profile.get("domain_terms", []) if term not in excluded]


def assess_question_answer_coverage(
    report: dict[str, Any], decisions: list[dict[str, Any]], sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Gate answer release on the user's requested conclusion, not one valid claim."""
    profile = report.get("pico_profile", {})
    supported = [item for item in decisions if item.get("status") == "supported"]
    cited = [
        (item, source_id, sources[source_id])
        for item in supported
        for source_id in item.get("evidence_ids", [])
        if source_id in sources
    ]
    claim_text = _normalise(" ".join(str(item.get("claim", "")) for item in supported))
    quote_text = _normalise(" ".join(str(item.get("supporting_quote", "")) for item in supported))
    source_text = _normalise(" ".join(str(source.get("passage", "")) for _, _, source in cited))

    treatment_mentions = sorted({
        term for term in TREATMENT_ENTITY_TERMS if _has_term(source_text, term)
    })
    requested_outcomes = set(profile.get("outcomes_requested", []))
    found_outcomes = sorted({
        outcome for outcome in requested_outcomes if outcome_present(quote_text, outcome)
    })
    population_cues = profile.get("population_cues", [])
    matched_population = sorted({
        cue for cue in population_cues
        if any(_has_term(source_text, synonym) for synonym in POPULATION_SYNONYMS.get(cue, (cue,)))
    })
    condition_terms = _condition_terms(profile)
    matched_conditions = sorted({term for term in condition_terms if _has_term(source_text, term)})
    comparative_sources = sorted({
        source_id for _, source_id, source in cited
        if re.search(COMPARATIVE_SUPPORT_PATTERN, _normalise(source.get("passage", "")))
    })
    limitation_present = bool(re.search(LIMITATION_PATTERN, claim_text + " " + quote_text))

    checks = {
        "named_treatment_or_class": {
            "pass": bool(treatment_mentions), "found": treatment_mentions,
        },
        "source_supported_comparative_or_ranking": {
            "pass": bool(comparative_sources), "source_ids": comparative_sources,
        },
        "requested_outcomes": {
            "pass": requested_outcomes.issubset(found_outcomes),
            "requested": sorted(requested_outcomes), "found": found_outcomes,
            "missing": sorted(requested_outcomes - set(found_outcomes)),
        },
        "population_alignment": {
            "pass": (not population_cues or bool(matched_population)) and (not condition_terms or bool(matched_conditions)),
            "requested_population": population_cues, "matched_population": matched_population,
            "required_condition_terms": condition_terms, "matched_condition_terms": matched_conditions,
        },
    }
    core_failures = [name for name, check in checks.items() if not check["pass"]]
    checks["limitation_statement_if_incomplete"] = {
        "pass": limitation_present if core_failures else True,
        "required": bool(core_failures), "present": limitation_present,
    }
    failures = [name for name, check in checks.items() if not check["pass"]]
    return {
        "status": "READY_FOR_ANSWER_RENDERING" if not failures else "INSUFFICIENT_FOR_REQUESTED_CONCLUSION",
        "answer_ready": not failures,
        "checks": checks,
        "failed_checks": failures,
        "limitation": (
            "Verified claims do not collectively support the requested treatment conclusion; show only limitations and evidence gaps."
            if failures else "Verified claims collectively cover the requested conclusion."
        ),
    }


def build_evidence_gap_report(report: dict[str, Any]) -> dict[str, Any]:
    """Stage 4.3: make an unreleased answer's missing evidence explicit."""
    verification = report.get("claim_verification", {})
    coverage = verification.get("question_answer_coverage", {})
    failed_checks = coverage.get("failed_checks", [])
    if verification.get("status") in {"MODEL_ERROR", "BLOCKED_EVIDENCE_PACKET_INCOMPLETE"}:
        return {
            "status": "VERIFICATION_UNAVAILABLE",
            "answer_released": False,
            "message": "No treatment-ranking answer was released because claim verification was unavailable.",
            "missing_answer_elements": [],
            "verification_reason": verification.get("reason", "Claim verification did not complete."),
            "safe_display_text": "Verification unavailable: no clinical conclusion can be displayed.",
        }
    if coverage.get("answer_ready") and verification.get("status") == "VERIFIED":
        return {
            "status": "NOT_REQUIRED", "answer_released": True,
            "message": "The verified claim set covers the requested conclusion.",
            "missing_answer_elements": [],
        }
    retrieval = report.get("coverage_gap_retrieval_round", {})
    return {
        "status": "EVIDENCE_GAP_REPORTED",
        "answer_released": False,
        "message": "No treatment-ranking answer was released because the verified claim set does not cover the requested conclusion.",
        "missing_answer_elements": failed_checks,
        "coverage_gate_limitation": coverage.get("limitation", "Question-answer coverage was not established."),
        "coverage_gap_retrieval": {
            "attempted": retrieval.get("attempted", False),
            "status": retrieval.get("status", "not_run"),
            "query": retrieval.get("query", ""),
            "new_unique_articles_after_merge": retrieval.get("new_unique_articles_after_merge", 0),
        },
        "safe_display_text": "Evidence gap: no verified, source-supported comparative conclusion can be displayed for this question.",
    }


async def run_claim_verification(report: dict[str, Any], timeout_seconds: float = 90.0) -> dict[str, Any]:
    """Run Stage 4; models are never called for an incomplete evidence packet."""
    if report.get("stage") not in {"evidence_extraction", "fulltext_evidence"}:
        raise ValueError("Expected a Stage 3, Stage 3.6, or Stage 3.7 evidence audit")
    packet = report.get("evidence_packet", {})
    updated = dict(report)
    if packet.get("status") != "READY_FOR_CLAIM_VERIFICATION":
        updated["claim_verification"] = {
            "status": "BLOCKED_EVIDENCE_PACKET_INCOMPLETE",
            "model_called": False,
            "reason": "Stage 3 must provide comparative and ranking evidence before Nemotron can draft claims.",
        }
        updated["clinical_answer_allowed"] = False
        updated["evidence_gap_report"] = build_evidence_gap_report(updated)
        return updated
    sources = _evidence_sources(packet)
    if not sources:
        raise ValueError("Evidence packet has no usable passages")
    try:
        candidate = await _call_nemotron(_model_messages(report, sources), timeout_seconds)
        decisions = verify_claims(candidate, sources)
    except ModelResponseError as exc:
        updated["claim_verification"] = {
            "status": "MODEL_ERROR", "model_called": True,
            "reason": str(exc),
        }
        updated["clinical_answer_allowed"] = False
        updated["evidence_gap_report"] = build_evidence_gap_report(updated)
        return updated

    initial_decisions = decisions
    rejected = [item for item in decisions if item["status"] != "supported"]
    repair_error = ""
    if rejected:
        try:
            candidate = await _call_nemotron(
                _repair_messages(report, sources, rejected), timeout_seconds,
            )
            decisions = verify_claims(candidate, sources)
        except ModelResponseError as exc:
            # Preserve the initial decisions and keep release blocked. A failed
            # repair attempt must not obscure why the first draft was rejected.
            repair_error = str(exc)
            decisions = initial_decisions
    approved = [item for item in decisions if item["status"] == "supported"]
    all_supported = len(approved) == len(decisions) and bool(approved)
    answer_coverage = assess_question_answer_coverage(report, decisions, sources)
    updated["claim_verification"] = {
        "status": "VERIFIED" if all_supported else "REJECTED_OR_PARTIAL",
        "model_called": True,
        "model": os.getenv("NEBIUS_MODEL", MODEL_ID),
        "decisions": decisions,
        "repair_attempted": bool(rejected),
        "initial_decisions": initial_decisions if rejected else [],
        "repair_error": repair_error or None,
        "question_answer_coverage": answer_coverage,
        "verified_evidence_statements": [
            {"claim": item["claim"], "evidence_ids": item["evidence_ids"]}
            for item in approved
        ],
        "limitations": [
            "Verification checks citation identity, an exact supporting quote, and lexical alignment; it is not a substitute for clinical expert review.",
            "Only verified evidence statements may be displayed. No uncited model narrative is released.",
        ],
    }
    updated["clinical_answer_allowed"] = all_supported and answer_coverage["answer_ready"]
    updated["evidence_gap_report"] = build_evidence_gap_report(updated)
    return updated
