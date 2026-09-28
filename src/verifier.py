import json


ALLOWED_STATUSES = {
    "Supported",
    "Conflicting",
    "Insufficient evidence",
}


VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verification_items": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "Supported",
                            "Conflicting",
                            "Insufficient evidence",
                        ],
                    },
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "claim",
                    "status",
                    "evidence",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verification_items"],
    "additionalProperties": False,
}


def normalise_status(status: str) -> str:
    """Return one safe evidence-status label."""

    if status in ALLOWED_STATUSES:
        return status

    return "Insufficient evidence"


def verify_claims(client, model: str, answer: str, sources: list[dict]) -> list[dict]:
    """
    Verify up to five claims using Nebius strict JSON-schema output.
    """

    source_context = "\n\n".join(
        [
            f"[{index}] {source.get('title', 'Untitled source')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Content: {source.get('content', '')[:1200]}"
            for index, source in enumerate(sources, start=1)
        ]
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evidence-verification agent. "
                    "Evaluate exactly three factual claims from the research summary "
                    "Keep each claim, evidence, and reason concise—one sentence each."
                    "using only the supplied source material. "
                    "Return the required JSON schema. "
                    "Do not give medical advice."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Research summary:\n{answer}\n\n"
                    f"Source material:\n{source_context}"
                ),
            },
        ],
        # response_format={
        #     "type": "json_schema",
        #     "json_schema": VERIFICATION_SCHEMA,
        # },
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "claim_verification",
                "schema": VERIFICATION_SCHEMA,
                "strict": True,
            },
        },
        temperature=0.0,
        max_tokens=1400,
    )

    raw_result = response.choices[0].message.content or "{}"
    print("\n--- VERIFIER STRUCTURED RESPONSE ---")
    print(raw_result)
    print("--- END VERIFIER STRUCTURED RESPONSE ---\n")

    result_data = json.loads(raw_result)

    verification_items = result_data.get("verification_items", [])
    clean_items = []

    for item in verification_items:
        clean_items.append(
            {
                "claim": item.get("claim", "Claim not available"),
                "status": normalise_status(item.get("status", "")),
                "evidence": item.get("evidence", "No source reference"),
                "reason": item.get("reason", "No explanation available"),
            }
        )

    return clean_items[:5]