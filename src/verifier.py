import json


ALLOWED_STATUSES = {
    "Supported",
    "Conflicting",
    "Insufficient evidence",
}


def verify_claims(client, model: str, answer: str, sources: list[dict]) -> list[dict]:
    """
    Check factual claims in the generated summary against the retrieved sources.
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
                    "Evaluate up to five factual claims from the research summary "
                    "using only the supplied source material. "
                    "Return valid JSON only: a list of objects with exactly these "
                    "keys: claim, status, evidence, reason. "
                    "Status must be exactly one of: Supported, Conflicting, "
                    "Insufficient evidence. "
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
        temperature=0.0,
        max_tokens=900,
    )

    raw_result = response.choices[0].message.content or "[]"
    raw_result = raw_result.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        verification_items = json.loads(raw_result)
    except json.JSONDecodeError:
        return []

    clean_items = []

    for item in verification_items:
        status = item.get("status", "Insufficient evidence")

        if status not in ALLOWED_STATUSES:
            status = "Insufficient evidence"

        clean_items.append(
            {
                "claim": item.get("claim", "Claim not available"),
                "status": status,
                "evidence": item.get("evidence", "No source reference"),
                "reason": item.get("reason", "No explanation available"),
            }
        )

    return clean_items[:5]