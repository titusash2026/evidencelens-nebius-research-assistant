def generate_research_plan(client, model: str, question: str) -> list[str]:
    """
    Use NVIDIA Nemotron to break one broad research question into
    three focused evidence-search questions.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical-AI research planning agent. "
                    "Turn the user's research question into exactly three focused, "
                    "searchable evidence questions. "
                    "Return only three short lines. Do not give medical advice."
                ),
            },
            {
                "role": "user",
                "content": f"Research question: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=250,
    )

    raw_plan = response.choices[0].message.content or ""

    plan_items = []
    for line in raw_plan.splitlines():
        cleaned_line = line.strip().lstrip("-•*0123456789. )").strip()

        if cleaned_line:
            plan_items.append(cleaned_line)

    return plan_items[:3] or [question]