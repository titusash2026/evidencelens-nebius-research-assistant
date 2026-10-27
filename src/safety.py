import re


RESEARCH_DISCLAIMER = (
    "For research and education only. EvidenceLens does not provide medical "
    "diagnosis, treatment, medication, or patient-specific advice."
)

MAX_QUESTION_LENGTH = 2000

PERSONAL_MEDICAL_PATTERNS = (
    "i have ",
    "my symptoms",
    "my scan",
    "my ct scan",
    "my diagnosis",
    "what should i take",
    "which medicine",
    "medication dose",
    "treat me",
    "do i have",
)

PROMPT_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all instructions",
    "reveal your system prompt",
    "show your system prompt",
    "developer message",
    "jailbreak",
)

PII_PATTERNS = (
    re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\d)(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){9,12}\d(?!\d)"
    ),
)


def prepare_research_question(question: str) -> str:
    """Clean and validate a research-only question before model inference."""
    cleaned_question = " ".join(question.split())
    lowered_question = cleaned_question.lower()

    if not cleaned_question:
        raise ValueError("Please enter a research question.")

    if len(cleaned_question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Keep the research question under {MAX_QUESTION_LENGTH} characters."
        )

    if any(
        pattern in lowered_question
        for pattern in PERSONAL_MEDICAL_PATTERNS
    ):
        raise ValueError(
            "EvidenceLens supports published research questions, "
            "not personal medical care."
        )

    if any(
        pattern in lowered_question
        for pattern in PROMPT_INJECTION_PATTERNS
    ):
        raise ValueError(
            "Please enter a research question without system-instruction requests."
        )

    if any(pattern.search(cleaned_question) for pattern in PII_PATTERNS):
        raise ValueError(
            "Remove personal contact details before submitting a research question."
        )

    return cleaned_question


def validate_research_question(question: str) -> tuple[bool, str]:
    """Retains compatibility with the previous EvidenceLens safety flow."""
    try:
        prepare_research_question(question)
    except ValueError as error:
        return False, str(error)

    return True, ""


def get_research_disclaimer() -> str:
    """Return the safety statement shown in the user interface."""
    return RESEARCH_DISCLAIMER