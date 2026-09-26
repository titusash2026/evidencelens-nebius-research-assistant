RESEARCH_DISCLAIMER = (
    "For research and education only. EvidenceLens does not provide medical "
    "diagnosis, treatment, medication, or patient-specific advice."
)

PERSONAL_MEDICAL_PATTERNS = [
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
]


def validate_research_question(question: str) -> tuple[bool, str]:
    """Allow research questions and block patient-specific medical requests."""
    cleaned_question = question.strip().lower()

    if not cleaned_question:
        return False, "Please enter a research question."

    if any(pattern in cleaned_question for pattern in PERSONAL_MEDICAL_PATTERNS):
        return (
            False,
            "EvidenceLens supports research questions only. "
            "Please ask about published evidence, not personal medical care.",
        )

    return True, ""


def get_research_disclaimer() -> str:
    """Return the safety statement shown in the user interface."""
    return RESEARCH_DISCLAIMER