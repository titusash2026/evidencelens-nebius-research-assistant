"""High-confidence PII detection for EvidenceLens research questions."""

from __future__ import annotations

import re


PII_PATTERNS: dict[str, str] = {
    "email address": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "phone number": (
        r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?"
        r"(?:\(?\d{2,4}\)?[ .-]?)?"
        r"\d{3,4}[ .-]\d{4}(?!\w)"
    ),
    "Aadhaar-like number": r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)",
    "medical record or patient ID": (
        r"\b(?:mrn|uhid|patient\s*(?:id|number)|"
        r"medical\s*record\s*(?:id|number))"
        r"\s*[:#-]?\s*[A-Z0-9-]{4,}\b"
    ),
    "date of birth": (
        r"\b(?:dob|date\s+of\s+birth)\s*[:#-]?\s*"
        r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    ),
}


def detect_pii(text: str) -> list[str]:
    """Return labels for high-confidence PII patterns found in text."""
    if not text:
        return []

    return [
        label
        for label, pattern in PII_PATTERNS.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def pii_block_message(pii_types: list[str]) -> str:
    """Create a safe message without repeating the submitted identifiers."""
    detected = ", ".join(pii_types)

    return (
        "EvidenceLens detected possible personal or patient information "
        f"({detected}). Please remove identifying details and use only "
        "anonymous, group-level research information."
    )