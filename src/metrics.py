from collections import Counter


def calculate_run_metrics(
    sources: list[dict],
    verification_items: list[dict],
    elapsed_seconds: float,
) -> dict:
    """Calculate transparent, per-run EvidenceLens metrics."""

    status_counts = Counter(
        item.get("status", "Insufficient evidence")
        for item in verification_items
    )

    return {
        "Sources retrieved": len(sources),
        "Claims checked": len(verification_items),
        "Supported claims": status_counts.get("Supported", 0),
        "Conflicting claims": status_counts.get("Conflicting", 0),
        "Insufficient evidence": status_counts.get(
            "Insufficient evidence",
            0,
        ),
        "Runtime seconds": round(elapsed_seconds, 1),
    }