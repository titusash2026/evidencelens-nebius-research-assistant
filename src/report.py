from datetime import datetime


def escape_markdown_table(value: str) -> str:
    """Keep text safe inside a Markdown table cell."""
    return str(value).replace("|", "/").replace("\n", " ")


def build_markdown_report(
    question: str,
    research_plan: list[str],
    summary: str,
    verification_items: list[dict],
    sources: list[dict],
) -> str:
    """Create an auditable EvidenceLens research report."""

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = [
        "# EvidenceLens Research Report",
        "",
        f"Generated: {created_at}",
        "",
        "> For research and education only. This report is not medical diagnosis, treatment, or patient-specific advice.",
        "",
        "## Research question",
        "",
        question,
        "",
        "## Agent research plan",
        "",
    ]

    for index, plan_question in enumerate(research_plan, start=1):
        report.append(f"{index}. {plan_question}")

    report.extend(
        [
            "",
            "## Evidence-based research summary",
            "",
            summary,
            "",
            "## Claim verification",
            "",
            "| Claim | Status | Evidence | Reason |",
            "|---|---|---|---|",
        ]
    )

    if verification_items:
        for item in verification_items:
            report.append(
                f"| {escape_markdown_table(item.get('claim', ''))} "
                f"| {escape_markdown_table(item.get('status', ''))} "
                f"| {escape_markdown_table(item.get('evidence', ''))} "
                f"| {escape_markdown_table(item.get('reason', ''))} |"
            )
    else:
        report.append("| No structured verification result | N/A | N/A | Review cited sources directly. |")

    report.extend(
        [
            "",
            "## Sources used",
            "",
        ]
    )

    for index, source in enumerate(sources, start=1):
        title = source.get("title", "Untitled source")
        url = source.get("url", "")
        report.append(f"{index}. [{title}]({url})")

    report.extend(
        [
            "",
            "## System transparency",
            "",
            "- Planning, synthesis, and verification: NVIDIA Nemotron via Nebius Token Factory",
            "- Evidence retrieval: Tavily",
            "- Source policy: trusted scientific and medical sources when selected",
        ]
    )

    return "\n".join(report)