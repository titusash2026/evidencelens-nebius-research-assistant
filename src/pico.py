from typing import Any


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().replace("-", " ").split())


def _has_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _source_text(source: dict[str, Any]) -> str:
    return _normalise(
        " ".join(
            str(source.get(field, ""))
            for field in ("title", "content", "raw_content", "snippet")
        )
    )


def classify_sources_by_pico(
    question: str,
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """
    Classify evidence into:
    - Direct: eligible for the primary research conclusion.
    - Supplementary: useful background/context, but not eligible to prove
      the primary PICO outcome.
    - Audit rows: transparent explanation for every supplementary source.
    """
    question_text = _normalise(question)

    target_type_2 = _has_any(
        question_text,
        ["type 2 diabetes", "t2d", "type ii diabetes"],
    )
    target_type_1 = _has_any(
        question_text,
        ["type 1 diabetes", "t1d", "type i diabetes"],
    )
    target_adults = _has_any(
        question_text,
        ["adults", "adult patients"],
    )
    target_children = _has_any(
        question_text,
        ["children", "child", "pediatric", "paediatric", "adolescent"],
    )
    target_ipf = _has_any(
        question_text,
        ["idiopathic pulmonary fibrosis", " ipf", "ipf "],
    )

    direct_sources = []
    supplementary_sources = []
    audit_rows = []

    for source in sources:
        text = _source_text(source)
        reasons = []

        has_type_1 = _has_any(
            text, ["type 1 diabetes", "t1d", "type i diabetes"]
        )
        has_type_2 = _has_any(
            text, ["type 2 diabetes", "t2d", "type ii diabetes"]
        )
        has_adult = _has_any(text, ["adult", "adults"])
        has_child = _has_any(
            text,
            ["child", "children", "pediatric", "paediatric", "adolescent"],
        )
        has_ipf = _has_any(
            text,
            ["idiopathic pulmonary fibrosis", " ipf", "ipf "],
        )
        has_other_lung_condition = _has_any(
            text,
            [
                "chronic obstructive pulmonary disease",
                "copd",
                "asthma",
                "cystic fibrosis",
            ],
        )

        if target_type_2 and has_type_1 and not has_type_2:
            reasons.append("Population concerns type 1 diabetes, not type 2 diabetes.")

        if target_type_1 and has_type_2 and not has_type_1:
            reasons.append("Population concerns type 2 diabetes, not type 1 diabetes.")

        if target_adults and has_child and not has_adult:
            reasons.append("Population concerns children/adolescents, not adults.")

        if target_children and has_adult and not has_child:
            reasons.append("Population concerns adults, not children/adolescents.")

        if target_ipf and has_other_lung_condition and not has_ipf:
            reasons.append(
                "Population concerns a different lung disease rather than IPF."
            )

        checked_source = dict(source)

        if reasons:
            checked_source["pico_status"] = "Supplementary"
            checked_source["pico_reason"] = (
                "Relevant as contextual/add-on evidence only. "
                + " ".join(reasons)
            )
            supplementary_sources.append(checked_source)

            audit_rows.append(
                {
                    "title": str(source.get("title", "Untitled source")),
                    "url": str(source.get("url", "")),
                    "role": "Supplementary evidence",
                    "reason": checked_source["pico_reason"],
                }
            )
        else:
            checked_source["pico_status"] = "Direct"
            checked_source["pico_reason"] = (
                "No explicit population mismatch detected."
            )
            direct_sources.append(checked_source)

    return direct_sources, supplementary_sources, audit_rows


def filter_sources_by_pico(
    question: str,
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Compatibility wrapper for older app.py imports.

    It retains both Direct and Supplementary evidence; nothing is silently
    discarded by the PICO stage.
    """
    direct_sources, supplementary_sources, audit_rows = classify_sources_by_pico(
        question=question,
        sources=sources,
    )
    return direct_sources + supplementary_sources, audit_rows