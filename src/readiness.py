from typing import Any


def check_system_readiness(
    nebius_key: str | None,
    tavily_key: str | None,
    planner_model: str,
    verifier_model: str,
    request_timeout: float,
    max_retrieval_queries: int,
    max_sources: int,
) -> dict[str, Any]:
    """Run local startup checks without calling external APIs."""

    checks: list[dict[str, str]] = []
    errors: list[str] = []

    def add_check(
        component: str,
        ready: bool,
        ready_detail: str,
        error_detail: str,
    ) -> None:
        checks.append(
            {
                "Component": component,
                "Status": "Ready" if ready else "Needs setup",
                "Details": ready_detail if ready else error_detail,
            }
        )
        if not ready:
            errors.append(error_detail)

    add_check(
        "Nebius Token Factory credential",
        bool(nebius_key and nebius_key.strip()),
        "NEBIUS_API_KEY is available.",
        "Add a valid NEBIUS_API_KEY to the .env file.",
    )
    add_check(
        "Tavily retrieval credential",
        bool(tavily_key and tavily_key.strip()),
        "TAVILY_API_KEY is available.",
        "Add a valid TAVILY_API_KEY to the .env file.",
    )
    add_check(
        "Nemotron planning model",
        bool(planner_model and planner_model.strip()),
        f"Configured model: {planner_model}",
        "Configure a valid Nemotron planning model ID.",
    )
    add_check(
        "Claim-verification model",
        bool(verifier_model and verifier_model.strip()),
        f"Configured model: {verifier_model}",
        "Configure a valid claim-verification model ID.",
    )

    limits_ready = (
        request_timeout > 0
        and max_retrieval_queries >= 1
        and max_sources >= 1
    )
    add_check(
        "Runtime limits",
        limits_ready,
        (
            f"Timeout: {request_timeout:.0f}s; "
            f"retrieval queries: {max_retrieval_queries}; "
            f"maximum sources: {max_sources}."
        ),
        "Runtime limits must be positive numbers.",
    )

    return {
        "ready": not errors,
        "checks": checks,
        "errors": errors,
    }