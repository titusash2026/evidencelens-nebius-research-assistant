import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


RUN_AUDIT_FILE = Path("data/run_records.jsonl")


class RunBudgetError(RuntimeError):
    """Raised when a run cannot be safely recorded and budgeted."""


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RunBudgetError(
            f"{name} must be a positive integer."
        ) from error

    if value < 1:
        raise RunBudgetError(f"{name} must be at least 1.")

    return value


def _append_audit_record(record: dict, audit_file: Path) -> None:
    try:
        audit_file.parent.mkdir(parents=True, exist_ok=True)

        with audit_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    except OSError as error:
        raise RunBudgetError(
            "EvidenceLens could not save the local run record."
        ) from error


def reserve_run_budget(
    *,
    question: str,
    settings: dict[str, str],
    requested_model_calls: int,
    requested_output_tokens: int,
    audit_file: Path = RUN_AUDIT_FILE,
) -> dict:
    """
    Create a privacy-preserving run record and reserve the maximum local budget
    before the first model call.
    """
    if not question.strip():
        raise RunBudgetError("A research question is required before reserving a run.")

    if requested_model_calls < 1 or requested_output_tokens < 1:
        raise RunBudgetError("Requested run budget must contain positive values.")

    allowed_model_calls = _positive_int_from_env(
        "EVIDENCELENS_MAX_MODEL_CALLS_PER_RUN",
        5,
    )
    allowed_output_tokens = _positive_int_from_env(
        "EVIDENCELENS_MAX_OUTPUT_TOKENS_PER_RUN",
        10_400,
    )

    if requested_model_calls > allowed_model_calls:
        raise RunBudgetError(
            "Run blocked: the configured model-call budget cannot cover "
            f"{requested_model_calls} required calls."
        )

    if requested_output_tokens > allowed_output_tokens:
        raise RunBudgetError(
            "Run blocked: the configured output-token budget cannot cover "
            f"{requested_output_tokens:,} reserved tokens."
        )

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    question_hash = hashlib.sha256(
        question.strip().encode("utf-8")
    ).hexdigest()

    run_record = {
        "event": "budget_reserved",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_sha256": question_hash,
        "settings": settings,
        "reserved_model_calls": requested_model_calls,
        "reserved_output_tokens": requested_output_tokens,
        "status": "reserved",
    }

    _append_audit_record(run_record, audit_file)
    return run_record


def mark_run_status(
    run_id: str,
    status: str,
    detail: str = "",
    audit_file: Path = RUN_AUDIT_FILE,
) -> None:
    """Append an auditable status event without saving the question itself."""
    _append_audit_record(
        {
            "event": "status_updated",
            "run_id": run_id,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "detail": detail[:300],
        },
        audit_file,
    )