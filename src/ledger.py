"""Append-only audit trail for one EvidenceLens research run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class LedgerEvent:
    sequence: int
    event_type: str
    payload: dict[str, Any]
    timestamp_utc: str


@dataclass
class EvidenceLedger:
    """Stores only controller decisions and source/claim provenance."""

    run_id: str
    events: list[LedgerEvent] = field(default_factory=list)

    def record(self, event_type: str, **payload: Any) -> None:
        self.events.append(
            LedgerEvent(
                sequence=len(self.events) + 1,
                event_type=event_type,
                payload=payload,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )

    def record_source(self, source_id: int, source: dict[str, Any]) -> None:
        self.record(
            "source_assessed",
            source_id=source_id,
            title=str(source.get("title", "Untitled source")),
            url=str(source.get("url", "")),
            evidence_role=str(source.get("pico_status", "Direct")),
            role_rationale=str(source.get("pico_reason", "")),
            evidence_text_status=str(source.get("evidence_text_status", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events": [asdict(event) for event in self.events],
        }

