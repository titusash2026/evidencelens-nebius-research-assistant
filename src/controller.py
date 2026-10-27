"""Controller-first EvidenceLens workflow.

Nemotron can propose a plan, extract candidate evidence, and draft language.
Only this Python controller can call retrieval, assign source roles, verify
quotes, decide on a repair pass, and release a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from .ledger import EvidenceLedger


class RunStage(str, Enum):
    INTAKE = "INTAKE"
    PLANNED = "PLANNED"
    RETRIEVED = "RETRIEVED"
    TRIAGED = "TRIAGED"
    VERIFIED = "VERIFIED"
    SYNTHESIZED = "SYNTHESIZED"
    RELEASED = "RELEASED"


ALLOWED_TRANSITIONS: dict[RunStage, set[RunStage]] = {
    RunStage.INTAKE: {RunStage.PLANNED},
    RunStage.PLANNED: {RunStage.RETRIEVED},
    RunStage.RETRIEVED: {RunStage.TRIAGED},
    RunStage.TRIAGED: {RunStage.VERIFIED},
    RunStage.VERIFIED: {RunStage.RETRIEVED, RunStage.SYNTHESIZED},
    RunStage.SYNTHESIZED: {RunStage.RELEASED},
    RunStage.RELEASED: set(),
}


@dataclass(frozen=True)
class ControllerPolicy:
    max_retrieval_queries: int = 12
    max_sources: int = 20
    max_repairs: int = 1
    min_verified_claims_for_full_release: int = 1


@dataclass
class ControlledRun:
    run_id: str
    question: str
    settings: dict[str, Any]
    stage: RunStage = RunStage.INTAKE
    repair_attempts: int = 0
    task: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    direct_sources: list[dict[str, Any]] = field(default_factory=list)
    supplementary_sources: list[dict[str, Any]] = field(default_factory=list)
    pico_audit: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    candidate_evidence: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    ledger: EvidenceLedger | None = None


PrepareTask = Callable[[str, dict[str, Any]], dict[str, Any]]
PlanResearch = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
Retrieve = Callable[[str, dict[str, Any], dict[str, Any]], list[dict[str, Any]]]
Triage = Callable[
    [str, list[dict[str, Any]]],
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
]
ExtractEvidence = Callable[
    [str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]], str
]
Verify = Callable[[str, str, list[dict[str, Any]]], dict[str, Any]]
RepairRetrieve = Callable[
    [str, dict[str, Any], dict[str, Any], int, dict[str, Any]],
    list[dict[str, Any]],
]
AssessCoverage = Callable[[str, dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


class ControlledEvidenceLens:
    """A deterministic execution shell around model calls and retrieval tools."""

    def __init__(
        self,
        *,
        prepare_task: PrepareTask,
        plan_research: PlanResearch,
        retrieve: Retrieve,
        triage: Triage,
        extract_evidence: ExtractEvidence,
        verify: Verify,
        assess_coverage: AssessCoverage | None = None,
        repair_retrieve: RepairRetrieve | None = None,
        policy: ControllerPolicy | None = None,
    ) -> None:
        self.prepare_task = prepare_task
        self.plan_research = plan_research
        self.retrieve = retrieve
        self.triage = triage
        self.extract_evidence = extract_evidence
        self.verify = verify
        self.assess_coverage = assess_coverage
        self.repair_retrieve = repair_retrieve
        self.policy = policy or ControllerPolicy()

    def run(self, question: str, settings: dict[str, Any]) -> ControlledRun:
        """Run the full controlled pipeline and return an auditable result."""
        run_id = uuid4().hex
        ledger = EvidenceLedger(run_id=run_id)
        state = ControlledRun(
            run_id=run_id,
            question=question,
            settings=dict(settings),
            ledger=ledger,
        )
        ledger.record("run_started", question=question, settings=state.settings)

        state.task = self._require_mapping(
            self.prepare_task(question, state.settings),
            "prepare_task",
        )
        ledger.record("intake_approved", task=state.task)

        self._transition(state, RunStage.PLANNED)
        proposed_plan = self._require_mapping(
            self.plan_research(state.task, state.settings),
            "plan_research",
        )
        state.plan = self._validate_plan(proposed_plan)
        ledger.record("plan_approved", plan=state.plan)

        self._retrieve_triage_and_verify(state, is_repair=False)

        while (
            not self._full_release_allowed(state)
            and self.repair_retrieve is not None
            and state.repair_attempts < self.policy.max_repairs
        ):
            state.repair_attempts += 1
            ledger.record(
                "repair_authorized",
                attempt=state.repair_attempts,
                reason="No verified claim met the release standard.",
            )
            self._transition(state, RunStage.RETRIEVED)
            repair_sources = self.repair_retrieve(
                state.question,
                state.plan,
                state.verification,
                state.repair_attempts,
                state.settings,
            )
            state.sources = self._deduplicate_sources(
                state.sources + self._require_source_list(repair_sources, "repair_retrieve")
            )
            self._triage_and_verify(state)

        self._transition(state, RunStage.SYNTHESIZED)
        state.report = self._build_controlled_report(state)
        ledger.record("report_synthesized", release_type=state.report["release_type"])

        self._transition(state, RunStage.RELEASED)
        ledger.record("report_released", stage=state.stage.value)
        return state

    def _retrieve_triage_and_verify(self, state: ControlledRun, *, is_repair: bool) -> None:
        self._transition(state, RunStage.RETRIEVED)
        sources = self.retrieve(state.question, state.plan, state.settings)
        state.sources = self._deduplicate_sources(
            self._require_source_list(sources, "retrieve")
        )
        state.ledger.record(
            "retrieval_completed",
            source_count=len(state.sources),
            is_repair=is_repair,
        )
        self._triage_and_verify(state)

    def _triage_and_verify(self, state: ControlledRun) -> None:
        self._transition(state, RunStage.TRIAGED)
        direct, supplementary, pico_audit = self.triage(state.question, state.sources)
        state.direct_sources = self._require_source_list(direct, "triage direct sources")
        state.supplementary_sources = self._require_source_list(
            supplementary,
            "triage supplementary sources",
        )
        state.pico_audit = list(pico_audit)
        state.coverage = (
            dict(self.assess_coverage(state.question, state.plan, state.direct_sources))
            if self.assess_coverage is not None
            else {"ready_for_full_comparison": True}
        )
        state.sources = self._deduplicate_sources(
            state.direct_sources + state.supplementary_sources
        )
        for index, source in enumerate(state.sources, start=1):
            state.ledger.record_source(index, source)
        state.ledger.record(
            "pico_triage_completed",
            direct_count=len(state.direct_sources),
            supplementary_count=len(state.supplementary_sources),
            audit_rows=state.pico_audit,
            coverage=state.coverage,
        )

        # Nemotron receives the evidence packet, but Python decides whether its
        # resulting claims are admissible.
        state.candidate_evidence = str(
            self.extract_evidence(
                state.question,
                state.plan,
                state.direct_sources,
                state.supplementary_sources,
            )
        ).strip()
        state.ledger.record(
            "candidate_evidence_created",
            character_count=len(state.candidate_evidence),
        )

        self._transition(state, RunStage.VERIFIED)
        state.verification = self._require_mapping(
            self.verify(state.question, state.candidate_evidence, state.sources),
            "verify",
        )
        state.ledger.record(
            "verification_completed",
            verified_claim_count=len(self._verified_items(state)),
            audit=state.verification.get("entailment_audit", []),
        )

    def _validate_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(plan)
        # ``src.intake.build_research_task`` uses ``search_queries``.  Older
        # adapters used ``retrieval_queries`` or ``queries``.  Accept each
        # spelling here, then publish one canonical list for the Python
        # retrieval adapter.  This is an interface normalization step only;
        # the controller still bounds the model-proposed query list.
        raw_queries = normalized.get(
            "search_queries",
            normalized.get("retrieval_queries", normalized.get("queries", [])),
        )
        if not isinstance(raw_queries, list):
            raise ValueError("Controller rejected plan: retrieval queries must be a list.")
        queries = [str(query).strip() for query in raw_queries if str(query).strip()]
        if not queries:
            raise ValueError("Controller rejected plan: no retrieval query was proposed.")
        if len(queries) > self.policy.max_retrieval_queries:
            queries = queries[: self.policy.max_retrieval_queries]
        normalized["retrieval_queries"] = queries
        normalized["search_queries"] = queries
        normalized.pop("queries", None)
        return normalized

    def _full_release_allowed(self, state: ControlledRun) -> bool:
        return (
            len(self._verified_items(state)) >= self.policy.min_verified_claims_for_full_release
            and state.coverage.get("ready_for_full_comparison", True) is True
        )

    def _verified_items(self, state: ControlledRun) -> list[dict[str, Any]]:
        items = state.verification.get("verification_items", [])
        if not isinstance(items, list):
            return []
        return [
            item
            for item in items
            if isinstance(item, dict)
            and item.get("status") == "Supported"
            and item.get("quote_verified") is True
            and item.get("source_proofs")
        ]

    def _build_controlled_report(self, state: ControlledRun) -> dict[str, Any]:
        verified_items = self._verified_items(state)
        source_coverage = []
        for index, source in enumerate(state.sources, start=1):
            source_coverage.append(
                {
                    "source_id": index,
                    "title": source.get("title", "Untitled source"),
                    "url": source.get("url", ""),
                    "role": source.get("pico_status", "Direct"),
                    "reason": source.get("pico_reason", ""),
                    "text_status": source.get("evidence_text_status", ""),
                }
            )
        coverage_ready = state.coverage.get("ready_for_full_comparison", True) is True
        if verified_items and coverage_ready:
            release_type = "full_verified_report"
            headline = "Verified evidence summary"
            conclusion = self._canonical_conclusion(verified_items)

        elif verified_items and not coverage_ready:
            release_type = "coverage_limited_synthesis"
            headline = "Coverage-limited evidence synthesis — full comparison unavailable"

            coverage_reasons = " ".join(state.coverage.get("reasons", [])).strip()
            verified_summary = self._canonical_conclusion(verified_items)
            conclusion = (
        "⚠️ **This is a partial evidence synthesis, not a complete comparison.** "
        "EvidenceLens found source-verified evidence for part of the question, "
        "but the retrieved sources do not adequately cover every requested "
        "treatment or outcome.\n\n"
        "## Source-verified findings\n\n"
        f"{verified_summary}\n\n"
        "## Evidence-coverage gaps\n\n"
        f"{coverage_reasons or 'Some requested treatments or outcomes were not directly covered by the retrieved sources.'}"
    )
        elif len(state.direct_sources) >= 2 and state.candidate_evidence:
            # The model may synthesize only the controller-selected evidence
            # packet.  This is useful to the researcher, but it is not labelled
            # as verified until the strict verbatim-quote gate succeeds.
            release_type = "provisional_grounded_synthesis"
            headline = "Source-grounded research synthesis — verification incomplete"
            conclusion = ("Verification did not establish supported claims. "
                          "The candidate draft is withheld; inspect the sources and retry.")
        else:
            release_type = "partial_report"
            headline = "Evidence availability summary — not a research conclusion"
            conclusion = (
                "No retrieved passage met the strict claim-evidence release gate. "
                "The retained sources and their roles are shown below."
            )
        return {
            "release_type": release_type,
            "headline": headline,
            "conclusion": conclusion,
            "verified_claims": verified_items,
            "source_coverage": source_coverage,
            "pico_audit": state.pico_audit,
            "coverage": state.coverage,
            "entailment_audit": state.verification.get("entailment_audit", []),
            "repair_attempts": state.repair_attempts,
            "run_id": state.run_id,
            "stage": state.stage.value,
            "ledger": state.ledger.as_dict(),
        }

    @staticmethod
    def _canonical_conclusion(verified_items: list[dict[str, Any]]) -> str:
        lines = []
        for item in verified_items:
            sources = ", ".join(
                f"Source {proof['source_id']}"
                for proof in item.get("source_proofs", [])
            )
            lines.append(f"- {item.get('claim', '').strip()} [{sources}]")
        return "\n".join(lines)

    def _transition(self, state: ControlledRun, next_stage: RunStage) -> None:
        if next_stage not in ALLOWED_TRANSITIONS[state.stage]:
            raise RuntimeError(
                f"Controller rejected invalid transition: {state.stage.value} → {next_stage.value}"
            )
        previous_stage = state.stage
        state.stage = next_stage
        state.ledger.record(
            "stage_changed",
            from_stage=previous_stage.value,
            to_stage=next_stage.value,
        )

    def _deduplicate_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                continue
            key = "|".join(
                [
                    str(source.get("url", "")).strip().casefold(),
                    str(source.get("title", "")).strip().casefold(),
                ]
            )
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(dict(source))
            if len(unique) >= self.policy.max_sources:
                break
        return unique

    @staticmethod
    def _require_mapping(value: Any, producer_name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"{producer_name} must return a dictionary.")
        return dict(value)

    @staticmethod
    def _require_source_list(value: Any, producer_name: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise TypeError(f"{producer_name} must return a list of sources.")
        return [dict(source) for source in value if isinstance(source, dict)]

