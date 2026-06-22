from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from behavior_lab import __version__
from behavior_lab.core import new_id, stable_hash, to_jsonable, utc_now
from behavior_lab.dsl import Formula
from behavior_lab.ledger import _ExclusiveFileLock
from behavior_lab.offerlab_models.common import (
    FEATURE_CONTRACT,
    FORBIDDEN_MODEL_FIELDS,
    PRODUCTION_EXPORT_ALLOWED,
    enriched_features,
    validate_feature_contract,
)


class ResearchPermissionError(PermissionError):
    pass


class ResearchBudgetError(RuntimeError):
    pass


class ResearchStoreIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class FormulaProposal:
    proposal_id: str
    terms: list[str]
    target_label: str
    falsification: str
    model_family: str = "logistic_formula"
    source: str = "agent"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


StoreGuard = Callable[[list[dict[str, Any]]], None]


class AppendOnlyResearchStore:
    """Small append-only event store with hash chaining and local file locking.

    The store is not a hostile-process security boundary, but concurrent local
    writers cannot both reserve the same scientific budget from a stale view.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        self._memory_lock = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path = self.path.with_suffix(self.path.suffix + ".write.lock")
            if self.path.exists():
                self.events = self._read_file()
                self._verify(self.events)
        else:
            import threading

            self.lock_path = None
            self._memory_lock = threading.RLock()

    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.append_guarded(event_type, payload)

    def append_guarded(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        guard: StoreGuard | None = None,
    ) -> dict[str, Any]:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be non-empty")
        normalized_payload = to_jsonable(payload)
        if self.path:
            assert self.lock_path is not None
            with _ExclusiveFileLock(self.lock_path):
                events = self._read_file()
                self._verify(events)
                if guard is not None:
                    guard(copy.deepcopy(events))
                event = self._build_event(events, event_type, normalized_payload)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self.events = events + [event]
                return copy.deepcopy(event)
        assert self._memory_lock is not None
        with self._memory_lock:
            events = list(self.events)
            self._verify(events)
            if guard is not None:
                guard(copy.deepcopy(events))
            event = self._build_event(events, event_type, normalized_payload)
            self.events.append(event)
            return copy.deepcopy(event)

    def all_events(self) -> list[dict[str, Any]]:
        if self.path:
            assert self.lock_path is not None
            with _ExclusiveFileLock(self.lock_path):
                events = self._read_file()
                self._verify(events)
                self.events = events
        return copy.deepcopy(self.events)

    def by_campaign(self, campaign_id: str) -> list[dict[str, Any]]:
        return [
            event
            for event in self.all_events()
            if event.get("payload", {}).get("campaign_id") == campaign_id
        ]

    def verify(self) -> bool:
        self._verify(self.all_events())
        return True

    def _read_file(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResearchStoreIntegrityError(
                    f"invalid research-store JSON at line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise ResearchStoreIntegrityError(
                    f"research-store event at line {line_number} is not an object"
                )
            events.append(event)
        return events

    @staticmethod
    def _build_event(
        events: list[dict[str, Any]], event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        previous_hash = events[-1]["event_hash"] if events else "GENESIS"
        event = {
            "event_id": new_id("research_event"),
            "event_type": event_type,
            "written_at": utc_now(),
            "payload": payload,
            "previous_hash": previous_hash,
        }
        event["event_hash"] = stable_hash(event)
        return event

    @staticmethod
    def _verify(events: list[dict[str, Any]]) -> None:
        previous_hash = "GENESIS"
        event_ids: set[str] = set()
        for index, event in enumerate(events, start=1):
            required = {
                "event_type",
                "payload",
                "previous_hash",
                "event_hash",
            }
            missing = required - set(event)
            if missing:
                raise ResearchStoreIntegrityError(
                    f"research-store event {index} is missing {sorted(missing)}"
                )
            if event.get("previous_hash") != previous_hash:
                raise ResearchStoreIntegrityError(
                    f"research-store hash chain is broken at event {index}"
                )
            body = dict(event)
            observed_hash = body.pop("event_hash", None)
            if stable_hash(body) != observed_hash:
                raise ResearchStoreIntegrityError(
                    f"research-store event hash mismatch at event {index}"
                )
            event_id = str(event.get("event_id", f"legacy:{index}"))
            if event_id in event_ids:
                raise ResearchStoreIntegrityError(
                    f"duplicate research-store event_id {event_id!r}"
                )
            event_ids.add(event_id)
            previous_hash = str(observed_hash)


class OfferLabResearchAPI:
    """Narrow OfferLab autonomous-research facade.

    Development and hidden budgets are append-only reservations. A reservation
    is consumed before evaluation begins, so a crash cannot create a free retry.
    Hidden submissions are bound to an exact proposal artifact, training data,
    and hidden case set.
    """

    def __init__(
        self,
        *,
        campaign_id: str,
        training_rows: list[dict[str, Any]],
        development_rows: list[dict[str, Any]],
        hidden_rows: list[dict[str, Any]],
        max_formula_terms: int = 8,
        development_evaluations: int = 20,
        hidden_submissions: int = 1,
        store: AppendOnlyResearchStore | None = None,
    ) -> None:
        if not campaign_id.strip():
            raise ValueError("campaign_id is required")
        if max_formula_terms <= 0:
            raise ValueError("max_formula_terms must be positive")
        if development_evaluations < 0 or hidden_submissions < 0:
            raise ValueError("budgets may not be negative")
        self.campaign_id = campaign_id
        self._training_rows = copy.deepcopy(list(training_rows))
        self._development_rows = copy.deepcopy(list(development_rows))
        self._hidden_rows = copy.deepcopy(list(hidden_rows))
        self.max_formula_terms = max_formula_terms
        if store is None:
            raise ValueError("OfferLabResearchAPI requires an explicit file-backed research store")
        elif store.path is None:
            raise ValueError("OfferLabResearchAPI requires a file-backed research store")
        self.store = store
        if not validate_feature_contract(
            self._training_rows + self._development_rows + self._hidden_rows
        ):
            raise ValueError(
                "feature contract contains forbidden future/outcome/participant fields"
            )

        self._training_rows_hash = _rows_hash(self._training_rows)
        self._development_rows_hash = _rows_hash(self._development_rows)
        self._hidden_rows_hash = _rows_hash(self._hidden_rows)
        self._hidden_case_tokens = _case_tokens(self._hidden_rows)
        self._hidden_case_set_hash = stable_hash(self._hidden_case_tokens)
        self._legacy_hidden_case_set_hash = stable_hash(
            sorted(_content_case_token(row) for row in self._hidden_rows)
        )
        self._campaign_fingerprint = stable_hash(
            {
                "training_rows_hash": self._training_rows_hash,
                "development_rows_hash": self._development_rows_hash,
                "hidden_rows_hash": self._hidden_rows_hash,
                "max_formula_terms": max_formula_terms,
                "development_evaluations": development_evaluations,
                "hidden_submissions": hidden_submissions,
            }
        )
        self._pinned_development_budget = development_evaluations
        self._pinned_hidden_budget = hidden_submissions
        self.proposals: dict[str, FormulaProposal] = {}
        self.development_results: dict[str, dict[str, Any]] = {}
        self._restore_and_pin_configuration()

    @property
    def training_rows(self) -> list[dict[str, Any]]:
        return [_public_training_row(row) for row in self._training_rows]

    @property
    def development_rows(self) -> list[dict[str, Any]]:
        raise ResearchPermissionError(
            "development rows are not directly inspectable; use budgeted evaluate_development"
        )

    @property
    def development_evaluations_remaining(self) -> int:
        consumed = _count_budget_reservations(
            self.store.by_campaign(self.campaign_id),
            "development_evaluation_reserved",
            legacy_completed="development_evaluated",
        )
        return max(0, self._pinned_development_budget - consumed)

    @property
    def hidden_submissions_remaining(self) -> int:
        consumed = _count_budget_reservations(
            self.store.by_campaign(self.campaign_id),
            "hidden_submission_reserved",
            legacy_completed="hidden_submitted",
        )
        return max(0, self._pinned_hidden_budget - consumed)

    @property
    def hidden_used_ids(self) -> frozenset[str]:
        ids = {
            str(event.get("payload", {}).get("requested_lockbox_id", ""))
            for event in self.store.by_campaign(self.campaign_id)
            if event.get("event_type") in {
                "hidden_submission_reserved",
                "hidden_submitted",
            }
        }
        ids.update(
            str(event.get("payload", {}).get("result", {}).get("lockbox_id", ""))
            for event in self.store.by_campaign(self.campaign_id)
            if event.get("event_type") == "hidden_submitted"
        )
        ids.discard("")
        ids.discard("None")
        return frozenset(ids)

    def inspect_schema(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_fingerprint": self._campaign_fingerprint,
            "feature_contract": list(FEATURE_CONTRACT),
            "forbidden_features": sorted(FORBIDDEN_MODEL_FIELDS),
            "allowed_methods": [
                "inspect_schema",
                "list_variables",
                "inspect_permitted_data",
                "register_formula",
                "evaluate_development",
                "submit_hidden_once",
                "development_summary",
            ],
            "hidden_rows_reserved": len(self._hidden_rows),
            "hidden_case_set_hash": self._hidden_case_set_hash,
            "production_export_allowed": PRODUCTION_EXPORT_ALLOWED,
            "security_boundary": "typed API only; not a sandbox for malicious in-process code",
        }

    def list_variables(self) -> list[str]:
        names = {
            name
            for row in self._training_rows
            for name, value in enriched_features(row).items()
            if name in FEATURE_CONTRACT
            and name not in FORBIDDEN_MODEL_FIELDS
            and isinstance(value, (int, float, bool))
        }
        return sorted(names)

    def inspect_permitted_data(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if limit < 0 or limit > 200:
            raise ValueError("limit must be between 0 and 200")
        output = []
        for row in self._training_rows[:limit]:
            features = {
                name: enriched_features(row).get(name) for name in self.list_variables()
            }
            output.append(
                {"row_id": row.get("row_id"), "label": row.get("label"), "features": features}
            )
        return output

    def query_hidden_data(self) -> None:
        raise ResearchPermissionError(
            "hidden rows are not inspectable through OfferLabResearchAPI"
        )

    def execute_code(self, _code: str) -> None:
        raise ResearchPermissionError(
            "generated code execution is not available through OfferLabResearchAPI"
        )

    def change_outcome(self, *_args: Any, **_kwargs: Any) -> None:
        raise ResearchPermissionError("outcomes are immutable through OfferLabResearchAPI")

    def set_budget(self, *_args: Any, **_kwargs: Any) -> None:
        raise ResearchPermissionError("agents may not choose or modify research budgets")

    def register_formula(self, proposal: dict[str, Any]) -> dict[str, Any]:
        parsed = self._parse_proposal(proposal)
        existing = self.proposals.get(parsed.proposal_id)
        if existing is not None:
            if stable_hash(existing.to_dict()) != stable_hash(parsed.to_dict()):
                raise ValueError(
                    f"proposal_id {parsed.proposal_id!r} already exists with different content"
                )
            return existing.to_dict()
        self.proposals[parsed.proposal_id] = parsed
        self.store.append(
            "proposal_registered",
            {"campaign_id": self.campaign_id, "proposal": parsed.to_dict()},
        )
        return parsed.to_dict()

    def evaluate_development(self, proposal_id: str) -> dict[str, Any]:
        proposal = self._proposal(proposal_id)
        artifact = self._artifact_descriptor(proposal)
        evaluation_id = new_id("dev_eval")

        def guard(events: list[dict[str, Any]]) -> None:
            self._guard_campaign(events)
            consumed = _count_budget_reservations(
                _events_for_campaign(events, self.campaign_id),
                "development_evaluation_reserved",
                legacy_completed="development_evaluated",
            )
            if consumed >= self._pinned_development_budget:
                raise ResearchBudgetError("development evaluation budget exhausted")

        self.store.append_guarded(
            "development_evaluation_reserved",
            {
                "campaign_id": self.campaign_id,
                "evaluation_id": evaluation_id,
                "proposal_id": proposal_id,
                "artifact": artifact,
                "development_rows_hash": self._development_rows_hash,
                "development_case_set_hash": stable_hash(
                    _case_tokens(self._development_rows)
                ),
            },
            guard=guard,
        )
        result = self._evaluate(proposal, self._development_rows, split="development")
        result.update(
            {
                "evaluation_id": evaluation_id,
                "artifact_id": artifact["artifact_id"],
                "training_rows_hash": self._training_rows_hash,
                "development_rows_hash": self._development_rows_hash,
                "development_evaluations_remaining": self.development_evaluations_remaining,
            }
        )
        self.development_results[proposal_id] = result
        self.store.append(
            "development_evaluated",
            {"campaign_id": self.campaign_id, "result": result},
        )
        return copy.deepcopy(result)

    def submit_hidden_once(self, proposal_id: str, *, lockbox_id: str) -> dict[str, Any]:
        if not lockbox_id.strip():
            raise ValueError("lockbox_id is required")
        proposal = self._proposal(proposal_id)
        development_result = self.development_results.get(proposal_id)
        if development_result is None:
            raise ResearchPermissionError(
                "proposal must be evaluated on development before hidden submission"
            )
        artifact = self._artifact_descriptor(proposal)
        if development_result.get("artifact_id") != artifact["artifact_id"]:
            raise ResearchPermissionError(
                "hidden submission artifact does not match the development-evaluated artifact"
            )
        submission_id = new_id("hidden_eval")

        def guard(events: list[dict[str, Any]]) -> None:
            self._guard_campaign(events)
            campaign_events = _events_for_campaign(events, self.campaign_id)
            consumed = _count_budget_reservations(
                campaign_events,
                "hidden_submission_reserved",
                legacy_completed="hidden_submitted",
            )
            if consumed >= self._pinned_hidden_budget:
                raise ResearchBudgetError(
                    "hidden submission budget exhausted for this campaign/lockbox"
                )
            requested_ids = {
                str(event.get("payload", {}).get("requested_lockbox_id", ""))
                for event in campaign_events
                if event.get("event_type") == "hidden_submission_reserved"
            }
            if lockbox_id in requested_ids:
                raise ResearchBudgetError(
                    "hidden submission budget exhausted for this campaign/lockbox"
                )
            requested_tokens = set(self._hidden_case_tokens)
            for event in events:
                if event.get("event_type") not in {
                    "hidden_submission_reserved",
                    "offerlab_hidden_submission_reserved",
                    "hidden_submitted",
                }:
                    continue
                payload = event.get("payload", {})
                result = payload.get("result", {})
                if not isinstance(result, dict):
                    result = {}
                previous_case_set = (
                    payload.get("hidden_case_set_hash")
                    or payload.get("canonical_lockbox_id")
                    or result.get("hidden_case_set_hash")
                    or result.get("canonical_lockbox_id")
                )
                if previous_case_set in {
                    self._hidden_case_set_hash,
                    self._legacy_hidden_case_set_hash,
                }:
                    raise ResearchBudgetError(
                        "hidden case set was already reserved"
                    )
                previous_tokens = set(payload.get("hidden_case_tokens", []) or result.get("hidden_case_tokens", []))
                if requested_tokens & previous_tokens:
                    raise ResearchBudgetError(
                        "hidden case overlap detected with a previously reserved lockbox"
                    )

        # Reservation is deliberately written before evaluation. A crash after
        # this point consumes the query and cannot be retried under a new name.
        self.store.append_guarded(
            "hidden_submission_reserved",
            {
                "campaign_id": self.campaign_id,
                "submission_id": submission_id,
                "requested_lockbox_id": lockbox_id,
                "canonical_lockbox_id": self._hidden_case_set_hash,
                "proposal_id": proposal_id,
                "artifact": artifact,
                "development_evaluation_id": development_result.get("evaluation_id"),
                "hidden_rows_hash": self._hidden_rows_hash,
                "hidden_case_set_hash": self._hidden_case_set_hash,
                "hidden_case_tokens": list(self._hidden_case_tokens),
            },
            guard=guard,
        )
        result = self._evaluate(proposal, self._hidden_rows, split="hidden")
        result.update(
            {
                "hidden_submission_count": 1,
                "submission_id": submission_id,
                "lockbox_id": lockbox_id,
                "canonical_lockbox_id": self._hidden_case_set_hash,
                "artifact_id": artifact["artifact_id"],
                "hidden_case_set_hash": self._hidden_case_set_hash,
                "hidden_submissions_remaining": self.hidden_submissions_remaining,
            }
        )
        self.store.append(
            "hidden_submitted",
            {"campaign_id": self.campaign_id, "result": result},
        )
        return copy.deepcopy(result)

    def development_summary(self) -> dict[str, Any]:
        ordered = sorted(
            self.development_results.values(),
            key=lambda item: (
                item["log_loss"],
                item["complexity"],
                item["proposal_id"],
            ),
        )
        return {
            "campaign_id": self.campaign_id,
            "evaluated": len(ordered),
            "best": copy.deepcopy(ordered[0]) if ordered else None,
            "development_evaluations_remaining": self.development_evaluations_remaining,
        }

    def promote(self, proposal_id: str, reason: str) -> None:
        self.store.append(
            "proposal_promoted",
            {
                "campaign_id": self.campaign_id,
                "proposal_id": proposal_id,
                "reason": reason,
            },
        )

    def retire(self, proposal_id: str, reason: str) -> None:
        self.store.append(
            "proposal_retired",
            {
                "campaign_id": self.campaign_id,
                "proposal_id": proposal_id,
                "reason": reason,
            },
        )

    def fail(self, proposal_id: str, reason: str) -> None:
        self.store.append(
            "proposal_failed",
            {
                "campaign_id": self.campaign_id,
                "proposal_id": proposal_id,
                "reason": reason,
            },
        )

    def _restore_and_pin_configuration(self) -> None:
        campaign_events = self.store.by_campaign(self.campaign_id)
        created = [
            event
            for event in campaign_events
            if event.get("event_type") == "api_created"
        ]
        if created:
            pinned = created[0].get("payload", {})
            if pinned.get("campaign_fingerprint") != self._campaign_fingerprint:
                raise ResearchPermissionError(
                    "campaign metadata or case sets changed after the research lockbox was created"
                )
            self._pinned_development_budget = int(
                pinned.get("development_evaluations", self._pinned_development_budget)
            )
            self._pinned_hidden_budget = int(
                pinned.get("hidden_submissions", self._pinned_hidden_budget)
            )
        else:
            self.store.append_guarded(
                "api_created",
                {
                    "campaign_id": self.campaign_id,
                    "campaign_fingerprint": self._campaign_fingerprint,
                    "training_rows": len(self._training_rows),
                    "development_rows": len(self._development_rows),
                    "hidden_rows_reserved": len(self._hidden_rows),
                    "training_rows_hash": self._training_rows_hash,
                    "development_rows_hash": self._development_rows_hash,
                    "hidden_rows_hash": self._hidden_rows_hash,
                    "hidden_case_set_hash": self._hidden_case_set_hash,
                    "max_formula_terms": self.max_formula_terms,
                    "development_evaluations": self._pinned_development_budget,
                    "hidden_submissions": self._pinned_hidden_budget,
                    "production_export_allowed": PRODUCTION_EXPORT_ALLOWED,
                    "software_version": __version__,
                },
                guard=lambda events: self._guard_new_campaign(events),
            )

        for event in campaign_events:
            payload = event.get("payload", {})
            if event.get("event_type") == "proposal_registered":
                proposal_payload = payload.get("proposal", {})
                try:
                    proposal = FormulaProposal(**proposal_payload)
                except TypeError:
                    continue
                existing = self.proposals.get(proposal.proposal_id)
                if existing is not None and stable_hash(existing.to_dict()) != stable_hash(
                    proposal.to_dict()
                ):
                    raise ResearchStoreIntegrityError(
                        f"proposal_id {proposal.proposal_id!r} has conflicting definitions"
                    )
                self.proposals[proposal.proposal_id] = proposal
            elif event.get("event_type") == "development_evaluated":
                result = payload.get("result", {})
                proposal_id = str(result.get("proposal_id", ""))
                if proposal_id:
                    self.development_results[proposal_id] = copy.deepcopy(result)

        # A legacy completed development result may lack an artifact ID. It is
        # not eligible for hidden submission until reevaluated under v0.4.

    def _guard_new_campaign(self, events: list[dict[str, Any]]) -> None:
        for event in _events_for_campaign(events, self.campaign_id):
            if event.get("event_type") != "api_created":
                continue
            fingerprint = event.get("payload", {}).get("campaign_fingerprint")
            if fingerprint != self._campaign_fingerprint:
                raise ResearchPermissionError(
                    "campaign metadata or case sets changed after lockbox creation"
                )
            raise ResearchPermissionError(
                "campaign lockbox was concurrently created; reopen the existing campaign"
            )

    def _guard_campaign(self, events: list[dict[str, Any]]) -> None:
        created = [
            event
            for event in _events_for_campaign(events, self.campaign_id)
            if event.get("event_type") == "api_created"
        ]
        if not created:
            raise ResearchStoreIntegrityError("campaign lockbox metadata is missing")
        if created[0].get("payload", {}).get("campaign_fingerprint") != self._campaign_fingerprint:
            raise ResearchPermissionError(
                "campaign metadata or case sets changed after lockbox creation"
            )

    def _artifact_descriptor(self, proposal: FormulaProposal) -> dict[str, Any]:
        proposal_hash = stable_hash(proposal.to_dict())
        artifact_id = stable_hash(
            {
                "proposal_hash": proposal_hash,
                "training_rows_hash": self._training_rows_hash,
                "feature_contract": list(FEATURE_CONTRACT),
                "software_version": __version__,
            }
        )
        return {
            "artifact_id": artifact_id,
            "proposal_hash": proposal_hash,
            "proposal_id": proposal.proposal_id,
            "training_rows_hash": self._training_rows_hash,
            "feature_contract_hash": stable_hash(FEATURE_CONTRACT),
            "software_version": __version__,
        }

    def _parse_proposal(self, proposal: dict[str, Any]) -> FormulaProposal:
        if not isinstance(proposal, dict):
            raise ValueError("proposal must be an object")
        forbidden_keys = {
            "code",
            "python",
            "execute",
            "hidden_rows",
            "hidden_data",
            "budget",
            "set_budget",
            "change_outcome",
        }
        overlap = sorted(forbidden_keys & set(proposal))
        if overlap:
            raise ResearchPermissionError(
                f"proposal contains forbidden keys: {overlap}"
            )
        if _has_causal_claim(proposal):
            raise ResearchPermissionError("autonomous proposals may not claim causality")
        raw_terms = proposal.get("terms", [])
        if not isinstance(raw_terms, list):
            raise ValueError("proposal terms must be a list")
        terms = [str(term).strip() for term in raw_terms if str(term).strip()]
        if not terms:
            raise ValueError("proposal requires at least one formula term")
        if len(terms) > self.max_formula_terms:
            raise ResearchBudgetError("proposal exceeds max_formula_terms")
        formula = Formula.parse(terms)
        unknown = formula.variables - set(self.list_variables())
        if unknown:
            raise ValueError(f"proposal uses unavailable variables: {sorted(unknown)}")
        proposal_id = str(proposal.get("proposal_id") or new_id("offerlab_h"))
        falsification = str(
            proposal.get("falsification")
            or proposal.get("falsification_condition")
            or ""
        ).strip()
        if not falsification:
            raise ValueError("proposal requires a falsification condition")
        return FormulaProposal(
            proposal_id=proposal_id,
            terms=terms,
            target_label=str(proposal.get("target_label", "accept")),
            falsification=falsification,
            model_family=str(proposal.get("model_family", "logistic_formula")),
            source=str(proposal.get("source", "agent")),
        )

    def _proposal(self, proposal_id: str) -> FormulaProposal:
        try:
            return self.proposals[proposal_id]
        except KeyError as exc:
            raise KeyError(f"unknown proposal_id {proposal_id!r}") from exc

    def _evaluate(
        self,
        proposal: FormulaProposal,
        rows: list[dict[str, Any]],
        *,
        split: str,
    ) -> dict[str, Any]:
        formula = Formula.parse(proposal.terms)
        train_scores = [
            _score_formula(formula, row) for row in self._training_rows
        ]
        positive = [
            score
            for score, row in zip(train_scores, self._training_rows, strict=True)
            if str(row["label"]) == proposal.target_label
        ]
        negative = [
            score
            for score, row in zip(train_scores, self._training_rows, strict=True)
            if str(row["label"]) != proposal.target_label
        ]
        threshold = (_mean(positive) + _mean(negative)) / 2.0
        base_rate = (
            (len(positive) + 0.5) / (len(self._training_rows) + 1.0)
            if self._training_rows
            else 0.5
        )
        total = 0.0
        predictions = []
        for row in rows:
            score = _score_formula(formula, row)
            probability = min(
                1.0 - 1e-6,
                max(1e-6, base_rate + 0.25 * math.tanh(score - threshold)),
            )
            observed = (
                1.0 if str(row["label"]) == proposal.target_label else 0.0
            )
            total -= observed * math.log(probability) + (
                1.0 - observed
            ) * math.log(1.0 - probability)
            predictions.append(
                {"row_id": row.get("row_id"), "probability": probability}
            )
        redacted_predictions = (
            [
                {"index": index, "redacted": True}
                for index, _item in enumerate(predictions)
            ]
            if split == "hidden"
            else [{"row_id": item["row_id"]} for item in predictions]
        )
        return {
            "campaign_id": self.campaign_id,
            "proposal_id": proposal.proposal_id,
            "split": split,
            "rows": len(rows),
            "target_label": proposal.target_label,
            "log_loss": total / len(rows) if rows else 0.0,
            "complexity": formula.complexity,
            "terms": list(proposal.terms),
            "predictions_redacted": redacted_predictions,
            "production_export_allowed": PRODUCTION_EXPORT_ALLOWED,
        }


def _events_for_campaign(
    events: list[dict[str, Any]], campaign_id: str
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("payload", {}).get("campaign_id") == campaign_id
    ]


def _count_budget_reservations(
    events: list[dict[str, Any]],
    reservation_type: str,
    *,
    legacy_completed: str,
) -> int:
    reservations = [event for event in events if event.get("event_type") == reservation_type]
    if reservations:
        return len(reservations)
    # Compatibility for pre-v0.4 stores that recorded only completed queries.
    return sum(1 for event in events if event.get("event_type") == legacy_completed)


def _public_training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row.get("row_id"),
        "label": row.get("label"),
        "features": copy.deepcopy(row.get("features", {})),
    }


def _content_case_token(row: dict[str, Any]) -> str:
    # Case identity intentionally ignores mutable display/participant IDs and
    # the hidden label. Renaming metadata or relabeling the same feature/time
    # case therefore cannot manufacture a fresh lockbox.
    identity = {
        "task": row.get("task"),
        "timestamp": row.get("timestamp"),
        "features": row.get("features", {}),
        "observed_history": row.get("observed_history", []),
    }
    return stable_hash(identity)


def _source_case_token(row: dict[str, Any]) -> str | None:
    source_identity = {
        "task": row.get("task"),
        "row_id": row.get("row_id"),
        "thread_id": row.get("thread_id"),
        "listing_id": row.get("listing_id"),
        "source_row_id": row.get("source_row_id"),
    }
    if not any(value not in {None, ""} for value in source_identity.values()):
        return None
    return stable_hash(source_identity)


def _case_token(row: dict[str, Any]) -> str:
    return _content_case_token(row)


def _case_tokens(rows: list[dict[str, Any]]) -> list[str]:
    tokens: set[str] = set()
    for row in rows:
        tokens.add(_content_case_token(row))
        source_token = _source_case_token(row)
        if source_token is not None:
            tokens.add(source_token)
    return sorted(tokens)


def _rows_hash(rows: list[dict[str, Any]]) -> str:
    return stable_hash([to_jsonable(row) for row in rows])


def _score_formula(formula: Formula, row: dict[str, Any]) -> float:
    features = enriched_features(row)
    values = formula.vector(features)
    return sum(values[1:]) if len(values) > 1 else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _has_causal_claim(proposal: dict[str, Any]) -> bool:
    causal_flag = proposal.get("causal_claim")
    if isinstance(causal_flag, bool) and causal_flag:
        return True
    if isinstance(causal_flag, str):
        return True
    text_fields = [
        "claim",
        "falsification",
        "falsification_condition",
        "model_family",
        "source",
    ]
    return any(
        "causal" in str(proposal.get(field, "")).lower() for field in text_fields
    )
