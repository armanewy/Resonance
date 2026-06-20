from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import Field, ValidationError, model_validator

from resonance.science.contracts import StrictModel, canonical_json, stable_hash
from resonance.science.experiments.contracts import ExperimentSpec
from resonance.science.ledger import DEFAULT_LEDGER_PATH, append_event, current_code_commit
from resonance.science.snapshots import DEFAULT_ARTIFACT_ROOT
from resonance.time_utils import ensure_utc, parse_utc, to_utc_iso, utc_now


EXPERIMENT_PLANNER_VERSION = "llm-experiment-planner-v1"
EXPERIMENT_PLANNING_ARTIFACT_SCHEMA_VERSION = 1


class ExperimentPlannerError(ValueError):
    """Raised when experiment planning cannot be represented safely."""


class PlannerRecommendation(str, Enum):
    REJECT = "reject"
    REVISE = "revise"
    APPROVAL_ELIGIBLE = "approval-eligible"


class BlindEvaluatedHypothesis(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    hypothesis_hash: str = Field(min_length=1)
    title: str = Field(min_length=1)
    observational_claim: str = Field(min_length=1)
    blind_status: str = Field(min_length=1)
    blind_metrics: dict[str, Any] = Field(default_factory=dict)
    blind_warnings: tuple[str, ...] = Field(default_factory=tuple)


class PlannerBrief(StrictModel):
    blind_evaluated_hypothesis: BlindEvaluatedHypothesis
    permitted_personal_metrics: tuple[str, ...] = Field(min_length=1)
    allowed_reversible_intervention_categories: tuple[str, ...] = Field(min_length=1)
    prior_experiment_memory_summaries: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_brief(self) -> PlannerBrief:
        if len(set(self.permitted_personal_metrics)) != len(self.permitted_personal_metrics):
            raise ValueError("permitted personal metrics must be unique")
        if len(set(self.allowed_reversible_intervention_categories)) != len(
            self.allowed_reversible_intervention_categories
        ):
            raise ValueError("allowed reversible intervention categories must be unique")
        return self

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", exclude_none=True))

    def artifact_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json", exclude_none=True))


PlanningContext = PlannerBrief


class PlannerReview(StrictModel):
    distinguishes_competing_explanations: bool
    outcome_measurable: bool
    schedule_feasible: bool
    time_of_day_confounding_addressed: bool
    randomization_and_washout_reasonable: bool
    low_risk: bool
    simpler_test_exists: bool
    rejection_reasons: tuple[str, ...] = Field(default_factory=tuple)
    recommendation: PlannerRecommendation
    human_approval_required: Literal[True] = True
    runner_start_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_recommendation(self) -> PlannerReview:
        failed_checks = (
            not self.distinguishes_competing_explanations
            or not self.outcome_measurable
            or not self.schedule_feasible
            or not self.time_of_day_confounding_addressed
            or not self.randomization_and_washout_reasonable
            or not self.low_risk
        )
        if self.recommendation == PlannerRecommendation.REJECT and not self.rejection_reasons:
            raise ValueError("rejected reviews must include rejection reasons")
        if failed_checks and self.recommendation == PlannerRecommendation.APPROVAL_ELIGIBLE:
            raise ValueError("approval-eligible review cannot have failed required checks")
        return self


class PlanningIssue(StrictModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class DeterministicPlanningDecision(StrictModel):
    accepted: bool
    issues: tuple[PlanningIssue, ...] = Field(default_factory=tuple)
    human_approval_required: Literal[True] = True
    runner_start_allowed: Literal[False] = False


class RejectedProtocol(StrictModel):
    proposal_hash: str
    error: str


class PlannerRunMetadata(StrictModel):
    provider_name: str
    model: str
    request_config: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str
    reviewer_name: str | None = None
    reviewer_model: str | None = None
    reviewer_prompt_version: str | None = None
    seed: int


class PlannerRun(StrictModel):
    metadata: PlannerRunMetadata
    brief_sha256: str
    raw_proposal_sha256: str
    spec: ExperimentSpec | None = None
    review: PlannerReview | None = None
    deterministic_decision: DeterministicPlanningDecision
    rejected_protocols: tuple[RejectedProtocol, ...] = Field(default_factory=tuple)
    status: Literal["rejected", "revise", "approval-eligible"]
    human_approval_required: Literal[True] = True
    runner_start_allowed: Literal[False] = False

    def artifact_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_json(self) -> str:
        return canonical_json(self.artifact_payload())

    def artifact_hash(self) -> str:
        return stable_hash(self.artifact_payload())


class ExperimentPlannerProvider(Protocol):
    name: str
    model: str
    prompt_version: str
    request_config: dict[str, Any]

    def propose(self, brief: PlannerBrief, seed: int) -> Any:
        """Return one raw ExperimentSpec JSON-compatible proposal."""


class ExperimentReviewerProvider(Protocol):
    name: str
    model: str
    prompt_version: str
    request_config: dict[str, Any]

    def review(self, brief: PlannerBrief, spec: ExperimentSpec, seed: int) -> Any:
        """Return one raw PlannerReview JSON-compatible review."""


class MockExperimentPlanner:
    def __init__(
        self,
        proposal: Any,
        *,
        name: str = "mock-experiment-planner",
        model: str = "mock-model",
        prompt_version: str = "mock-experiment-planner-v1",
        request_config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self.request_config = dict(request_config or {})
        self._proposal = proposal
        self.last_raw_proposal: Any = None

    def propose(self, brief: PlannerBrief, seed: int) -> Any:
        self.last_raw_proposal = self._proposal
        return self._proposal


class FileExperimentPlanner:
    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "file-experiment-planner",
        model: str = "file",
        prompt_version: str = "file-experiment-planner-v1",
        request_config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self.request_config = {"path": str(self.path), **dict(request_config or {})}
        self.last_raw_proposal: Any = None

    def propose(self, brief: PlannerBrief, seed: int) -> Any:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        proposal = payload.get("proposal", payload) if isinstance(payload, dict) else payload
        self.last_raw_proposal = proposal
        return proposal


class MockExperimentReviewer:
    def __init__(
        self,
        review: Any,
        *,
        name: str = "mock-experiment-reviewer",
        model: str = "mock-model",
        prompt_version: str = "mock-experiment-reviewer-v1",
        request_config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self.request_config = dict(request_config or {})
        self._review = review
        self.last_raw_review: Any = None

    def review(self, brief: PlannerBrief, spec: ExperimentSpec, seed: int) -> Any:
        self.last_raw_review = self._review
        return self._review


class FileExperimentReviewer:
    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "file-experiment-reviewer",
        model: str = "file",
        prompt_version: str = "file-experiment-reviewer-v1",
        request_config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self.request_config = {"path": str(self.path), **dict(request_config or {})}
        self.last_raw_review: Any = None

    def review(self, brief: PlannerBrief, spec: ExperimentSpec, seed: int) -> Any:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw_review = payload.get("review", payload) if isinstance(payload, dict) else payload
        self.last_raw_review = raw_review
        return raw_review


def plan_experiment(
    planner: ExperimentPlannerProvider,
    brief: PlannerBrief,
    *,
    seed: int,
    reviewer: ExperimentReviewerProvider | None = None,
) -> PlannerRun:
    raw_proposal = planner.propose(brief, seed)
    raw_proposal = getattr(planner, "last_raw_proposal", raw_proposal)
    proposal_hash = stable_hash(_artifact_value(raw_proposal))
    metadata = PlannerRunMetadata(
        provider_name=planner.name,
        model=planner.model,
        request_config=dict(planner.request_config),
        prompt_version=planner.prompt_version,
        reviewer_name=getattr(reviewer, "name", None),
        reviewer_model=getattr(reviewer, "model", None),
        reviewer_prompt_version=getattr(reviewer, "prompt_version", None),
        seed=seed,
    )

    try:
        spec = ExperimentSpec.model_validate(raw_proposal)
    except ValidationError as exc:
        rejected = RejectedProtocol(proposal_hash=proposal_hash, error=str(exc))
        return PlannerRun(
            metadata=metadata,
            brief_sha256=brief.artifact_hash(),
            raw_proposal_sha256=proposal_hash,
            deterministic_decision=DeterministicPlanningDecision(
                accepted=False,
                issues=(PlanningIssue(code="invalid_experiment_spec", message="proposal failed ExperimentSpec validation"),),
            ),
            rejected_protocols=(rejected,),
            status="rejected",
        )

    review = _review_or_default(reviewer, brief, spec, seed)
    decision = validate_planned_experiment(spec, brief, review=review)
    status = _planning_status(decision, review)
    rejected_protocols = ()
    if status == "rejected":
        rejected_protocols = (
            RejectedProtocol(
                proposal_hash=proposal_hash,
                error="; ".join(issue.message for issue in decision.issues) or "; ".join(review.rejection_reasons),
            ),
        )
    return PlannerRun(
        metadata=metadata,
        brief_sha256=brief.artifact_hash(),
        raw_proposal_sha256=proposal_hash,
        spec=spec,
        review=review,
        deterministic_decision=decision,
        rejected_protocols=rejected_protocols,
        status=status,
    )


def validate_planned_experiment(
    spec: ExperimentSpec,
    brief: PlannerBrief,
    *,
    review: PlannerReview | None = None,
) -> DeterministicPlanningDecision:
    permitted_metrics = set(brief.permitted_personal_metrics)
    used_metrics = {spec.primary_outcome_metric, *spec.secondary_outcome_metrics}
    allowed_categories = set(brief.allowed_reversible_intervention_categories)
    issues: list[PlanningIssue] = []

    unsupported_metrics = used_metrics - permitted_metrics
    if unsupported_metrics:
        issues.append(
            PlanningIssue(
                code="unsupported_outcome_metric",
                message=f"outcome metrics are not permitted: {', '.join(sorted(unsupported_metrics))}",
            )
        )
    if spec.intervention_condition.name not in allowed_categories:
        issues.append(
            PlanningIssue(
                code="unsupported_intervention_category",
                message="intervention condition must name an allowed reversible intervention category",
            )
        )
    if spec.requires_manual_confirmation is not True:
        issues.append(
            PlanningIssue(
                code="manual_confirmation_required",
                message="human approval and manual confirmation are required",
            )
        )
    if review is not None and review.recommendation == PlannerRecommendation.REJECT:
        issues.append(
            PlanningIssue(
                code="review_rejected",
                message="skeptical reviewer recommended rejection",
            )
        )
    return DeterministicPlanningDecision(accepted=not issues, issues=tuple(_unique_issues(issues)))


def record_planning_memory(
    run: PlannerRun,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    now_utc: datetime | str | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root)
    artifact = store_planning_artifact(root, run.artifact_payload())
    spec_hash = run.spec.experiment_hash() if run.spec is not None else None
    payload = {
        "planner_version": EXPERIMENT_PLANNER_VERSION,
        "planning_status": run.status,
        "experiment_hash": spec_hash,
        "title": run.spec.title if run.spec is not None else None,
        "hypothesis_id": (
            run.spec.hypothesis_id
            if run.spec is not None
            else None
        ),
        "human_approval_required": True,
        "manual_only": True,
        "automatic_intervention_applied": False,
        "runner_started": False,
        "review_recommendation": run.review.recommendation.value if run.review is not None else None,
        "deterministic_accepted": run.deterministic_decision.accepted,
        "artifact_root": str(root.resolve()),
        "artifacts": {"planning_run": artifact},
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    return append_event(
        "experiment_planned",
        payload,
        artifact_hashes={"planning_run": artifact["sha256"]},
        code_commit=current_code_commit(),
        ledger_path=ledger_path,
        timestamp_utc=_timestamp(now_utc),
    )


def store_planning_artifact(root: str | Path, payload: Mapping[str, Any]) -> dict[str, str]:
    artifact_payload = {
        "schema_version": EXPERIMENT_PLANNING_ARTIFACT_SCHEMA_VERSION,
        "record_type": "experiment_planning_run",
        "planner_version": EXPERIMENT_PLANNER_VERSION,
        "payload": _artifact_value(payload),
    }
    content = json.dumps(artifact_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = stable_hash(json.loads(content.decode("utf-8")))
    relative = f"sha256/{digest[:2]}/{digest}.json"
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ExperimentPlannerError(f"artifact hash collision at {path}")
    else:
        path.write_bytes(content)
    return {"sha256": digest, "path": relative, "format": "json"}


def _review_or_default(
    reviewer: ExperimentReviewerProvider | None,
    brief: PlannerBrief,
    spec: ExperimentSpec,
    seed: int,
) -> PlannerReview:
    if reviewer is None:
        return PlannerReview(
            distinguishes_competing_explanations=False,
            outcome_measurable=spec.primary_outcome_metric in brief.permitted_personal_metrics,
            schedule_feasible=True,
            time_of_day_confounding_addressed=False,
            randomization_and_washout_reasonable=True,
            low_risk=True,
            simpler_test_exists=True,
            rejection_reasons=("skeptical reviewer was not provided",),
            recommendation=PlannerRecommendation.REJECT,
        )
    raw_review = reviewer.review(brief, spec, seed)
    raw_review = getattr(reviewer, "last_raw_review", raw_review)
    try:
        return PlannerReview.model_validate(raw_review)
    except ValidationError as exc:
        raise ExperimentPlannerError(f"reviewer returned invalid PlannerReview: {exc}") from exc


def _planning_status(
    decision: DeterministicPlanningDecision,
    review: PlannerReview,
) -> Literal["rejected", "revise", "approval-eligible"]:
    if not decision.accepted or review.recommendation == PlannerRecommendation.REJECT:
        return "rejected"
    if review.recommendation == PlannerRecommendation.REVISE:
        return "revise"
    return "approval-eligible"


def _unique_issues(issues: Sequence[PlanningIssue]) -> list[PlanningIssue]:
    seen: set[str] = set()
    unique: list[PlanningIssue] = []
    for issue in issues:
        key = f"{issue.code}:{issue.message}"
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


def _artifact_value(value: Any) -> Any:
    if isinstance(value, StrictModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {
            str(key): _artifact_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_artifact_value(item) for item in value]
    return value


def _timestamp(value: datetime | str | None) -> str:
    if value is None:
        return to_utc_iso(utc_now())
    if isinstance(value, str):
        return to_utc_iso(parse_utc(value))
    return to_utc_iso(ensure_utc(value))


__all__ = [
    "EXPERIMENT_PLANNER_VERSION",
    "BlindEvaluatedHypothesis",
    "DeterministicPlanningDecision",
    "ExperimentPlannerError",
    "ExperimentPlannerProvider",
    "ExperimentReviewerProvider",
    "FileExperimentPlanner",
    "FileExperimentReviewer",
    "MockExperimentPlanner",
    "MockExperimentReviewer",
    "PlannerBrief",
    "PlannerRecommendation",
    "PlannerReview",
    "PlannerRun",
    "PlannerRunMetadata",
    "PlanningContext",
    "PlanningIssue",
    "RejectedProtocol",
    "plan_experiment",
    "record_planning_memory",
    "store_planning_artifact",
    "validate_planned_experiment",
]
