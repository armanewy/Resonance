from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from resonance.science.experiments import (
    FileExperimentPlanner,
    MockExperimentPlanner,
    MockExperimentReviewer,
    PlannerBrief,
    generate_randomized_schedule,
    plan_experiment,
    record_planning_memory,
)
from resonance.science.ledger import read_entries, verify_ledger


def test_planner_brief_contains_only_allowed_planning_inputs() -> None:
    brief = _brief()
    serialized = brief.canonical_json()

    assert "blind_evaluated_hypothesis" in serialized
    assert "permitted_personal_metrics" in serialized
    assert "allowed_reversible_intervention_categories" in serialized
    assert "prior_experiment_memory_summaries" in serialized
    assert "BLIND_OBSERVATION_SENTINEL" not in serialized


def test_mock_planner_accepts_valid_spec_as_approval_eligible_without_starting_runner() -> None:
    run = plan_experiment(
        MockExperimentPlanner(_valid_experiment()),
        _brief(),
        seed=17,
        reviewer=MockExperimentReviewer(_approval_eligible_review()),
    )

    assert run.status == "approval-eligible"
    assert run.spec is not None
    assert run.spec.requires_manual_confirmation is True
    assert run.human_approval_required is True
    assert run.runner_start_allowed is False
    assert run.deterministic_decision.accepted is True
    assert run.rejected_protocols == ()


def test_invalid_experiment_spec_is_rejected_without_repair() -> None:
    invalid = _valid_experiment()
    invalid["randomized_schedule"] = list(reversed(invalid["randomized_schedule"]))

    run = plan_experiment(
        MockExperimentPlanner(invalid),
        _brief(),
        seed=17,
        reviewer=MockExperimentReviewer(_approval_eligible_review()),
    )

    assert run.status == "rejected"
    assert run.spec is None
    assert run.review is None
    assert run.deterministic_decision.accepted is False
    assert run.deterministic_decision.issues[0].code == "invalid_experiment_spec"
    assert run.rejected_protocols[0].proposal_hash == run.raw_proposal_sha256


def test_skeptical_reviewer_can_reject_valid_protocol() -> None:
    run = plan_experiment(
        MockExperimentPlanner(_valid_experiment()),
        _brief(),
        seed=17,
        reviewer=MockExperimentReviewer(
            {
                **_approval_eligible_review(),
                "distinguishes_competing_explanations": False,
                "recommendation": "reject",
                "rejection_reasons": ["The intervention does not distinguish time pressure from desk setup."],
            }
        ),
    )

    assert run.status == "rejected"
    assert run.spec is not None
    assert run.review is not None
    assert run.review.human_approval_required is True
    assert run.review.runner_start_allowed is False
    assert any(issue.code == "review_rejected" for issue in run.deterministic_decision.issues)


def test_deterministic_validator_rejects_unsupported_metrics_and_interventions() -> None:
    proposal = _valid_experiment()
    proposal["primary_outcome_metric"] = "unapproved_metric"
    proposal["intervention_condition"]["name"] = "unsupported category"

    run = plan_experiment(
        MockExperimentPlanner(proposal),
        _brief(),
        seed=17,
        reviewer=MockExperimentReviewer(_approval_eligible_review()),
    )

    assert run.status == "rejected"
    assert {issue.code for issue in run.deterministic_decision.issues} == {
        "unsupported_outcome_metric",
        "unsupported_intervention_category",
    }


def test_file_planner_loads_experiment_spec_json(tmp_path: Path) -> None:
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps({"proposal": _valid_experiment()}), encoding="utf-8")

    run = plan_experiment(
        FileExperimentPlanner(proposal_path),
        _brief(),
        seed=17,
        reviewer=MockExperimentReviewer(_approval_eligible_review()),
    )

    assert run.status == "approval-eligible"
    assert run.metadata.provider_name == "file-experiment-planner"
    assert run.metadata.seed == 17


def test_rejected_protocol_is_recordable_as_scientific_memory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    ledger_path = tmp_path / "ledger.jsonl"
    invalid = _valid_experiment()
    invalid["control_condition"]["changes_router_or_os_settings_automatically"] = True
    run = plan_experiment(
        MockExperimentPlanner(invalid),
        _brief(),
        seed=17,
        reviewer=MockExperimentReviewer(_approval_eligible_review()),
    )

    entry = record_planning_memory(
        run,
        artifact_root=artifact_root,
        ledger_path=ledger_path,
        now_utc=datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc),
    )

    entries = read_entries(ledger_path)
    assert entry["event_type"] == "experiment_planned"
    assert entries[0]["payload"]["planning_status"] == "rejected"
    assert entries[0]["payload"]["human_approval_required"] is True
    assert entries[0]["payload"]["automatic_intervention_applied"] is False
    assert entries[0]["payload"]["runner_started"] is False
    assert (
        entries[0]["artifact_hashes"]["planning_run"]
        == entries[0]["payload"]["artifacts"]["planning_run"]["sha256"]
    )
    assert verify_ledger(ledger_path).valid is True
    assert (artifact_root / entries[0]["payload"]["artifacts"]["planning_run"]["path"]).exists()


def test_experiment_planner_prompts_state_contracts() -> None:
    prompt_dir = Path(__file__).parents[3] / "resonance" / "science" / "prompts"
    planner_prompt = (prompt_dir / "experiment_planner_v1.md").read_text(encoding="utf-8")
    reviewer_prompt = (prompt_dir / "experiment_reviewer_v1.md").read_text(encoding="utf-8")

    assert "only one `ExperimentSpec`" in planner_prompt
    assert "strict list of allowed reversible intervention categories" in planner_prompt
    assert "Do not return prose, markdown, comments, code" in planner_prompt
    assert "Do not call a runner" in planner_prompt
    assert "device-control instructions" in planner_prompt
    assert "exactly one `PlannerReview`" in reviewer_prompt
    assert "distinguishes_competing_explanations" in reviewer_prompt
    assert "time_of_day_confounding_addressed" in reviewer_prompt
    assert "`reject`, `revise`, or `approval-eligible`" in reviewer_prompt
    assert "runner_start_allowed` to false" in reviewer_prompt


def _brief() -> PlannerBrief:
    return PlannerBrief(
        blind_evaluated_hypothesis={
            "hypothesis_id": "hypothesis-123",
            "hypothesis_hash": "h" * 64,
            "title": "Quiet desk predicts focus",
            "observational_claim": "Quiet desk periods are associated with higher focus scores.",
            "blind_status": "pass",
            "blind_metrics": {"spearman_rho": 0.24, "observation_count": 30},
            "blind_warnings": (),
        },
        permitted_personal_metrics=("focus_score", "self_reported_distraction"),
        allowed_reversible_intervention_categories=("quiet desk", "usual desk"),
        prior_experiment_memory_summaries=("No prior controlled desk setup test.",),
    )


def _approval_eligible_review() -> dict:
    return {
        "distinguishes_competing_explanations": True,
        "outcome_measurable": True,
        "schedule_feasible": True,
        "time_of_day_confounding_addressed": True,
        "randomization_and_washout_reasonable": True,
        "low_risk": True,
        "simpler_test_exists": False,
        "rejection_reasons": [],
        "recommendation": "approval-eligible",
        "human_approval_required": True,
        "runner_start_allowed": False,
    }


def _valid_experiment() -> dict:
    planned_start = datetime(2026, 1, 3, 9, 0, tzinfo=timezone.utc)
    schedule = generate_randomized_schedule(
        planned_start=planned_start,
        block_duration_seconds=1800,
        number_of_blocks=4,
        washout_duration_seconds=300,
        randomization_seed=8675309,
    )
    return {
        "schema_version": "1.0",
        "title": "Manual quiet desk focus comparison",
        "hypothesis_id": "hypothesis-123",
        "intervention_condition": {
            "name": "quiet desk",
            "instructions": "Use the quiet desk setup selected before the experiment.",
            "execution_mode": "human_executed",
            "is_medical_intervention": False,
            "involves_hazardous_physical_action": False,
            "changes_router_or_os_settings_automatically": False,
            "prevents_emergency_communication": False,
        },
        "control_condition": {
            "name": "usual desk",
            "instructions": "Use the usual desk setup selected before the experiment.",
            "execution_mode": "human_executed",
            "is_medical_intervention": False,
            "involves_hazardous_physical_action": False,
            "changes_router_or_os_settings_automatically": False,
            "prevents_emergency_communication": False,
        },
        "primary_outcome_metric": "focus_score",
        "secondary_outcome_metrics": ["self_reported_distraction"],
        "block_duration_seconds": 1800,
        "number_of_blocks": 4,
        "washout_duration_seconds": 300,
        "randomization_seed": 8675309,
        "randomized_schedule": [block.model_dump(mode="json") for block in schedule],
        "planned_start": planned_start.isoformat(),
        "inclusion_rules": [{"description": "Only start blocks during preselected work hours."}],
        "exclusion_rules": [{"description": "Exclude blocks interrupted by unavoidable external events."}],
        "stopping_rules": [{"description": "Stop after all confirmed blocks are complete."}],
        "abort_conditions": [{"description": "Abort immediately if safety or communication access is uncertain."}],
        "minimum_effect": 0.5,
        "analysis_method": "paired_block_difference",
        "safety_notes": "Low-risk, reversible desk setup only; no medical, hazardous, or automated setting changes.",
        "requires_manual_confirmation": True,
        "prohibited_automatic_actions": [
            "medical_interventions",
            "hazardous_physical_actions",
            "automatic_router_or_os_setting_changes",
            "blocking_emergency_communication",
        ],
        "maximum_experiment_duration_seconds": 86400,
    }
