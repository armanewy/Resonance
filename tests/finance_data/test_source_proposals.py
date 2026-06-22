from __future__ import annotations

import _bootstrap  # noqa: F401
import pytest

from behavior_lab.finance_data import FinanceDataError, SourceProposal


def _proposal(**overrides: object) -> SourceProposal:
    payload = {
        "proposal_id": "proposal_macro_calendar",
        "title": "Macro release calendar research proposal",
        "researcher_prompt": "Evaluate whether this source contract can support point-in-time macro releases.",
        "proposed_provider": "Example Macro Vendor",
        "observation_types": ["economic_release", "revision_record", "vintage_snapshot"],
        "coverage": {"geography": "US", "start": "2000-01-01", "frequency": "release"},
        "access_method": "human-reviewed contract and offline artifact import",
        "license_summary": "Unknown until counsel reviews the provider terms.",
        "permitted_uses": {"research": True, "production_export": False},
        "prohibited_uses": ["scraping", "automated activation", "broker/order integration"],
        "artifact_hash_plan": "hash every delivered source file before parsing",
        "time_semantics": {
            "event_time": "release or fixing timestamp",
            "available_at": "vendor publication timestamp",
            "ingested_at": "local immutable import timestamp",
        },
        "risks": ["license uncertainty", "revision restatement ambiguity"],
        "open_questions": ["Does the vendor expose original vintage files?"],
        "created_at": "2026-06-22T12:00:00+00:00",
    }
    payload.update(overrides)
    return SourceProposal(**payload)


def test_source_proposal_is_llm_research_schema_not_an_active_source() -> None:
    proposal = _proposal()
    payload = proposal.to_dict()

    assert proposal.is_active_source is False
    assert proposal.requires_human_source_contract is True
    assert payload["is_active_source"] is False
    assert payload["scraping_allowed"] is False
    assert "economic_release" in payload["observation_types"]
    with pytest.raises(FinanceDataError, match="cannot activate"):
        proposal.activation_contract()


def test_source_proposal_rejects_automatic_activation_and_unknown_observation_types() -> None:
    with pytest.raises(FinanceDataError, match="cannot request activation"):
        _proposal(activation_requested=True)
    with pytest.raises(FinanceDataError, match="approved source contracts"):
        _proposal(approved_source_contract_id="approved-contract")
    with pytest.raises(FinanceDataError, match="unknown observation_types"):
        _proposal(observation_types=["broker_order_api"])
