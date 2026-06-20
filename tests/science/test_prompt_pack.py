from __future__ import annotations

from pathlib import Path


PROMPT_DIR = Path(__file__).parents[2] / "resonance" / "science" / "prompts"


def test_proposer_prompt_states_scientist_constraints() -> None:
    prompt = _prompt("proposer_v1.md")

    assert "at most 8 hypotheses" in prompt
    assert "only `HypothesisSpec`" in prompt
    assert "Do not return prose, markdown, comments, code" in prompt
    assert "Use at most 3 input metrics" in prompt
    assert "rationale" in prompt
    assert "falsification_conditions" in prompt
    assert "negative_controls" in prompt
    assert "minimum_blind_effect" in prompt
    assert "minimum_baseline_improvement" in prompt
    assert "structurally different" in prompt
    assert "Do not use causal language" in prompt
    assert "seasonality and autocorrelation risks" in prompt


def test_reviewer_prompt_states_skeptic_contract() -> None:
    prompt = _prompt("reviewer_v1.md")

    assert "One `DiscoveryBrief`" in prompt
    assert "Exactly one `HypothesisSpec`" in prompt
    assert "must not receive tuning data, blind data" in prompt
    assert "exactly one `ReviewSpec`" in prompt
    assert "confounders" in prompt
    assert "simpler_explanation" in prompt
    assert "leakage_risk" in prompt
    assert "mechanical_correlation_risk" in prompt
    assert "suggested_controls_or_falsifications" in prompt
    assert "executable" in prompt
    assert "distinct_from_prior" in prompt
    assert "`reject`, `revise`, or `preregistration-eligible`" in prompt
    assert "may criticize or reject" in prompt
    assert "Do not assign statistical significance" in prompt
    assert "Do not predict blind-set success" in prompt


def _prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")
