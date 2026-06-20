from __future__ import annotations

from pathlib import Path
from pprint import pprint

from behavior_lab.core import HypothesisSpec
from behavior_lab.gym import WorldGym
from behavior_lab.research_api import ResearchAPI
from behavior_lab.worlds import make_world


run_dir = Path("runs/first-session")

gym = WorldGym(
    run_dir,
    world=make_world("habit", seed=101),
)

if not gym.decision_episodes():
    gym.seed(240)

api = ResearchAPI(gym, campaign_id="first-session")

print("\nSCHEMA")
pprint(api.inspect_schema())

print("\nTARGET")
pprint(api.describe_target())

print("\nAVAILABLE VARIABLES")
pprint(api.list_variables())

print("\nBASELINE MODEL ZOO")
zoo = api.fit_model_zoo()
for model in zoo:
    print(model.model_id, type(model).__name__, model.complexity)

hypothesis = HypothesisSpec.formula(
    hypothesis_id="h_first_manual",
    target_name=gym.target_name,
    terms=[
        "deadline_near",
        "public_commitment",
        "fatigue",
        "recent_context_switches",
        "explicit_first_step * indicator(ambiguity > 0.6)",
    ],
    falsification_conditions=[
        "Does not improve development log loss over the base-rate model",
        "Fails after the candidate is frozen",
    ],
)

api.submit_hypothesis(hypothesis)
fit = api.fit_hypothesis(hypothesis.hypothesis_id)

print("\nFITTED PARAMETERS")
pprint(fit)

print("\nDEVELOPMENT EVALUATION")
pprint(api.evaluate_hypothesis(fit["model_id"], split="development"))

print("\nWORST DEVELOPMENT ERRORS")
pprint(api.inspect_residuals(fit["model_id"], limit=5))

print("\nPROPOSED DISCRIMINATING EXPERIMENT")
proposal = api.propose_experiment(
    [
        fit["model_id"],
        zoo[0].model_id,
        zoo[-1].model_id,
    ]
)
pprint(proposal)

print("\nOFFLINE EXPERIMENT INGESTION")
pprint(api.run_offline_experiment(proposal, trials=8))
