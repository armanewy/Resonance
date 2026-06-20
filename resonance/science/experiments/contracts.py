from __future__ import annotations

import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationInfo, model_validator

from resonance.science.contracts import StrictModel, canonical_json, stable_hash


EXPERIMENT_SCHEMA_VERSION = "1.0"
DEFAULT_MAX_EXPERIMENT_DURATION_SECONDS = 7 * 24 * 60 * 60
MAX_CONFIGURABLE_EXPERIMENT_DURATION_SECONDS = 30 * 24 * 60 * 60
MIN_BLOCK_DURATION_SECONDS = 60
MIN_WASHOUT_DURATION_SECONDS = 60


class ConditionName(str, Enum):
    INTERVENTION = "intervention"
    CONTROL = "control"


class AnalysisMethod(str, Enum):
    MEAN_DIFFERENCE = "mean_difference"
    PAIRED_BLOCK_DIFFERENCE = "paired_block_difference"
    NONPARAMETRIC_SIGN_TEST = "nonparametric_sign_test"


class ManualCondition(StrictModel):
    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    execution_mode: Literal["human_executed"] = "human_executed"
    is_medical_intervention: Literal[False] = False
    involves_hazardous_physical_action: Literal[False] = False
    changes_router_or_os_settings_automatically: Literal[False] = False
    prevents_emergency_communication: Literal[False] = False


class ExperimentRule(StrictModel):
    description: str = Field(min_length=1)


class ScheduledBlock(StrictModel):
    block_index: Annotated[int, Field(ge=0)]
    condition: ConditionName
    planned_start: datetime
    planned_end: datetime
    requires_user_confirmation: Literal[True] = True

    @model_validator(mode="after")
    def validate_interval(self) -> ScheduledBlock:
        if self.planned_end <= self.planned_start:
            raise ValueError("scheduled block end must be after start")
        return self


class ExperimentSpec(StrictModel):
    schema_version: Literal[EXPERIMENT_SCHEMA_VERSION] = EXPERIMENT_SCHEMA_VERSION
    title: str = Field(min_length=1)
    hypothesis_id: Annotated[str, Field(min_length=1)]
    intervention_condition: ManualCondition
    control_condition: ManualCondition
    primary_outcome_metric: Annotated[str, Field(min_length=1)]
    secondary_outcome_metrics: tuple[Annotated[str, Field(min_length=1)], ...] = Field(default_factory=tuple)
    block_duration_seconds: Annotated[int, Field(ge=MIN_BLOCK_DURATION_SECONDS)]
    number_of_blocks: Annotated[int, Field(ge=2)]
    washout_duration_seconds: Annotated[int, Field(ge=MIN_WASHOUT_DURATION_SECONDS)]
    randomization_seed: int
    randomized_schedule: tuple[ScheduledBlock, ...]
    planned_start: datetime
    inclusion_rules: tuple[ExperimentRule, ...] = Field(default_factory=tuple)
    exclusion_rules: tuple[ExperimentRule, ...] = Field(default_factory=tuple)
    stopping_rules: Annotated[tuple[ExperimentRule, ...], Field(min_length=1)]
    abort_conditions: Annotated[tuple[ExperimentRule, ...], Field(min_length=1)]
    minimum_effect: Annotated[float, Field(gt=0.0)]
    analysis_method: AnalysisMethod
    safety_notes: Annotated[str, Field(min_length=1)]
    requires_manual_confirmation: Literal[True] = True
    prohibited_automatic_actions: Annotated[tuple[Annotated[str, Field(min_length=1)], ...], Field(min_length=1)]
    maximum_experiment_duration_seconds: Annotated[
        int,
        Field(gt=0, le=MAX_CONFIGURABLE_EXPERIMENT_DURATION_SECONDS),
    ] = DEFAULT_MAX_EXPERIMENT_DURATION_SECONDS

    @model_validator(mode="after")
    def validate_experiment_contract(self, info: ValidationInfo) -> ExperimentSpec:
        max_duration = configured_max_duration(
            info.context,
            self.maximum_experiment_duration_seconds,
        )
        if self.total_duration_seconds() > max_duration:
            raise ValueError("experiment duration exceeds configured maximum")
        if self.number_of_blocks % 2 != 0:
            raise ValueError("number_of_blocks must be even to balance conditions")
        if self.primary_outcome_metric in self.secondary_outcome_metrics:
            raise ValueError("primary outcome metric cannot also be secondary")
        if len(set(self.secondary_outcome_metrics)) != len(self.secondary_outcome_metrics):
            raise ValueError("secondary outcome metrics must be unique")
        if self.intervention_condition == self.control_condition:
            raise ValueError("intervention and control conditions must differ")
        self._validate_schedule()
        self._validate_prohibited_actions()
        return self

    def total_duration_seconds(self) -> int:
        return (
            self.number_of_blocks * self.block_duration_seconds
            + (self.number_of_blocks - 1) * self.washout_duration_seconds
        )

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", exclude_none=True))

    def frozen_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def experiment_hash(self) -> str:
        return stable_hash(self.frozen_content())

    def _validate_schedule(self) -> None:
        if len(self.randomized_schedule) != self.number_of_blocks:
            raise ValueError("randomized_schedule length must match number_of_blocks")
        expected = generate_randomized_schedule(
            planned_start=self.planned_start,
            block_duration_seconds=self.block_duration_seconds,
            number_of_blocks=self.number_of_blocks,
            washout_duration_seconds=self.washout_duration_seconds,
            randomization_seed=self.randomization_seed,
        )
        if self.randomized_schedule != expected:
            raise ValueError("randomized_schedule must match the frozen deterministic seed schedule")
        counts = {
            ConditionName.INTERVENTION: 0,
            ConditionName.CONTROL: 0,
        }
        for block in self.randomized_schedule:
            counts[block.condition] += 1
        if counts[ConditionName.INTERVENTION] != counts[ConditionName.CONTROL]:
            raise ValueError("randomized_schedule must balance intervention and control blocks")

    def _validate_prohibited_actions(self) -> None:
        required = {
            "medical_interventions",
            "hazardous_physical_actions",
            "automatic_router_or_os_setting_changes",
            "blocking_emergency_communication",
        }
        missing = required - set(self.prohibited_automatic_actions)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"prohibited automatic actions missing: {names}")


def generate_randomized_schedule(
    *,
    planned_start: datetime,
    block_duration_seconds: int,
    number_of_blocks: int,
    washout_duration_seconds: int,
    randomization_seed: int,
) -> tuple[ScheduledBlock, ...]:
    if block_duration_seconds < MIN_BLOCK_DURATION_SECONDS:
        raise ValueError("block duration is shorter than the minimum")
    if washout_duration_seconds < MIN_WASHOUT_DURATION_SECONDS:
        raise ValueError("washout duration is shorter than the minimum")
    if number_of_blocks < 2 or number_of_blocks % 2 != 0:
        raise ValueError("number_of_blocks must be an even value of at least 2")

    conditions = [ConditionName.INTERVENTION, ConditionName.CONTROL] * (number_of_blocks // 2)
    random.Random(randomization_seed).shuffle(conditions)
    block_delta = timedelta(seconds=block_duration_seconds)
    washout_delta = timedelta(seconds=washout_duration_seconds)
    blocks: list[ScheduledBlock] = []
    cursor = planned_start
    for index, condition in enumerate(conditions):
        planned_end = cursor + block_delta
        blocks.append(
            ScheduledBlock(
                block_index=index,
                condition=condition,
                planned_start=cursor,
                planned_end=planned_end,
            )
        )
        cursor = planned_end + washout_delta
    return tuple(blocks)


def configured_max_duration(context: Any, spec_max_duration_seconds: int) -> int:
    configured = (context or {}).get("max_experiment_duration_seconds")
    if configured is None:
        return spec_max_duration_seconds
    configured_int = int(configured)
    if configured_int <= 0:
        raise ValueError("configured maximum experiment duration must be positive")
    if configured_int > MAX_CONFIGURABLE_EXPERIMENT_DURATION_SECONDS:
        raise ValueError("configured maximum experiment duration exceeds hard bound")
    return min(spec_max_duration_seconds, configured_int)


__all__ = [
    "DEFAULT_MAX_EXPERIMENT_DURATION_SECONDS",
    "EXPERIMENT_SCHEMA_VERSION",
    "MAX_CONFIGURABLE_EXPERIMENT_DURATION_SECONDS",
    "MIN_BLOCK_DURATION_SECONDS",
    "MIN_WASHOUT_DURATION_SECONDS",
    "AnalysisMethod",
    "ConditionName",
    "ExperimentRule",
    "ExperimentSpec",
    "ManualCondition",
    "ScheduledBlock",
    "generate_randomized_schedule",
]
