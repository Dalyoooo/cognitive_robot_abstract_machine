from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from thesis_demo.validation.guard import (
    allowed_locations_for,
    allowed_objects_for,
    context_names,
    verify,
)
from thesis_demo.validation.schema import (
    PLAN_STEP_FIELDS,
    parse_clarification,
    parse_plan,
)

OUTCOMES = {"plan", "clarification"}

MAIN_METRICS = (
    "planning_success",
    "grounding_success",
    "execution_success",
    "task_success",
)


class EvaluationMode(StrEnum):
    PLANNING = "planning"
    END_TO_END = "end_to_end"


class ModelVariant(StrEnum):
    BASE = "base"
    FINETUNED = "finetuned"


def _valid_expected_plan(plan, outcome):
    if outcome != "plan" or not plan:
        return False
    steps = plan if isinstance(plan, list) else [plan]
    return all(
        isinstance(step, dict) and step and not (set(step) - PLAN_STEP_FIELDS)
        for step in steps
    )


@dataclass(frozen=True)
class LiveCase:
    id: str
    instruction: str
    robot: str = "hsrb"
    environment: str = "kitchen"
    expected_outcome: str = "plan"
    expected_plan: dict[str, Any] | list[dict[str, Any]] | None = None
    clarification_answer: str | None = None
    expected_clarification_targets: list[str] = field(default_factory=list)
    expected_follow_up_plan: dict[str, Any] | list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, row):
        strings = {
            "id": row.get("id"),
            "instruction": row.get("instruction"),
            "robot": row.get("robot", "hsrb"),
            "environment": row.get("environment", "kitchen"),
        }
        if not all(
            isinstance(value, str) and value.strip() for value in strings.values()
        ):
            raise ValueError("id, instruction, robot, and environment are required")

        outcome = row.get("expected_outcome", "plan")
        plan = row.get("expected_plan")
        answer = row.get("clarification_answer")
        targets = row.get("expected_clarification_targets")
        follow_up_plan = row.get("expected_follow_up_plan")
        if outcome not in OUTCOMES:
            raise ValueError(f"{strings['id']}: invalid expected_outcome")
        if plan is not None and not _valid_expected_plan(plan, outcome):
            raise ValueError(f"{strings['id']}: invalid expected_plan")

        clarification_fields = (answer, targets, follow_up_plan)
        if outcome == "clarification":
            if (
                not isinstance(targets, list)
                or not targets
                or not all(
                    isinstance(target, str) and target.strip() for target in targets
                )
            ):
                raise ValueError(
                    f"{strings['id']}: expected_clarification_targets must be "
                    "a non-empty string list"
                )
            scripted_dialog = answer is not None or follow_up_plan is not None
            if scripted_dialog:
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"{strings['id']}: invalid clarification_answer")
                if not _valid_expected_plan(follow_up_plan, "plan"):
                    raise ValueError(
                        f"{strings['id']}: invalid expected_follow_up_plan"
                    )
        elif any(value is not None for value in clarification_fields):
            raise ValueError(
                f"{strings['id']}: clarification fields require "
                "expected_outcome='clarification'"
            )

        return cls(
            **strings,
            expected_outcome=outcome,
            expected_plan=plan,
            clarification_answer=answer,
            expected_clarification_targets=targets or [],
            expected_follow_up_plan=follow_up_plan,
        )

    def goals(self):
        plan = self.expected_follow_up_plan or self.expected_plan
        if not plan:
            return []
        if isinstance(plan, list):
            return plan
        return [plan]

    def follow_up_case(self):
        if not self.clarification_answer or not self.expected_follow_up_plan:
            return None
        return LiveCase(
            id=self.id,
            instruction=self.clarification_answer,
            robot=self.robot,
            environment=self.environment,
            expected_outcome="plan",
            expected_plan=self.expected_follow_up_plan,
        )


@dataclass(frozen=True)
class EvaluationConfiguration:
    mode: EvaluationMode = EvaluationMode.END_TO_END
    visualization_delay_s: float = 0.0
    world_settle_delay_s: float = 0.0
    case_timeout_s: float | None = 300.0

    def __post_init__(self):
        if self.world_settle_delay_s < 0:
            raise ValueError("world_settle_delay_s must not be negative")
        if self.case_timeout_s is not None and self.case_timeout_s <= 0:
            raise ValueError("case_timeout_s must be positive or None")


@dataclass
class LiveResult:
    case: LiveCase
    model_variant: ModelVariant | None = None
    evaluation_mode: EvaluationMode = EvaluationMode.END_TO_END
    planner_outcome: str = "error"
    planning_success: bool | None = None
    grounding_success: bool | None = None
    execution_success: bool | None = None
    execution_status: str = "not_attempted"
    timed_out: bool = False
    task_success: bool | None = None
    failure_stage: str | None = None
    error: str | None = None
    planning_latency_s: float = 0.0
    execution_latency_s: float = 0.0
    total_latency_s: float = 0.0
    plan: dict[str, Any] | None = None
    raw_responses: list[str] = field(default_factory=list)
    inference_parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_record(self):
        return {
            "id": self.case.id,
            "instruction": self.case.instruction,
            "expected_outcome": self.case.expected_outcome,
            "expected_plan": self.case.expected_plan,
            "clarification_answer": self.case.clarification_answer,
            "expected_clarification_targets": self.case.expected_clarification_targets,
            "expected_follow_up_plan": self.case.expected_follow_up_plan,
            "model_variant": (
                self.model_variant.value if self.model_variant is not None else None
            ),
            "evaluation_mode": self.evaluation_mode.value,
            "planner_outcome": self.planner_outcome,
            "planning_success": self.planning_success,
            "grounding_success": self.grounding_success,
            "execution_success": self.execution_success,
            "execution_status": self.execution_status,
            "timed_out": self.timed_out,
            "task_success": self.task_success,
            "failure_stage": self.failure_stage,
            "error": self.error,
            "planning_latency_s": self.planning_latency_s,
            "execution_latency_s": self.execution_latency_s,
            "total_latency_s": self.total_latency_s,
            "plan": self.plan,
            "raw_responses": self.raw_responses,
            "inference": self.inference_parameters,
            "diagnostics": self.metrics,
        }


def _metric_value(result, name):
    values = {
        "planning_success": result.planning_success,
        "grounding_success": result.grounding_success,
        "execution_success": result.execution_success,
        "task_success": result.task_success,
    }
    return values[name]


def metric_counts(results, name):
    values = [_metric_value(result, name) for result in results]
    values = [value for value in values if value is not None]
    return sum(bool(value) for value in values), len(values)


@dataclass(frozen=True)
class MetricAggregate:
    successes: int
    observed_cases: int
    total_cases: int

    @classmethod
    def from_results(cls, results, name):
        relevant_results = results
        if name != "planning_success":
            relevant_results = [
                result
                for result in results
                if result.evaluation_mode == EvaluationMode.END_TO_END
            ]
        successes, observed_cases = metric_counts(relevant_results, name)
        return cls(successes, observed_cases, len(relevant_results))

    @property
    def stage_percent(self):
        if not self.observed_cases:
            return None
        return round(100.0 * self.successes / self.observed_cases, 1)

    @property
    def benchmark_percent(self):
        if not self.total_cases:
            return None
        return round(100.0 * self.successes / self.total_cases, 1)

    def to_record(self):
        return {
            "successes": self.successes,
            "observed_cases": self.observed_cases,
            "total_cases": self.total_cases,
            "stage_percent": self.stage_percent,
            "benchmark_percent": self.benchmark_percent,
        }


@dataclass(frozen=True)
class LiveSummary:
    cases: int
    metric_rates: dict[str, Any]

    @classmethod
    def from_results(cls, results):
        metric_rates = {}
        for name in MAIN_METRICS:
            metric_rates[name] = MetricAggregate.from_results(results, name).to_record()

        return cls(
            cases=len(results),
            metric_rates=metric_rates,
        )


def _step_matches(step, reference):
    return isinstance(step, dict) and all(
        field_matches(step.get(key), value) for key, value in reference.items()
    )


def field_matches(actual, expected):
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _reference_satisfied(steps, expected):
    if isinstance(expected, list):
        next_step_index = 0
        for reference in expected:
            matching_index = None
            for step_index in range(next_step_index, len(steps)):
                if _step_matches(steps[step_index], reference):
                    matching_index = step_index
                    break
            if matching_index is None:
                return False
            next_step_index = matching_index + 1
        return True
    if "action" in expected:
        return any(_step_matches(step, expected) for step in steps)
    goal = {
        key: expected[key]
        for key in ("object", "location", "relation")
        if key in expected
    }
    destination = any(_step_matches(step, goal) for step in steps)
    source = "source" not in expected or any(
        isinstance(step, dict)
        and step.get("object") == expected.get("object")
        and field_matches(step.get("source"), expected["source"])
        for step in steps
    )
    return destination and source


def outcome_error(case, outcome):
    if outcome != case.expected_outcome:
        return f"expected outcome {case.expected_outcome!r}, got {outcome!r}"
    return None


def reference_match(case, payload):
    if case.expected_outcome != "plan" or not case.expected_plan:
        return None
    steps = payload.get("plan", []) if isinstance(payload, dict) else []
    return _reference_satisfied(steps, case.expected_plan)


def allowed_names_for_step(step, names):
    action = step.get("action")
    relation = step.get("relation")
    return {
        "object": allowed_objects_for(action, names),
        "location": allowed_locations_for(action, relation, names),
        "source": names.sources,
    }


def plan_quality_metrics(case, outcome, payload, context, planner_metadata=None):
    planner_metadata = planner_metadata or {}
    normalized = {"clarification": payload} if outcome == "clarification" else payload
    parsed_payload = (outcome == "plan" and isinstance(payload, dict)) or (
        outcome == "clarification" and isinstance(payload, str)
    )
    json_valid = bool(planner_metadata.get("json_valid", parsed_payload))
    schema_valid = False
    guard_valid = False
    if outcome == "error":
        schema_valid = bool(planner_metadata.get("schema_valid", False))
        guard_valid = bool(planner_metadata.get("guard_valid", False))
    elif parsed_payload:
        try:
            if outcome == "clarification":
                parse_clarification(normalized)
            else:
                parse_plan(normalized)
            schema_valid = True
        except ValueError:
            schema_valid = False
        guard_valid = schema_valid and verify(normalized, context)[0]

    return {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "guard_valid": guard_valid,
        "plan_valid": bool(schema_valid and guard_valid),
        "outcome_match": outcome == case.expected_outcome,
        "reference_goal_match": reference_match(case, payload),
    }


CASES_FILE = Path(__file__).with_name("kitchen_eval_samples.jsonl")


def validate_entities(case, context, label):
    steps = case.goals()
    if not steps:
        return
    names = context_names(context)
    for step in steps:
        for field_name, allowed_names in allowed_names_for_step(step, names).items():
            value = step.get(field_name)
            values = value if isinstance(value, list) else [value]
            missing = [name for name in values if name and name not in allowed_names]
            if missing:
                raise ValueError(
                    f"{case.id}: {label} has no {field_name} {missing[0]!r}"
                )

        object_name = step.get("object")
        source = step.get("source")
        locations = context.get("object_locations", {}).get(object_name, [])
        sources = source if isinstance(source, list) else [source]
        if (
            object_name in names.objects
            and source
            and not set(sources).intersection(locations)
        ):
            raise ValueError(
                f"{case.id}: {label} source of {object_name!r} is "
                f"{locations!r}, not {source!r}"
            )


def covered_objects(cases):
    return {
        goal["object"] for case in cases for goal in case.goals() if goal.get("object")
    }


def live_entity_names(context):
    names = set()
    for key in ("objects", "surfaces", "containers", "openables", "furniture", "rooms"):
        names.update(context.get(key, []))
    names.update(context.get("object_locations", {}))
    return names


def validate_kitchen_inventory(context, cases, label):
    missing = sorted(covered_objects(cases) - live_entity_names(context))
    if missing:
        raise ValueError(f"{label} is missing benchmark objects: {', '.join(missing)}")


def load_cases(path=CASES_FILE):
    cases = [
        LiveCase.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case.id for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("case file requires non-empty unique ids")
    if any(len(case_id) != 3 or not case_id.isdecimal() for case_id in ids):
        raise ValueError("case ids must contain exactly three decimal digits")

    for case in cases:
        if case.environment != "kitchen":
            raise ValueError(f"{case.id}: only the kitchen environment is supported")
    return cases
