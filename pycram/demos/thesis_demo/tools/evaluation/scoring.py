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
)
from thesis_demo.validation.schema import (
    DIRECTIONAL_RELATIONS,
    PLAN_STEP_FIELDS,
    render,
)

OUTCOMES = {"plan", "clarification"}

MAIN_METRICS = (
    "planning_success",
    "reachability_success",
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


def _validate_clarification(case_id, answer, targets, follow_up_plan):
    if not isinstance(targets, list) or not targets:
        raise ValueError(
            f"{case_id}: expected_clarification_targets must be "
            "a non-empty string list"
        )
    if not all(isinstance(target, str) and target.strip() for target in targets):
        raise ValueError(
            f"{case_id}: expected_clarification_targets must be "
            "a non-empty string list"
        )

    if answer is None and follow_up_plan is None:
        return
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{case_id}: invalid clarification_answer")
    if not _valid_expected_plan(follow_up_plan, "plan"):
        raise ValueError(f"{case_id}: invalid expected_follow_up_plan")


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

        if outcome != "clarification":
            if any(value is not None for value in (answer, targets, follow_up_plan)):
                raise ValueError(
                    f"{strings['id']}: clarification fields require "
                    "expected_outcome='clarification'"
                )
        else:
            _validate_clarification(strings["id"], answer, targets, follow_up_plan)

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
    reachability_success: bool | None = None
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
            "reachability_success": self.reachability_success,
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
        "reachability_success": result.reachability_success,
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


def plan_quality_metrics(case, outcome, payload, planner_metadata=None):
    planner_metadata = planner_metadata or {}
    return {
        "json_object_valid": bool(planner_metadata.get("json_valid")),
        "schema_valid": bool(planner_metadata.get("schema_valid")),
        "names_valid": planner_metadata.get("names_valid"),
        "sequence_valid": planner_metadata.get("sequence_valid"),
        "outcome_match": outcome == case.expected_outcome,
        "planned_goal_match": reference_match(case, payload),
    }


class FailedCheck(StrEnum):
    JSON_OBJECT = "json_object"
    SCHEMA = "schema"
    NAMES = "names"
    SEQUENCE = "sequence"
    OUTCOME = "outcome"
    CLARIFICATION_TARGET = "clarification_target"
    PLANNED_GOAL = "planned_goal"
    REACHABILITY = "reachability"
    EXECUTION = "execution"
    WORLD_GOAL = "world_goal"


def first_failed_check(metrics, reachability_success, execution_success):
    gates = (
        (FailedCheck.JSON_OBJECT, metrics.get("json_object_valid")),
        (FailedCheck.SCHEMA, metrics.get("schema_valid")),
        (FailedCheck.NAMES, metrics.get("names_valid")),
        (FailedCheck.SEQUENCE, metrics.get("sequence_valid")),
        (FailedCheck.OUTCOME, metrics.get("outcome_match")),
        (FailedCheck.CLARIFICATION_TARGET, metrics.get("clarification_target_match")),
        (FailedCheck.PLANNED_GOAL, metrics.get("planned_goal_match")),
        (FailedCheck.REACHABILITY, reachability_success),
        (FailedCheck.EXECUTION, execution_success),
        (FailedCheck.WORLD_GOAL, metrics.get("world_goal_reached")),
    )
    for gate, passed in gates:
        if passed is False:
            return gate.value
    return None


@dataclass
class GoalChecks:
    expected: int = 0
    checked: int = 0
    errors: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)

    def record(self, passed, description):
        self.expected += 1
        self.checked += 1
        if not passed:
            self.errors.append(f"not satisfied: {description}")

    def skip(self, description, reason):
        self.expected += 1
        self.unavailable.append(f"{description}: {reason}")

    def merged_with(self, other):
        return GoalChecks(
            expected=self.expected + other.expected,
            checked=self.checked + other.checked,
            errors=self.errors + other.errors,
            unavailable=self.unavailable + other.unavailable,
        )

    @property
    def success(self):
        if not self.expected:
            return None
        return self.checked == self.expected and not self.errors


def final_location_goals(case):
    goals_by_object = {}
    for goal in case.goals():
        object_description = goal.get("object")
        if object_description and goal.get("location"):
            goals_by_object[render(object_description)] = goal
    return list(goals_by_object.values())


def _final_action_per_object(case):
    actions = {}
    for goal in case.goals():
        if goal.get("action") and goal.get("object"):
            actions[render(goal["object"])] = goal["action"]
    return actions


def _containers_left_open_on_purpose(case):
    container_states = {}
    for goal in case.goals():
        if goal.get("action") == "OpenAction":
            container_states[render(goal.get("object"))] = True
        elif goal.get("action") == "CloseAction":
            container_states[render(goal.get("object"))] = False
    return {
        name
        for name, should_remain_open in container_states.items()
        if should_remain_open
    }


def location_checks(case, final_context):
    checks = GoalChecks()
    object_locations = final_context.get("object_locations")
    for goal in final_location_goals(case):
        if goal.get("relation") in DIRECTIONAL_RELATIONS:
            continue

        object_description = render(goal["object"])
        object_type = goal["object"]["type"]
        expected_locations = goal["location"]
        expected_locations = (
            expected_locations
            if isinstance(expected_locations, list)
            else [expected_locations]
        )
        expected_location_types = [
            location["type"] for location in expected_locations
        ]
        description = (
            f"location: {object_description} is at one of "
            f"{expected_location_types!r}"
        )

        if (
            not isinstance(object_locations, dict)
            or not object_locations.get(object_type)
        ):
            checks.skip(description, "final location is not in the world context")
            continue
        checks.record(
            bool(
                set(object_locations[object_type]).intersection(
                    expected_location_types
                )
            ),
            f"{description}, but it is at {object_locations[object_type]!r}",
        )
    return checks


def physical_checks(case, observations):
    checks = GoalChecks()
    observations = observations if isinstance(observations, dict) else {}
    _check_final_navigation(checks, case, observations)
    _check_containers(checks, case, observations)
    _check_gripper(checks, case, observations)
    _check_directions(checks, case, observations)
    return checks


def _check_final_navigation(checks, case, observations):
    goals = case.goals()
    if not goals or goals[-1].get("action") != "NavigateAction":
        return
    location = render(goals[-1].get("location"))
    description = f"navigation: arrived at {location}"

    navigation = observations.get("navigation")
    if not isinstance(navigation, list) or not navigation:
        checks.skip(description, "no navigation was observed")
        return
    arrival = navigation[-1]
    checks.record(
        arrival.get("location") == location and arrival.get("success") is True,
        description,
    )


def _check_containers(checks, case, observations):
    container_states = observations.get("container_states")
    if not isinstance(container_states, dict):
        return
    containers_left_open = _containers_left_open_on_purpose(case)

    for container_name in sorted(container_states):
        expected_state = (
            "open" if container_name in containers_left_open else "closed"
        )
        description = f"container: {container_name!r} is {expected_state}"
        actual_state = container_states[container_name]
        if actual_state is None or actual_state == "unknown":
            checks.skip(description, "the container state could not be read")
            continue
        checks.record(actual_state == expected_state, description)


def _check_gripper(checks, case, observations):
    held_objects = observations.get("held_objects")
    if not isinstance(held_objects, list):
        return
    final_actions = _final_action_per_object(case)

    for object_name, action in sorted(final_actions.items()):
        if action == "PickUpAction":
            checks.record(
                object_name in held_objects, f"gripper: {object_name} is held"
            )

    for goal in final_location_goals(case):
        object_name = render(goal["object"])
        if final_actions.get(object_name) == "PickUpAction":
            continue
        checks.record(
            object_name not in held_objects,
            f"gripper: {object_name} was released",
        )


def _check_directions(checks, case, observations):
    directional_results = observations.get("directional_relations")
    for goal in final_location_goals(case):
        if goal.get("relation") not in DIRECTIONAL_RELATIONS:
            continue
        object_description = render(goal["object"])
        location_description = render(goal["location"])
        description = (
            f"direction: {object_description} is "
            f"{goal['relation']} {location_description}"
        )
        if not isinstance(directional_results, list):
            checks.skip(description, "no directional relation was observed")
            continue

        observed_success = None
        for result in reversed(directional_results):
            if (
                result.get("object") == object_description
                and result.get("relation") == goal["relation"]
                and result.get("location") == location_description
            ):
                observed_success = result.get("success")
                break
        if not isinstance(observed_success, bool):
            checks.skip(description, "no directional relation was observed")
            continue
        checks.record(observed_success, description)


def empty_metrics():
    return {
        "json_object_valid": None,
        "schema_valid": None,
        "names_valid": None,
        "sequence_valid": None,
        "outcome_match": None,
        "clarification_target_match": None,
        "planned_goal_match": None,
        "attempts": None,
        "first_failed_check": None,
        **_world_metrics(GoalChecks()),
    }


def _world_metrics(checks):
    return {
        "world_goal_reached": checks.success,
        "world_checks_expected": checks.expected,
        "world_checks_evaluated": checks.checked,
        "world_failures": checks.errors,
        "world_unchecked": checks.unavailable,
    }


def verified_goal_metrics(case, final_context, observations):
    observations = observations if isinstance(observations, dict) else {}
    checks = location_checks(case, final_context)
    checks = checks.merged_with(physical_checks(case, observations))
    return _world_metrics(checks)


def task_succeeded(
    case,
    planning_success,
    execution_status,
    world_goal_reached,
    planned_goal_match=None,
    clarification_target_match=None,
):
    if not planning_success:
        return False
    if case.expected_outcome == "clarification":
        if clarification_target_match is not True:
            return False
        if not case.clarification_answer:
            return execution_status == "not_required"
    if execution_status != "ok":
        return False
    if world_goal_reached is not None:
        return world_goal_reached
    return planned_goal_match is True


CASES_FILE = Path(__file__).with_name("kitchen_eval_samples.jsonl")


def validate_entities(case, context, label):
    steps = case.goals()
    if not steps:
        return
    names = context_names(context)
    for step in steps:
        for field_name, allowed_types in allowed_names_for_step(step, names).items():
            described = step.get(field_name)
            if described is None:
                continue
            entity_types = [
                item["type"]
                for item in (described if isinstance(described, list) else [described])
            ]
            missing = [name for name in entity_types if name not in allowed_types]
            if missing:
                raise ValueError(
                    f"{case.id}: {label} has no {field_name} {missing[0]!r}"
                )

        described_object = step.get("object")
        described_source = step.get("source")
        if described_object is None or described_source is None:
            continue
        object_type = described_object["type"]
        source_types = {
            item["type"]
            for item in (
                described_source
                if isinstance(described_source, list)
                else [described_source]
            )
        }
        places = context.get("object_locations", {}).get(object_type, [])
        if object_type in names.objects and not source_types.intersection(places):
            raise ValueError(
                f"{case.id}: {label} source of {object_type!r} is "
                f"{places!r}, not {sorted(source_types)!r}"
            )


def covered_objects(cases):
    return {
        goal["object"]["type"]
        for case in cases
        for goal in case.goals()
        if goal.get("object")
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
