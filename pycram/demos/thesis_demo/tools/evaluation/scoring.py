import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from ...validation.guard import (
    allowed_locations_for,
    allowed_objects_for,
    context_names,
    verify,
)
from ...validation.schema import parse_clarification, parse_plan

OUTCOMES = {"plan", "clarification"}
PLAN_FIELDS = {"action", "object", "location", "relation", "source"}

MAIN_METRICS = (
    "json_valid",
    "schema_valid",
    "guard_valid",
    "planner_success",
    "grounding_success",
    "execution_success",
    "postcondition_success",
    "physical_goal_success",
    "goal_success",
    "task_success",
)


def _valid_expected_plan(plan, outcome):
    if outcome != "plan" or not plan:
        return False
    steps = plan if isinstance(plan, list) else [plan]
    return all(
        isinstance(step, dict) and step and not (set(step) - PLAN_FIELDS)
        for step in steps
    )


def json_cell(value):
    return "" if value is None else json.dumps(value, separators=(",", ":"))


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
        """Create a validated case from one JSONL row."""
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
class EvaluationConfig:
    world_timeout_s: float = 60.0
    execution_timeout_s: float = 180.0
    poll_interval_s: float = 0.2
    visualization_delay_s: float = 0.0


@dataclass
class LiveResult:
    case: LiveCase
    planner_outcome: str = "error"
    planner_success: bool = False
    execution_status: str = "not_attempted"
    task_success: bool = False
    failure_stage: str | None = None
    error: str | None = None
    planning_latency_s: float = 0.0
    execution_latency_s: float = 0.0
    total_latency_s: float = 0.0
    plan: dict[str, Any] | None = None
    raw_responses: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_row(self):
        row = {
            "id": self.case.id,
            "instruction": self.case.instruction,
            "expected_outcome": self.case.expected_outcome,
            "expected_plan": json_cell(self.case.expected_plan),
            "clarification_answer": self.case.clarification_answer or "",
            "expected_clarification_targets": json_cell(
                self.case.expected_clarification_targets
            ),
            "expected_follow_up_plan": json_cell(self.case.expected_follow_up_plan),
            "planner_outcome": self.planner_outcome,
            "planner_success": self.planner_success,
            "execution_status": self.execution_status,
            "task_success": self.task_success,
            "failure_stage": self.failure_stage or "",
            "error": self.error or "",
            "planning_latency_s": self.planning_latency_s,
            "execution_latency_s": self.execution_latency_s,
            "total_latency_s": self.total_latency_s,
            "plan": json_cell(self.plan),
            "raw_responses": json_cell(self.raw_responses),
        }
        for name, value in self.metrics.items():
            row[name] = json_cell(value) if isinstance(value, (dict, list)) else value
        return row


def _metric_value(result, name):
    if name in result.metrics:
        return result.metrics[name]
    return getattr(result, name, None)


def metric_counts(results, name):
    values = [_metric_value(result, name) for result in results]
    values = [value for value in values if value is not None]
    return sum(bool(value) for value in values), len(values)


def confusion_rates(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return precision, recall, f1


def clarification_summary(results):
    outcomes = Counter(
        result.metrics.get("clarification_outcome") for result in results
    )
    tp = outcomes.get("TP", 0)
    fp = outcomes.get("FP", 0)
    fn = outcomes.get("FN", 0)
    tn = outcomes.get("TN", 0)
    precision, recall, f1 = confusion_rates(tp, fp, fn, tn)
    scripted = [result for result in results if result.case.clarification_answer]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "scripted_cases": len(scripted),
        "scripted_successes": sum(
            bool(result.metrics.get("dialog_resolution_success")) for result in scripted
        ),
    }


@dataclass(frozen=True)
class LiveSummary:
    cases: int
    metric_rates: dict[str, Any]
    clarification: dict[str, Any]
    failure_stages: dict[str, int]
    mean_planning_latency_s: float | None
    mean_execution_latency_s: float | None
    mean_total_latency_s: float | None

    @classmethod
    def from_results(cls, results):
        metric_rates = {}
        for name in MAIN_METRICS:
            successes, observed = metric_counts(results, name)
            metric_rates[name] = {
                "successes": successes,
                "cases": observed,
                "percent": 100.0 * successes / observed if observed else None,
            }

        planned = [r for r in results if r.metrics.get("planning_attempted")]
        executed = [r for r in results if r.metrics.get("execution_attempted")]
        failure_stages = Counter(
            result.failure_stage for result in results if result.failure_stage
        )

        def _mean(values):
            values = list(values)
            return statistics.fmean(values) if values else None

        return cls(
            cases=len(results),
            metric_rates=metric_rates,
            clarification=clarification_summary(results),
            failure_stages=dict(sorted(failure_stages.items())),
            mean_planning_latency_s=_mean(r.planning_latency_s for r in planned),
            mean_execution_latency_s=_mean(r.execution_latency_s for r in executed),
            mean_total_latency_s=_mean(r.total_latency_s for r in results),
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


def exact_plan_match(case, payload):
    expected = case.expected_plan
    if case.expected_outcome != "plan" or not isinstance(expected, list):
        return None
    steps = payload.get("plan", []) if isinstance(payload, dict) else []
    return len(steps) == len(expected) and all(
        _step_matches(step, reference) for step, reference in zip(steps, expected)
    )


def allowed_names_for_step(step, names):
    action = step.get("action")
    relation = step.get("relation")
    return {
        "object": allowed_objects_for(action, names),
        "location": allowed_locations_for(action, relation, names),
        "source": names.sources,
    }


def has_hallucinated_name(plan, context):
    names = context_names(context)
    for step in plan.get("plan", []):
        if not isinstance(step, dict):
            continue
        for field_name, allowed_names in allowed_names_for_step(step, names).items():
            value = step.get(field_name)
            if value and value not in allowed_names:
                return True
    return False


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
        except Exception:
            schema_valid = False
        guard_valid = schema_valid and verify(normalized, context)[0]

    hallucination = (
        has_hallucinated_name(payload, context)
        if outcome == "plan" and isinstance(payload, dict)
        else None
    )

    return {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "guard_valid": guard_valid,
        "plan_valid": bool(schema_valid and guard_valid),
        "outcome_match": outcome == case.expected_outcome,
        "reference_goal_match": reference_match(case, payload),
        "exact_plan_match": exact_plan_match(case, payload),
        "hallucination": hallucination,
    }


CASES_FILE = Path(__file__).with_name("kitchen_eval_samples.jsonl")

KITCHEN_OBJECT_IDS = {
    "cheezeit",
    "gelatinbox",
    "gelatinbox_counter",
    "kettle",
    "mustard_bottle",
    "pringles",
    "saltcontainer",
    "soap_bottle",
    "tomatosoup",
    "tunacan",
    "wine_bottle",
}


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
    objects = set()
    for case in cases:
        for goal in case.goals():
            object_id = goal.get("object")
            if object_id in KITCHEN_OBJECT_IDS:
                objects.add(object_id)
    return objects


def validate_kitchen_inventory(context, label):
    live_objects = set(context.get("objects", []))
    if live_objects == KITCHEN_OBJECT_IDS:
        return

    missing = sorted(KITCHEN_OBJECT_IDS - live_objects)
    added = sorted(live_objects - KITCHEN_OBJECT_IDS)
    raise ValueError(
        f"{label} object inventory changed; missing={missing!r}, added={added!r}"
    )


def load_cases(path=CASES_FILE):
    cases = [
        LiveCase.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case.id for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("case file requires non-empty unique ids")

    for case in cases:
        if case.environment != "kitchen":
            raise ValueError(f"{case.id}: only the kitchen environment is supported")

    missing_objects = KITCHEN_OBJECT_IDS - covered_objects(cases)
    if missing_objects:
        missing = ", ".join(sorted(missing_objects))
        raise ValueError(f"case file does not cover Kitchen objects: {missing}")
    return cases
