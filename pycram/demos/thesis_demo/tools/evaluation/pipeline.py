from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field

from thesis_demo.tools.evaluation.scoring import (
    EvaluationConfiguration,
    EvaluationMode,
    field_matches,
    LiveResult,
    outcome_error,
    plan_quality_metrics,
    validate_entities,
    validate_kitchen_inventory,
)
from thesis_demo.validation.schema import DIRECTIONAL_RELATIONS


def create_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"evaluation.{path.resolve()}")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(logger, event, payload, level=logging.INFO):
    logger.log(
        level,
        "%s %s",
        event,
        json.dumps(payload, sort_keys=True, default=str),
    )


def log_exception(logger, event, payload):
    logger.exception(
        "%s %s",
        event,
        json.dumps(payload, sort_keys=True, default=str),
    )


def _trace(logger, event, case, phase, **details):
    if logger is not None:
        log_event(
            logger,
            event,
            {"case_id": case.id, "phase": phase, **details},
        )


def _empty_metrics():
    return {
        "json_valid": None,
        "schema_valid": None,
        "guard_valid": None,
        "plan_valid": None,
        "outcome_match": None,
        "reference_goal_match": None,
        "postcondition_success": None,
        "postcondition_goal_count": 0,
        "postcondition_checked_count": 0,
        "postcondition_errors": [],
        "postcondition_unavailable": [],
        "physical_goal_success": None,
        "physical_goal_count": 0,
        "physical_goal_checked_count": 0,
        "physical_goal_errors": [],
        "physical_goal_unavailable": [],
        "goal_success": None,
        "goal_count": 0,
        "goal_checked_count": 0,
        "goal_errors": [],
        "goal_unavailable": [],
        "clarification_target_correct": None,
    }


def _final_location_goals(case):
    goals_by_object = {}
    for goal in case.goals():
        object_name = goal.get("object")
        if object_name and goal.get("location"):
            goals_by_object[object_name] = goal
    return list(goals_by_object.values())


@dataclass
class GoalChecks:
    success: bool | None = None
    count: int = 0
    checked_count: int = 0
    errors: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)

    def record_checked(self, error=None):
        self.checked_count += 1
        if error is not None:
            self.errors.append(error)

    def as_metrics(self, prefix):
        return {
            f"{prefix}success": self.success,
            f"{prefix}count": self.count,
            f"{prefix}checked_count": self.checked_count,
            f"{prefix}errors": self.errors,
            f"{prefix}unavailable": self.unavailable,
        }


def postcondition_metrics(case, final_context):
    goals = _final_location_goals(case)
    checks = GoalChecks(count=len(goals))
    locations = final_context.get("object_locations")

    for goal in goals:
        object_name = goal["object"]
        location = goal["location"]
        relation = goal.get("relation")

        if relation in DIRECTIONAL_RELATIONS:
            checks.unavailable.append(
                f"{object_name!r}: exact {relation!r} is not stored in world_context"
            )
            continue
        if not isinstance(locations, dict) or not locations.get(object_name):
            checks.unavailable.append(
                f"{object_name!r}: final location is not available in world_context"
            )
            continue

        actual = locations[object_name]
        expected = location if isinstance(location, list) else [location]
        error = None
        if not set(actual).intersection(expected):
            error = (
                f"postcondition failed: {object_name!r} is at {actual!r}, "
                f"expected one of {expected!r}"
            )
        checks.record_checked(error)

    if checks.checked_count:
        checks.success = not checks.errors
    metrics = checks.as_metrics("postcondition_")
    metrics["postcondition_goal_count"] = metrics.pop("postcondition_count")
    return metrics


def expected_physical_checks(case):
    # Map benchmark goals to the observations exposed by the demo executor.
    goals = case.goals()
    if not goals:
        return []

    checks = []
    last_goal = goals[-1]
    if last_goal.get("action") == "NavigateAction":
        checks.append(
            {
                "type": "navigation",
                "location": last_goal.get("location"),
            }
        )

    container_states = {}
    final_object_actions = {}
    for goal in goals:
        action = goal.get("action")
        object_name = goal.get("object")
        if action == "OpenAction":
            container_states[object_name] = "open"
        elif action == "CloseAction":
            container_states[object_name] = "closed"
        if action and object_name:
            final_object_actions[object_name] = action

    for container_name, expected_state in container_states.items():
        checks.append(
            {
                "type": "container",
                "object": container_name,
                "state": expected_state,
            }
        )

    for object_name, action in final_object_actions.items():
        if action == "PickUpAction":
            checks.append({"type": "held", "object": object_name})

    for goal in _final_location_goals(case):
        if goal.get("relation") in DIRECTIONAL_RELATIONS:
            checks.append(
                {
                    "type": "directional",
                    "object": goal["object"],
                    "location": goal["location"],
                    "relation": goal["relation"],
                }
            )
    return checks


def _directional_result(observations, check):
    results = observations.get("directional_relations")
    if not isinstance(results, list):
        return None

    for result in reversed(results):
        same_object = result.get("object") == check["object"]
        same_relation = result.get("relation") == check["relation"]
        same_location = field_matches(
            result.get("location"),
            check["location"],
        )
        if same_object and same_relation and same_location:
            return result.get("success")
    return None


def _physical_check_result(observations, check):
    check_type = check["type"]
    if check_type == "navigation":
        navigation = observations.get("navigation")
        if not isinstance(navigation, list) or not navigation:
            return None
        final_navigation = navigation[-1]
        correct_target = field_matches(
            final_navigation.get("location"),
            check["location"],
        )
        return bool(correct_target and final_navigation.get("success") is True)

    if check_type == "container":
        states = observations.get("container_states")
        if not isinstance(states, dict):
            return None
        actual_state = states.get(check["object"])
        if actual_state == "unknown" or actual_state is None:
            return None
        return actual_state == check["state"]

    if check_type == "held":
        held_objects = observations.get("held_objects")
        if not isinstance(held_objects, list):
            return None
        return check["object"] in held_objects

    if check_type == "directional":
        result = _directional_result(observations, check)
        return result if isinstance(result, bool) else None

    return None


def _physical_check_name(check):
    check_type = check["type"]
    if check_type == "navigation":
        return f"final navigation to {check['location']!r}"
    if check_type == "container":
        return f"{check['object']!r} is {check['state']}"
    if check_type == "held":
        return f"{check['object']!r} is held"
    return f"{check['object']!r} is {check['relation']} {check['location']!r}"


def physical_goal_metrics(case, observations):
    expected_checks = expected_physical_checks(case)
    checks = GoalChecks(count=len(expected_checks))
    if not expected_checks:
        return checks.as_metrics("physical_goal_")

    observations = observations if isinstance(observations, dict) else {}
    for check in expected_checks:
        check_name = _physical_check_name(check)
        result = _physical_check_result(observations, check)
        if result is None:
            checks.unavailable.append(f"physical observation unavailable: {check_name}")
            continue
        error = None if result else f"physical goal failed: {check_name}"
        checks.record_checked(error)

    checks.success = bool(checks.checked_count == checks.count and not checks.errors)
    return checks.as_metrics("physical_goal_")


def combined_goal_metrics(case, postconditions, physical_goals):
    observable_locations = [
        goal
        for goal in _final_location_goals(case)
        if goal.get("relation") not in DIRECTIONAL_RELATIONS
    ]
    checks = GoalChecks(
        count=len(observable_locations) + physical_goals["physical_goal_count"],
        checked_count=(
            postconditions["postcondition_checked_count"]
            + physical_goals["physical_goal_checked_count"]
        ),
        errors=(
            postconditions["postcondition_errors"]
            + physical_goals["physical_goal_errors"]
        ),
        unavailable=list(physical_goals["physical_goal_unavailable"]),
    )
    missing_locations = (
        len(observable_locations) - postconditions["postcondition_checked_count"]
    )
    if missing_locations > 0:
        checks.unavailable.append(
            f"{missing_locations} final object location observation(s) unavailable"
        )

    if checks.count:
        checks.success = bool(
            checks.checked_count == checks.count and not checks.errors
        )
    return checks.as_metrics("goal_")


def task_succeeded(
    case,
    planning_success,
    execution_status,
    goal_success,
    reference_goal_match=None,
    clarification_target_correct=None,
):
    if not planning_success:
        return False
    if case.expected_outcome == "clarification":
        if clarification_target_correct is not True:
            return False
        if not case.clarification_answer:
            return execution_status == "not_required"
    if execution_status != "ok":
        return False
    if goal_success is not None:
        return goal_success
    return reference_goal_match is True


def _run_planner(planner, instruction, context, conversation=None, inference=None):
    started = time.perf_counter()
    planner_result = planner.plan(
        instruction,
        conversation=conversation,
        context=context,
        inference=inference,
    )
    return (
        planner_result.outcome,
        planner_result.payload,
        planner_result.history,
        asdict(planner_result.metrics),
        time.perf_counter() - started,
    )


def _assess_planner_response(case, outcome, payload, context, metadata):
    error = outcome_error(case, outcome)
    if outcome == "error":
        rejection_reason = metadata.get("rejection_reason")
        if not rejection_reason and isinstance(payload, str):
            rejection_reason = payload
        if rejection_reason:
            error = f"planner validation failed: {rejection_reason}"
    quality = plan_quality_metrics(case, outcome, payload, context, metadata)
    if error is None and not quality["json_valid"]:
        error = "planner response is not exactly one JSON object"
    elif error is None and not quality["plan_valid"]:
        error = "planner response failed schema or guard validation"
    elif (
        error is None
        and case.expected_outcome == "plan"
        and quality["reference_goal_match"] is not True
    ):
        error = "planner response does not contain the expected symbolic goal"
    return quality, error


def _target_matches(question, targets):
    ignored_words = {
        "a",
        "an",
        "at",
        "by",
        "from",
        "i",
        "in",
        "it",
        "me",
        "of",
        "on",
        "or",
        "that",
        "the",
        "them",
        "this",
        "to",
        "we",
        "you",
    }
    question_words = set(re.findall(r"[a-z0-9]+", question.casefold()))
    for target in targets:
        target_words = set(re.findall(r"[a-z0-9]+", target.casefold()))
        meaningful_words = target_words - ignored_words
        if meaningful_words and meaningful_words.issubset(question_words):
            return True
    return False


def _execute_plan(case, session, payload, logger):
    _trace(
        logger,
        "execution_start",
        case,
        "execution",
        step_count=len(payload.get("plan", [])),
    )
    started = time.perf_counter()
    session.execute_plan(payload)
    result = session.execution_result()
    if not isinstance(result, dict) or "status" not in result:
        raise RuntimeError("synchronous execution returned no status")
    latency = time.perf_counter() - started
    _trace(
        logger,
        "execution_complete",
        case,
        result.get("phase") or "execution",
        status=result.get("status", "error"),
        latency_s=latency,
    )
    return result, latency


def _stop_world_logged(case, session, logger):
    try:
        session.stop_world()
    except Exception as exc:
        if logger is not None:
            log_exception(
                logger,
                "case_exception",
                {
                    "case_id": case.id,
                    "phase": "teardown",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    _trace(logger, "world_stopped", case, "teardown")


def evaluate_case(
    case,
    planner,
    session,
    configuration=None,
    logger=None,
    model_variant=None,
    inference=None,
):
    configuration = configuration or EvaluationConfiguration()
    started = time.perf_counter()
    result = LiveResult(
        case=case,
        model_variant=model_variant,
        evaluation_mode=configuration.mode,
        inference_parameters=asdict(inference) if inference is not None else {},
        metrics=_empty_metrics(),
    )
    try:
        _run_case(result, case, planner, session, configuration, logger, inference)
    finally:
        _stop_world_logged(case, session, logger)
    result.total_latency_s = time.perf_counter() - started
    return result


@dataclass
class _CaseStage:
    name: str = "world_setup"
    started: float = field(default_factory=time.perf_counter)

    def enter(self, name):
        self.name = name
        self.started = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.started


def world_context(session):
    return session.context()


def _setup_world(case, session, logger):
    session.setup_world(case.robot, case.environment)
    context = world_context(session)
    if case.environment == "kitchen":
        validate_entities(case, context, "live Kitchen")
    _trace(
        logger,
        "world_ready",
        case,
        "world",
        robot=case.robot,
        environment=case.environment,
    )
    return context


def _verify_goals(result, case, session, executor_result, logger):
    metrics = result.metrics
    postconditions = postcondition_metrics(case, world_context(session))
    physical_goals = physical_goal_metrics(case, executor_result.get("observations"))
    goals = combined_goal_metrics(case, postconditions, physical_goals)
    metrics.update(postconditions)
    metrics.update(physical_goals)
    metrics.update(goals)
    _trace(
        logger,
        "goal_summary",
        case,
        "goal_verification",
        goal_count=goals["goal_count"],
        goal_checked_count=goals["goal_checked_count"],
        goal_success=goals["goal_success"],
        errors=len(goals["goal_errors"]),
        unavailable=len(goals["goal_unavailable"]),
    )

    if goals["goal_success"] is False:
        problems = goals["goal_errors"] or goals["goal_unavailable"]
        result.error = problems[0] if problems else "required goal observation failed"
        result.execution_status = "goal_mismatch"
        result.failure_stage = "goal_verification"


@dataclass
class _PlanningResult:
    payload: object = None
    final_outcome: str = None
    error: str = None


def _plan_turns(result, case, planner, context, logger, inference):
    metrics = result.metrics
    result.planning_success = False
    outcome, payload, conversation, metadata, latency = _run_planner(
        planner, case.instruction, context, inference=inference
    )
    result.planner_outcome = outcome
    result.planning_latency_s += latency
    if metadata.get("raw_response") is not None:
        result.raw_responses.append(metadata["raw_response"])
    quality, error = _assess_planner_response(case, outcome, payload, context, metadata)
    metrics.update(quality)
    _trace(
        logger,
        "planner_complete",
        case,
        "planning",
        outcome=outcome,
        success=error is None,
        latency_s=latency,
    )

    asked = outcome == "clarification"
    if case.expected_outcome == "clarification":
        target_correct = bool(
            asked
            and isinstance(payload, str)
            and _target_matches(payload, case.expected_clarification_targets)
        )
        metrics["clarification_target_correct"] = target_correct
        if error is None and not target_correct:
            error = "clarification question did not match the expected target"

    final_outcome = outcome
    follow_up_case = case.follow_up_case()
    if error is None and asked and follow_up_case:
        _trace(logger, "clarification_answered", case, "clarification")
        final_outcome, payload, conversation, metadata, latency = _run_planner(
            planner,
            case.clarification_answer,
            context,
            conversation,
            inference,
        )
        result.planning_latency_s += latency
        if metadata.get("raw_response") is not None:
            result.raw_responses.append(metadata["raw_response"])
        follow_up_quality, error = _assess_planner_response(
            follow_up_case, final_outcome, payload, context, metadata
        )
        for name in (
            "json_valid",
            "schema_valid",
            "guard_valid",
            "plan_valid",
            "outcome_match",
        ):
            follow_up_quality[name] = bool(quality[name] and follow_up_quality[name])
        metrics.update(follow_up_quality)
        _trace(
            logger,
            "clarification_follow_up_complete",
            case,
            "planning",
            outcome=final_outcome,
            success=error is None,
            latency_s=latency,
        )

    if error is not None:
        result.error = error
        result.failure_stage = "planning"
        return _PlanningResult(payload, final_outcome, error)

    result.planning_success = True
    return _PlanningResult(payload, final_outcome, None)


def _execute(result, case, session, planning, configuration, stage, logger):
    if planning.final_outcome == "clarification":
        result.execution_status = "not_required"
        return None

    if not isinstance(planning.payload, dict):
        result.planning_success = False
        result.error = f"planner returned non-object payload: {planning.payload!r}"
        result.failure_stage = "planning"
        return None

    result.plan = planning.payload
    stage.enter("execution")
    if configuration.visualization_delay_s > 0:
        time.sleep(configuration.visualization_delay_s)
    executor_result, latency = _execute_plan(case, session, planning.payload, logger)
    result.execution_latency_s += latency
    status = executor_result.get("status", "error")
    executor_phase = executor_result.get("phase")
    result.execution_status = status
    if status == "ok":
        result.grounding_success = True
        result.execution_success = True
    elif executor_phase == "grounding":
        result.grounding_success = False
    elif executor_phase in ("execution", "context_refresh"):
        result.grounding_success = True
        result.execution_success = False
    if status != "ok":
        result.error = executor_result.get("error")
        result.failure_stage = (
            "grounding" if executor_phase == "grounding" else "execution"
        )
    return executor_result


def _run_case(result, case, planner, session, configuration, logger, inference):
    metrics = result.metrics
    stage = _CaseStage()
    try:
        context = _setup_world(case, session, logger)

        stage.enter("planning")
        planning = _plan_turns(result, case, planner, context, logger, inference)
        if planning.error is not None:
            return

        if isinstance(planning.payload, dict):
            result.plan = planning.payload
        if configuration.mode == EvaluationMode.PLANNING:
            return

        executor_result = _execute(
            result, case, session, planning, configuration, stage, logger
        )
        if result.failure_stage is not None:
            return

        if result.execution_status == "ok":
            stage.enter("goal_verification")
            _verify_goals(result, case, session, executor_result, logger)

        result.task_success = task_succeeded(
            case,
            result.planning_success,
            result.execution_status,
            metrics["goal_success"],
            metrics["reference_goal_match"],
            metrics["clarification_target_correct"],
        )
        failed_without_a_stage = (
            not result.task_success and result.failure_stage is None
        )
        if failed_without_a_stage:
            result.failure_stage = "goal_verification"
            result.error = result.error or "required goals could not be confirmed"
    except Exception as exc:
        elapsed = stage.elapsed()
        result.error = f"{type(exc).__name__}: {exc}"
        if stage.name == "execution":
            result.execution_latency_s += elapsed
            result.execution_status = "error"
            result.execution_success = False
            result.failure_stage = "execution"
        else:
            if stage.name == "planning":
                result.planning_latency_s += elapsed
            result.failure_stage = stage.name
        if logger is not None:
            log_exception(
                logger,
                "case_exception",
                {
                    "case_id": case.id,
                    "phase": stage.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )


def validate_demo_world(cases, session):
    try:
        session.setup_world("hsrb", "kitchen")
        context = world_context(session)
        validate_kitchen_inventory(context, cases, "live Kitchen")
        for case in cases:
            validate_entities(case, context, "live Kitchen")
    finally:
        session.stop_world()
