import json
import logging
import re
import time
from dataclasses import dataclass, field

from .scoring import (
    EvaluationConfig,
    field_matches,
    LiveResult,
    outcome_error,
    plan_quality_metrics,
    validate_entities,
    validate_kitchen_inventory,
)
from .context import load_context

from ...validation.schema import DIRECTIONAL_RELATIONS

# Executor result phases that map directly to a failure stage.
EXECUTOR_STAGES = ("input", "world_setup", "grounding", "context_refresh", "executor")


def create_log(path):
    """Create a fresh line-oriented evaluation log."""
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
    """Write one structured and immediately flushed log event."""
    logger.log(
        level,
        "%s %s",
        event,
        json.dumps(payload, sort_keys=True, default=str),
    )


def log_exception(logger, event, payload):
    """Write a structured event together with the active exception traceback."""
    logger.exception(
        "%s %s",
        event,
        json.dumps(payload, sort_keys=True, default=str),
    )


def _trace(logger, event, case, phase, **details):
    """Write one case event when evaluation logging is enabled."""
    if logger is not None:
        log_event(
            logger,
            event,
            {"case_id": case.id, "phase": phase, **details},
        )


def wait(read, ready, timeout_s, poll_interval_s):
    """Poll a value until it is ready or the timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = read()
        if ready(value):
            return value
        time.sleep(poll_interval_s)
    raise TimeoutError(f"timed out after {timeout_s:.1f} seconds")


def wait_for_world(session, config):
    """Wait for a session world to become ready."""

    def world_ready():
        result = session.execution_result()
        if result and result.get("status") == "error":
            raise RuntimeError(result.get("error", "world setup failed"))
        return session.is_world_ready()

    wait(world_ready, bool, config.world_timeout_s, config.poll_interval_s)


def _empty_metrics():
    """Return the per-case metric fields with their pre-run defaults."""
    return {
        "planning_attempted": False,
        "execution_attempted": False,
        "json_valid": None,
        "schema_valid": None,
        "guard_valid": None,
        "plan_valid": None,
        "outcome_match": None,
        "reference_goal_match": None,
        "exact_plan_match": None,
        "hallucination": None,
        "planner_success": None,
        "grounding_success": None,
        "execution_success": None,
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
        "clarification_outcome": None,
        "clarification_target_correct": None,
        "dialog_resolution_success": None,
    }


def _final_location_goals(case):
    """Return the last expected location for each object.

    A reference can contain intermediate moves. Only the last destination is a
    final-state postcondition.
    """
    goals_by_object = {}
    for goal in case.goals():
        object_name = goal.get("object")
        if object_name and goal.get("location"):
            goals_by_object[object_name] = goal
    return list(goals_by_object.values())


def postcondition_metrics(case, final_context=None):
    """Check final object locations that ``world_context`` can observe.

    Missing observations are neutral. Exact directional relations are not
    stored in ``world_context`` and therefore remain unobserved.
    """
    if final_context is None:
        final_context = load_context()

    goals = _final_location_goals(case)
    metrics = {
        "postcondition_success": None,
        "postcondition_goal_count": len(goals),
        "postcondition_checked_count": 0,
        "postcondition_errors": [],
        "postcondition_unavailable": [],
    }
    locations = final_context.get("object_locations")

    for goal in goals:
        object_name = goal["object"]
        location = goal["location"]
        relation = goal.get("relation")

        if relation in DIRECTIONAL_RELATIONS:
            metrics["postcondition_unavailable"].append(
                f"{object_name!r}: exact {relation!r} is not stored in world_context"
            )
            continue
        if not isinstance(locations, dict) or not locations.get(object_name):
            metrics["postcondition_unavailable"].append(
                f"{object_name!r}: final location is not available in world_context"
            )
            continue

        actual = locations[object_name]
        expected = location if isinstance(location, list) else [location]
        metrics["postcondition_checked_count"] += 1
        if not set(actual).intersection(expected):
            metrics["postcondition_errors"].append(
                f"postcondition failed: {object_name!r} is at {actual!r}, "
                f"expected one of {expected!r}"
            )

    if metrics["postcondition_checked_count"]:
        metrics["postcondition_success"] = not metrics["postcondition_errors"]
    return metrics


def _expected_physical_checks(case):
    """Build physical checks from the benchmark goals."""
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
    park_requested = False
    for goal in goals:
        action = goal.get("action")
        object_name = goal.get("object")
        if action == "OpenAction":
            container_states[object_name] = "open"
        elif action == "CloseAction":
            container_states[object_name] = "closed"
        if action and object_name:
            final_object_actions[object_name] = action
        if action == "ParkArmsAction":
            park_requested = True

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

    if park_requested:
        checks.append({"type": "parked_arms"})

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
    """Return the newest matching directional observation."""
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
    """Return one physical check result, or None when it is unavailable."""
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

    if check_type == "parked_arms":
        parked = observations.get("arms_parked")
        return parked if isinstance(parked, bool) else None

    if check_type == "directional":
        result = _directional_result(observations, check)
        return result if isinstance(result, bool) else None

    return None


def _physical_check_name(check):
    """Return a short readable name for one physical check."""
    check_type = check["type"]
    if check_type == "navigation":
        return f"final navigation to {check['location']!r}"
    if check_type == "container":
        return f"{check['object']!r} is {check['state']}"
    if check_type == "held":
        return f"{check['object']!r} is held"
    if check_type == "parked_arms":
        return "both arms are parked"
    return f"{check['object']!r} is {check['relation']} " f"{check['location']!r}"


def physical_goal_metrics(case, observations):
    """Check goals that are absent from the serialized world context."""
    checks = _expected_physical_checks(case)
    metrics = {
        "physical_goal_success": None,
        "physical_goal_count": len(checks),
        "physical_goal_checked_count": 0,
        "physical_goal_errors": [],
        "physical_goal_unavailable": [],
    }
    if not checks:
        return metrics

    observations = observations if isinstance(observations, dict) else {}
    for check in checks:
        check_name = _physical_check_name(check)
        result = _physical_check_result(observations, check)
        if result is None:
            metrics["physical_goal_unavailable"].append(
                f"physical observation unavailable: {check_name}"
            )
            continue

        metrics["physical_goal_checked_count"] += 1
        if not result:
            metrics["physical_goal_errors"].append(
                f"physical goal failed: {check_name}"
            )

    metrics["physical_goal_success"] = bool(
        metrics["physical_goal_checked_count"] == len(checks)
        and not metrics["physical_goal_errors"]
    )
    return metrics


def combined_goal_metrics(case, postconditions, physical_goals):
    """Combine serialized locations and direct physical observations."""
    observable_locations = [
        goal
        for goal in _final_location_goals(case)
        if goal.get("relation") not in DIRECTIONAL_RELATIONS
    ]
    total = len(observable_locations) + physical_goals["physical_goal_count"]
    checked = (
        postconditions["postcondition_checked_count"]
        + physical_goals["physical_goal_checked_count"]
    )
    errors = (
        postconditions["postcondition_errors"] + physical_goals["physical_goal_errors"]
    )
    unavailable = list(physical_goals["physical_goal_unavailable"])
    missing_locations = (
        len(observable_locations) - postconditions["postcondition_checked_count"]
    )
    if missing_locations > 0:
        unavailable.append(
            f"{missing_locations} final object location observation(s) unavailable"
        )

    if not total:
        success = None
    else:
        success = bool(checked == total and not errors)

    return {
        "goal_success": success,
        "goal_count": total,
        "goal_checked_count": checked,
        "goal_errors": errors,
        "goal_unavailable": unavailable,
    }


def task_succeeded(
    case,
    planner_success,
    execution_status,
    goal_success,
    reference_goal_match=None,
    clarification_target_correct=None,
):
    """Return whether the required task goal was successfully completed."""
    if not planner_success:
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


def _run_planner(planner, instruction, context, conversation=None):
    """Run one planner turn and return its result, metadata, and latency."""
    started = time.perf_counter()
    outcome, payload, history = planner.plan(
        instruction,
        conversation=conversation,
        context=context,
    )
    metadata_reader = getattr(planner, "get_last_run_metrics", None)
    metadata = metadata_reader() if callable(metadata_reader) else {}
    return outcome, payload, history, metadata, time.perf_counter() - started


def _assess_planner_response(case, outcome, payload, context, metadata):
    """Validate one planner response and return its quality and error."""
    error = outcome_error(case, outcome, payload)
    if outcome == "error":
        rejection_reason = metadata.get("rejection_reason")
        if not rejection_reason and isinstance(payload, str):
            rejection_reason = payload
        if rejection_reason:
            error = f"planner validation failed: {rejection_reason}"
    quality = plan_quality_metrics(case, outcome, payload, context, metadata)
    if error is None and not quality["plan_valid"]:
        error = "planner response failed schema or guard validation"
    return quality, error


def _target_matches(question, targets):
    """Match a clarification question against explicit benchmark target terms."""
    words = re.findall(r"[a-z0-9]+", question.casefold())
    padded_question = f" {' '.join(words)} "
    for target in targets:
        target_words = re.findall(r"[a-z0-9]+", target.casefold())
        padded_target = f" {' '.join(target_words)} "
        if padded_target in padded_question:
            return True
    return False


def _execute_plan(case, session, payload, config, logger):
    """Submit one plan, wait for its result, and trace execution."""
    _trace(
        logger,
        "execution_start",
        case,
        "execution",
        step_count=len(payload.get("plan", [])),
    )
    started = time.perf_counter()
    session.execute_plan(payload)
    result = wait(
        session.execution_result,
        lambda value: isinstance(value, dict) and "status" in value,
        config.execution_timeout_s,
        config.poll_interval_s,
    )
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


def evaluate_case(case, planner, session, config=EvaluationConfig(), logger=None):
    """Plan and execute one case in a fresh world.

    The returned result carries ``failure_stage``: the first pipeline stage
    that failed, or ``None`` when the case succeeded.
    """
    started = time.perf_counter()
    result = LiveResult(case=case, metrics=_empty_metrics())
    try:
        _run_case(result, case, planner, session, config, logger)
    finally:
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
        else:
            _trace(logger, "world_stopped", case, "teardown")
    result.total_latency_s = time.perf_counter() - started
    return result


@dataclass
class _CaseStage:
    """Track the running pipeline stage so failures are attributed correctly."""

    name: str = "world_setup"
    started: float = field(default_factory=time.perf_counter)

    def enter(self, name):
        """Begin a new stage and restart its latency clock."""
        self.name = name
        self.started = time.perf_counter()

    def elapsed(self):
        """Return seconds spent in the current stage."""
        return time.perf_counter() - self.started


def world_context(session):
    """Project the live demo world into the planner context."""
    return session.context()


def _setup_world(case, session, config, logger):
    """Build a fresh CRAM world for the case and return its planner context."""
    session.setup_world(case.robot, case.environment)
    wait_for_world(session, config)
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
    """Compare the final world state with the case goals and record the outcome."""
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
    elif goals["goal_success"] is not True and (
        metrics["reference_goal_match"] is False
    ):
        result.error = "submitted plan does not contain the expected symbolic goal"
        result.execution_status = "goal_mismatch"
        result.failure_stage = "goal_verification"


@dataclass
class _PlanningResult:
    """What the planning stage hands to execution."""

    payload: object = None
    final_outcome: str = None
    error: str = None


def _plan_turns(result, case, planner, context, logger):
    """Run the planner turns for one case and record their quality metrics."""
    metrics = result.metrics
    metrics["planning_attempted"] = True
    outcome, payload, conversation, metadata, latency = _run_planner(
        planner, case.instruction, context
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

    # Confusion-matrix cell for the first response.
    asked = outcome == "clarification"
    if case.expected_outcome == "clarification":
        target_correct = bool(
            asked
            and isinstance(payload, str)
            and _target_matches(payload, case.expected_clarification_targets)
        )
        metrics["clarification_target_correct"] = target_correct
        metrics["clarification_outcome"] = "TP" if target_correct else "FN"
    else:
        metrics["clarification_outcome"] = "FP" if asked else "TN"

    # Scripted dialog: answer one clarification with a second user turn.
    final_outcome = outcome
    follow_up_case = case.follow_up_case()
    if error is None and asked and follow_up_case:
        _trace(logger, "clarification_answered", case, "clarification")
        final_outcome, payload, conversation, metadata, latency = _run_planner(
            planner, case.clarification_answer, context, conversation
        )
        result.planning_latency_s += latency
        if metadata.get("raw_response") is not None:
            result.raw_responses.append(metadata["raw_response"])
        follow_up_quality, error = _assess_planner_response(
            follow_up_case, final_outcome, payload, context, metadata
        )
        # Both dialog turns must be valid to count as one valid response.
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
        if not metrics["json_valid"]:
            result.failure_stage = "planner_json"
        elif not metrics["schema_valid"]:
            result.failure_stage = "planner_schema"
        elif not metrics["guard_valid"]:
            result.failure_stage = "planner_guard"
        else:
            result.failure_stage = "planner_mode"
        return _PlanningResult(payload, final_outcome, error)

    result.planner_success = True
    metrics["planner_success"] = True
    return _PlanningResult(payload, final_outcome, None)


def _execute(result, case, session, planning, config, stage, logger):
    """Submit the validated plan and record the executor outcome."""
    metrics = result.metrics
    if planning.final_outcome == "clarification":
        result.execution_status = "not_required"
        return None

    if not isinstance(planning.payload, dict):
        result.planner_success = False
        metrics["planner_success"] = False
        result.error = f"planner returned non-object payload: {planning.payload!r}"
        result.failure_stage = "planning"
        return None

    result.plan = planning.payload
    stage.enter("execution")
    metrics["execution_attempted"] = True
    if config.visualization_delay_s > 0:
        time.sleep(config.visualization_delay_s)
    executor_result, latency = _execute_plan(
        case, session, planning.payload, config, logger
    )
    result.execution_latency_s += latency
    status = executor_result.get("status", "error")
    executor_phase = executor_result.get("phase")
    result.execution_status = status
    metrics["execution_success"] = status == "ok"
    if status == "ok" or executor_phase in ("execution", "context_refresh"):
        metrics["grounding_success"] = True
    elif executor_phase == "grounding":
        metrics["grounding_success"] = False
    if status != "ok":
        result.error = executor_result.get("error")
        if executor_phase in EXECUTOR_STAGES:
            result.failure_stage = executor_phase
        else:
            result.failure_stage = "execution"
    return executor_result


def _run_case(result, case, planner, session, config, logger):
    """Run the pipeline stages for one case and fill the result in place."""
    metrics = result.metrics
    stage = _CaseStage()
    try:
        context = _setup_world(case, session, config, logger)

        stage.enter("planning")
        planning = _plan_turns(result, case, planner, context, logger)
        if planning.error is not None:
            return

        executor_result = _execute(
            result, case, session, planning, config, stage, logger
        )
        if result.failure_stage is not None:
            return

        # Goal verification: compare the final world state with the case goals.
        if result.execution_status == "ok":
            stage.enter("goal_verification")
            _verify_goals(result, case, session, executor_result, logger)

        clarification_missed_target = (
            metrics["clarification_target_correct"] is False
        )
        first_failure_unrecorded = result.error is None
        if clarification_missed_target and first_failure_unrecorded:
            result.error = "clarification question did not match the expected target"
            result.failure_stage = "planner_mode"

        result.task_success = task_succeeded(
            case,
            result.planner_success,
            result.execution_status,
            metrics["goal_success"],
            metrics["reference_goal_match"],
            metrics["clarification_target_correct"],
        )
        if case.clarification_answer:
            metrics["dialog_resolution_success"] = result.task_success

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
            timed_out = isinstance(exc, TimeoutError)
            result.execution_status = "timeout" if timed_out else "error"
            result.failure_stage = "timeout" if timed_out else "execution"
            metrics["execution_success"] = False
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


def validate_demo_world(cases, session, config):
    """Check all cases against one live Kitchen context."""
    try:
        session.setup_world("hsrb", "kitchen")
        wait_for_world(session, config)
        context = world_context(session)
        validate_kitchen_inventory(context, "live Kitchen")
        for case in cases:
            validate_entities(case, context, "live Kitchen")
    finally:
        session.stop_world()
