from __future__ import annotations

import json
import logging
import re
import signal
import threading
import time
from dataclasses import asdict, dataclass, field

from thesis_demo.tools.evaluation.scoring import (
    empty_metrics,
    first_failed_check,
    EvaluationConfiguration,
    EvaluationMode,
    LiveResult,
    outcome_error,
    plan_quality_metrics,
    task_succeeded,
    validate_entities,
    validate_kitchen_inventory,
    verified_goal_metrics,
)

IGNORED_CLARIFICATION_WORDS = {
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


class CaseTimeoutError(TimeoutError):
    pass


@dataclass
class CaseDeadline:
    seconds: float | None
    previous_handler: object = field(init=False, default=None)
    active: bool = field(init=False, default=False)

    def __enter__(self):
        if self.seconds is None:
            return self
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("case deadlines require the main thread")

        # pyCRAM has no timeout API for its synchronous execution calls.
        self.previous_handler = signal.signal(signal.SIGALRM, self._expire)
        self.active = True
        self.reset()
        return self

    def __exit__(self, _error_type, _error, _traceback):
        if not self.active:
            return
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        self.active = False

    def reset(self):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def _expire(self, _signal_number, _frame):
        raise CaseTimeoutError(f"case made no progress for {self.seconds:g} seconds")


@dataclass(eq=False)
class _ActionProgressHandler(logging.Handler):
    callback: object

    def __post_init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        prefix = "Performing action "
        message = record.getMessage()
        if message.startswith(prefix):
            self.callback(message.removeprefix(prefix))


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


def _assess_planner_response(case, outcome, payload, metadata):
    error = outcome_error(case, outcome)
    if outcome == "error":
        rejection_reason = metadata.get("rejection_reason")
        if not rejection_reason and isinstance(payload, str):
            rejection_reason = payload
        if rejection_reason:
            error = f"planner validation failed: {rejection_reason}"
    quality = plan_quality_metrics(case, outcome, payload, metadata)
    if error is None and not quality["json_object_valid"]:
        error = "planner response is not exactly one JSON object"
    elif error is None and not (
        quality["schema_valid"]
        and quality["names_valid"] is not False
        and quality["sequence_valid"] is not False
    ):
        error = "planner response failed schema or guard validation"
    elif (
        error is None
        and case.expected_outcome == "plan"
        and quality["planned_goal_match"] is not True
    ):
        error = "planner response does not contain the expected symbolic goal"
    return quality, error


def _target_matches(question, targets):
    question_words = set(re.findall(r"[a-z0-9]+", question.casefold()))
    for target in targets:
        target_words = set(re.findall(r"[a-z0-9]+", target.casefold()))
        meaningful_words = target_words - IGNORED_CLARIFICATION_WORDS
        if meaningful_words and meaningful_words.issubset(question_words):
            return True
    return False


def _execute_plan(case, session, payload, logger, deadline):
    _trace(
        logger,
        "execution_start",
        case,
        "execution",
        step_count=len(payload.get("plan", [])),
    )
    started = time.perf_counter()

    def step_completed(step_index, step):
        deadline.reset()
        _trace(
            logger,
            "execution_progress",
            case,
            "execution",
            step_index=step_index,
            action=step.get("action"),
        )

    def action_started(action):
        deadline.reset()
        _trace(
            logger,
            "execution_action_started",
            case,
            "execution",
            action=action,
        )

    deadline.reset()
    action_logger = logging.getLogger("pycram.robot_plans.actions.base")
    # pyCRAM has no callback for nested actions, so its action log marks progress.
    progress_handler = _ActionProgressHandler(action_started)
    action_logger.addHandler(progress_handler)
    try:
        session.execute_plan(payload, step_callback=step_completed)
    finally:
        action_logger.removeHandler(progress_handler)
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
    except Exception as error:
        if logger is not None:
            log_exception(
                logger,
                "case_exception",
                {
                    "case_id": case.id,
                    "phase": "teardown",
                    "error_type": type(error).__name__,
                    "error": str(error),
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
        metrics=empty_metrics(),
    )
    try:
        with CaseDeadline(configuration.case_timeout_s) as deadline:
            _run_case(
                result,
                case,
                planner,
                session,
                configuration,
                logger,
                inference,
                deadline,
            )
    finally:
        _stop_world_logged(case, session, logger)
    result.total_latency_s = time.perf_counter() - started
    result.metrics["first_failed_check"] = first_failed_check(
        result.metrics,
        result.reachability_success,
        result.execution_success,
    )
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


def _wait_for_world(delay_s):
    if delay_s <= 0:
        return
    # pyCRAM provides no readiness signal after constructing this demo world.
    time.sleep(delay_s)


def _setup_world(case, session, configuration, logger):
    session.setup_world(case.robot, case.environment)
    _wait_for_world(configuration.world_settle_delay_s)
    context = session.context()
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
    goals = verified_goal_metrics(
        case,
        session.context(),
        executor_result.get("observations"),
    )
    result.metrics.update(goals)
    _trace(
        logger,
        "goal_summary",
        case,
        "goal_verification",
        expected=goals["world_checks_expected"],
        evaluated=goals["world_checks_evaluated"],
        reached=goals["world_goal_reached"],
        failures=len(goals["world_failures"]),
        unchecked=len(goals["world_unchecked"]),
    )

    if goals["world_goal_reached"] is False:
        problems = goals["world_failures"] or goals["world_unchecked"]
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
    metrics["attempts"] = metadata.get("attempts", 1)
    if metadata.get("raw_response") is not None:
        result.raw_responses.append(metadata["raw_response"])
    quality, error = _assess_planner_response(case, outcome, payload, metadata)
    metrics.update(quality)
    _trace(
        logger,
        "planner_complete",
        case,
        "planning",
        outcome=outcome,
        success=error is None,
        latency_s=latency,
        attempts=metrics["attempts"],
    )

    asked = outcome == "clarification"
    if case.expected_outcome == "clarification":
        target_correct = bool(
            asked
            and isinstance(payload, str)
            and _target_matches(payload, case.expected_clarification_targets)
        )
        metrics["clarification_target_match"] = target_correct
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
            follow_up_case, final_outcome, payload, metadata
        )
        for name in ("json_object_valid", "schema_valid", "outcome_match"):
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


def _execute(
    result,
    case,
    session,
    planning,
    configuration,
    stage,
    logger,
    deadline,
):
    if planning.final_outcome == "clarification":
        result.execution_status = "not_required"
        return None

    if not isinstance(planning.payload, dict):
        result.planning_success = False
        result.error = f"planner returned non-object payload: {planning.payload!r}"
        result.failure_stage = "planning"
        return None

    stage.enter("execution")
    if configuration.visualization_delay_s > 0:
        time.sleep(configuration.visualization_delay_s)
    executor_result, latency = _execute_plan(
        case,
        session,
        planning.payload,
        logger,
        deadline,
    )
    result.execution_latency_s += latency
    status = executor_result.get("status", "error")
    executor_phase = executor_result.get("phase")
    result.execution_status = status
    if status == "ok":
        result.reachability_success = True
        result.execution_success = True
    elif executor_phase == "grounding":
        result.reachability_success = False
    elif executor_phase in ("execution", "context_refresh"):
        result.reachability_success = True
        result.execution_success = False
    if status != "ok":
        result.error = executor_result.get("error")
        result.failure_stage = (
            "grounding" if executor_phase == "grounding" else "execution"
        )
    return executor_result


def _run_case(
    result,
    case,
    planner,
    session,
    configuration,
    logger,
    inference,
    deadline,
):
    metrics = result.metrics
    stage = _CaseStage()
    try:
        context = _setup_world(case, session, configuration, logger)

        stage.enter("planning")
        deadline.reset()
        planning = _plan_turns(result, case, planner, context, logger, inference)
        if planning.error is not None:
            return

        if isinstance(planning.payload, dict):
            result.plan = planning.payload
        if configuration.mode == EvaluationMode.PLANNING:
            return

        executor_result = _execute(
            result,
            case,
            session,
            planning,
            configuration,
            stage,
            logger,
            deadline,
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
            metrics["world_goal_reached"],
            metrics["planned_goal_match"],
            metrics["clarification_target_match"],
        )
        failed_without_a_stage = (
            not result.task_success and result.failure_stage is None
        )
        if failed_without_a_stage:
            result.failure_stage = "goal_verification"
            result.error = result.error or "required goals could not be confirmed"
    except CaseTimeoutError as error:
        _record_case_failure(result, case, stage, logger, error, timed_out=True)
    except Exception as error:
        _record_case_failure(result, case, stage, logger, error, timed_out=False)


def _record_case_failure(result, case, stage, logger, error, timed_out):
    elapsed = stage.elapsed()
    result.timed_out = timed_out
    result.failure_stage = stage.name
    result.error = str(error) if timed_out else f"{type(error).__name__}: {error}"

    if stage.name == "execution":
        result.execution_latency_s = elapsed
        result.execution_status = "timeout" if timed_out else "error"
        result.execution_success = False
    elif stage.name == "planning":
        result.planning_latency_s = elapsed
        if timed_out:
            result.planning_success = False

    if logger is not None:
        log_exception(
            logger,
            "case_timeout" if timed_out else "case_exception",
            {
                "case_id": case.id,
                "phase": stage.name,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


def validate_demo_world(cases, session, world_settle_delay_s=0.0):
    try:
        session.setup_world("hsrb", "kitchen")
        _wait_for_world(world_settle_delay_s)
        context = session.context()
        validate_kitchen_inventory(context, cases, "live Kitchen")
        for case in cases:
            validate_entities(case, context, "live Kitchen")
    finally:
        session.stop_world()
