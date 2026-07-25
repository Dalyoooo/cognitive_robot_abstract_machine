import json
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from thesis_demo_msgs.action import ExecutePlan

from thesis_demo.execution.execution import run_plan
from thesis_demo.execution.grounding import GroundingError
from thesis_demo.planner.world_context import build_world_context
from thesis_demo.validation.schema import parse_plan
from thesis_demo.world.nlp_demo import build_world

ACTION_NAME = "execute_plan"
CONTEXT_TOPIC = "world_context"


def _run_dir():
    path = Path(os.environ.get("NLP_RUN_DIR", "~/nlp-binder-run")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _context_file():
    return _run_dir() / "world_context.json"


def _result_file():
    return _run_dir() / "plan_result.json"


def _atomic_write_json(path, data):
    # The Binder reads these files concurrently, so it must never see partial JSON.
    temporary_path = path.with_name(f".{path.name}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(temporary_path, path)


def _error_result(phase, error, **extra):
    return {"status": "error", "phase": phase, "error": error, **extra}


def _parse_steps(plan_json):
    try:
        plan_data = json.loads(plan_json)
    except json.JSONDecodeError as error:
        raise ValueError(f"Plan is not valid JSON: {error}") from error
    return [step.as_dict() for step in parse_plan(plan_data)]


def _latched_qos():
    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


@dataclass
class PlanExecutor:
    node: object
    world: object
    robot: object
    context: object
    busy: bool = False
    context_publisher: object = field(init=False, default=None)
    action_server: object = field(init=False, default=None)

    def start(self):
        self.context_publisher = self.node.create_publisher(
            String, CONTEXT_TOPIC, _latched_qos()
        )
        self.publish_context()
        self.action_server = ActionServer(
            self.node,
            ExecutePlan,
            ACTION_NAME,
            execute_callback=self.execute,
            goal_callback=self.accept_or_reject,
            cancel_callback=self.reject_cancellation,
        )

    def accept_or_reject(self, _goal_request):
        if self.busy:
            return GoalResponse.REJECT
        self.busy = True
        return GoalResponse.ACCEPT

    def reject_cancellation(self, _cancel_request):
        return CancelResponse.REJECT

    def execute(self, goal_handle):
        self.busy = True
        try:
            result = self._run_requested_plan(goal_handle)
            result = self._refresh_context_before_result(result)
            _atomic_write_json(_result_file(), result)
            # The Binder receives domain failures inside result_json.
            goal_handle.succeed()
            response = ExecutePlan.Result()
            response.result_json = json.dumps(result)
            return response
        finally:
            self.busy = False

    def publish_context(self):
        context_data = build_world_context(self.world, self.robot)
        _atomic_write_json(_context_file(), context_data)
        message = String()
        message.data = json.dumps(context_data)
        self.context_publisher.publish(message)

    def _run_requested_plan(self, goal_handle):
        try:
            steps = _parse_steps(goal_handle.request.plan_json)
        except ValueError as error:
            return _error_result("input", str(error))

        def publish_feedback(step_index, step):
            feedback = ExecutePlan.Feedback()
            feedback.step_index = step_index
            feedback.step_json = json.dumps(step)
            goal_handle.publish_feedback(feedback)

        try:
            observations = run_plan(
                self.world,
                self.robot,
                self.context,
                steps,
                step_callback=publish_feedback,
            )
        except GroundingError as error:
            return _error_result(
                "grounding", str(error), grounding_error=error.to_dict()
            )
        except Exception as error:
            self.node.get_logger().error(
                f"Execution failed: {error!r}\n{traceback.format_exc()}"
            )
            return _error_result(
                "execution",
                f"{type(error).__name__}: {error}",
                error_type=type(error).__name__,
            )
        return {"status": "ok", "phase": "execution", "observations": observations}

    def _refresh_context_before_result(self, result):
        try:
            self.publish_context()
        except Exception as error:
            self.node.get_logger().error(f"Context refresh failed: {error!r}")
            # The stale context log no longer matches the world.
            _context_file().unlink(missing_ok=True)
            return _error_result(
                "context_refresh",
                f"{type(error).__name__}: {error}",
                previous_result=result,
            )
        return result


def _build_selected_world():
    selection = json.loads(os.environ.get("NLP_WORLD_SELECTION", "{}"))
    return build_world(
        robot_name=selection.get("robot", "hsrb"),
        environment=selection.get("environment", "apartment"),
        visualize=os.environ.get("NLP_VISUALIZE", "1") != "0",
    )


def main():
    # Initialize ROS once up front.
    rclpy.init()
    try:
        world, robot, context, visualization_node = _build_selected_world()
    except Exception as error:
        # Startup failures are persisted because no action result can exist yet.
        message = f"World setup failed: {type(error).__name__}: {error}"
        print(f"[action_server] {message}\n{traceback.format_exc()}", flush=True)
        _atomic_write_json(_result_file(), _error_result("world_setup", message))
        raise SystemExit(1)

    node = rclpy.create_node("thesis_demo_executor")
    plan_executor = PlanExecutor(node, world, robot, context)
    plan_executor.start()

    ros_executor = SingleThreadedExecutor()
    ros_executor.add_node(node)
    if visualization_node is not None:
        ros_executor.add_node(visualization_node)
    print("[action_server] ready", flush=True)
    ros_executor.spin()


if __name__ == "__main__":
    main()
