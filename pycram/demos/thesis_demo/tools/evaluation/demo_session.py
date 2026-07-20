from dataclasses import dataclass, field

from ...execution.execution import run_plan
from ...execution.grounding import GroundingError
from ...planner.world_context import classify_world
from ...world.nlp_demo import build_world


@dataclass
class DemoSession:
    """Drive the demo in this process, replacing the binder's file handoff.

    Exposes the call surface the benchmark already used (setup_world,
    is_world_ready, execute_plan, execution_result, stop_world) so the
    pipeline stages stay unchanged, and returns the same result schema the
    executor produced.
    """

    visualize: bool = False
    world: object = None
    robot: object = None
    demo_context: object = None
    visualization_node: object = None
    result: dict = field(default=None)

    def setup_world(self, robot, environment):
        """Build a fresh world for one case."""
        self.stop_world()
        (
            self.world,
            self.robot,
            self.demo_context,
            self.visualization_node,
        ) = build_world(
            robot_name=robot, environment=environment, visualize=self.visualize
        )

    def is_world_ready(self):
        """Return whether a world is currently built."""
        return self.world is not None

    def context(self):
        """Project the live world into the planner context."""
        return classify_world(self.world, self.robot)

    def execute_plan(self, plan_dict):
        """Run one validated plan and record the executor-shaped result."""
        steps = plan_dict.get("plan", [])
        try:
            observations = run_plan(
                self.world, self.robot, self.demo_context, steps
            )
        except GroundingError as error:
            self.result = {
                "status": "error",
                "phase": "grounding",
                "error": str(error),
                "grounding_error": error.to_dict(),
            }
        except Exception as error:
            self.result = {
                "status": "error",
                "phase": "execution",
                "error": f"{type(error).__name__}: {error}",
                "error_type": type(error).__name__,
            }
        else:
            self.result = {
                "status": "ok",
                "phase": "execution",
                "observations": observations,
            }

    def execution_result(self):
        """Return the most recent execution result."""
        return self.result

    def stop_world(self):
        """Drop the current world so the next case starts clean."""
        self.world = None
        self.robot = None
        self.demo_context = None
        self.visualization_node = None
        self.result = None
