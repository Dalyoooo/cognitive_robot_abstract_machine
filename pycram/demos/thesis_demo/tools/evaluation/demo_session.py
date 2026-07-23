from dataclasses import dataclass, field

from thesis_demo.execution.execution import run_plan
from thesis_demo.execution.grounding import GroundingError
from thesis_demo.planner.world_context import classify_world
from thesis_demo.world.nlp_demo import build_world


@dataclass
class DemoSession:
    visualize: bool = False
    world: object = None
    robot: object = None
    demo_context: object = None
    visualization_node: object = None
    result: dict = field(default=None)

    def setup_world(self, robot, environment):
        self.stop_world()
        (
            self.world,
            self.robot,
            self.demo_context,
            self.visualization_node,
        ) = build_world(
            robot_name=robot, environment=environment, visualize=self.visualize
        )

    def context(self):
        return classify_world(self.world, self.robot)

    def execute_plan(self, plan_dict, step_callback=None):
        steps = plan_dict.get("plan", [])
        try:
            observations = run_plan(
                self.world,
                self.robot,
                self.demo_context,
                steps,
                step_callback=step_callback,
            )
        except TimeoutError:
            raise
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
        return self.result

    def stop_world(self):
        self.world = None
        self.robot = None
        self.demo_context = None
        self.visualization_node = None
        self.result = None
