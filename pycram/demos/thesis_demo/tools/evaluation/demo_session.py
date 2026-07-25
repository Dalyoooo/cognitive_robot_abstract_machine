from copy import deepcopy
from dataclasses import dataclass, field

from pycram.datastructures.dataclasses import Context

from thesis_demo.execution.execution import run_plan_as_result
from thesis_demo.planner.world_context import PlannerNames, build_world_context
from thesis_demo.world.nlp_demo import start_visualization, build_world_model


def _copied_robot_view(world, pristine_robot):
    robot_type = type(pristine_robot)
    matches = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(robot_type)
        if type(annotation) is robot_type
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {robot_type.__name__} annotation in the "
            f"copied world, found {len(matches)}"
        )
    return matches[0]


@dataclass
class DemoSession:
    visualize: bool = False
    world: object = None
    robot: object = None
    demo_context: object = None
    visualization_node: object = None
    names: object = None
    pristine: object = None
    pristine_key: tuple = None
    result: dict = field(default=None)

    def setup_world(self, robot, environment):
        self.stop_world()
        if self.pristine_key != (robot, environment):
            self.pristine = build_world_model(robot, environment)
            self.pristine_key = (robot, environment)
        pristine_world, pristine_robot, _pristine_context = self.pristine
        self.world = deepcopy(pristine_world)
        self.robot = _copied_robot_view(self.world, pristine_robot)
        self.demo_context = Context(world=self.world, robot=self.robot)
        self.demo_context.evaluate_conditions = False
        self.visualization_node = (
            start_visualization(self.world) if self.visualize else None
        )
        self.names = PlannerNames.build(self.world)

    def context(self):
        return build_world_context(self.world, self.robot, self.names)

    def execute_plan(self, plan_dict, step_callback=None):
        self.result = run_plan_as_result(
            self.demo_context,
            plan_dict.get("plan", []),
            step_callback=step_callback,
            names=self.names,
        )

    def execution_result(self):
        return self.result

    def stop_world(self):
        if self.visualization_node is not None:
            self.visualization_node.destroy_node()
        self.world = None
        self.robot = None
        self.demo_context = None
        self.visualization_node = None
        self.names = None
        self.result = None
