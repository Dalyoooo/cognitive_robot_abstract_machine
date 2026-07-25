from dataclasses import dataclass, field

import pycram.alternative_motion_mappings.tiago_motion_mapping as _tiago_motion_mapping  # noqa: F401
from pycram.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import make_node, sequential
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.container import CloseAction, OpenAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from pycram.view_manager import ViewManager

from thesis_demo.execution.grounding import Grounding, GroundingError
from thesis_demo.execution.observations import (
    _container_state,
    _directional_observation,
    _held_objects,
    _navigation_observation,
    _post_condition_ok,
)
from thesis_demo.validation.schema import DIRECTIONAL_RELATIONS, VALID_PYCRAM_ACTIONS

ACTIONS_WITH_INTERNAL_NAVIGATION = {
    "OpenAction",
    "CloseAction",
    "PickUpAction",
    "PlaceAction",
}


@dataclass
class ActionMapper:
    world: object
    robot: object
    context: object
    names: object = None
    grounding: Grounding = field(init=False)
    arms: list = field(init=False)
    arm: object = field(init=False)
    pending_pickup: object = field(init=False, default=None)
    _dispatch: dict = field(init=False)

    def __post_init__(self):
        self.grounding = Grounding(self.world, self.robot, names=self.names)
        arm_count = len(self.robot.get_arms())
        if arm_count == 1:
            self.arms = [Arms.LEFT]
        elif arm_count == 2:
            self.arms = [Arms.RIGHT, Arms.LEFT]
        else:
            raise GroundingError(
                f"The Binder supports one or two robot arms, found {arm_count}"
            )
        self.arm = self.arms[0]
        self._dispatch = {
            "NavigateAction": self._navigate,
            "PickUpAction": self._pick_up,
            "PlaceAction": self._place,
            "TransportAction": self._transport,
            "OpenAction": self._open,
            "CloseAction": self._close,
            "ParkArmsAction": self._park,
        }
        if set(self._dispatch) != VALID_PYCRAM_ACTIONS:
            raise RuntimeError(
                "Action dispatch does not match the schema's action specs"
            )

    def map(self, step):
        action = step["action"]
        handler = self._dispatch.get(action)
        if handler is None:
            raise ValueError(
                f"unsupported action {action!r}; "
                f"valid: {', '.join(sorted(VALID_PYCRAM_ACTIONS))}"
            )
        return handler(
            step.get("object"),
            step.get("location"),
            step.get("relation"),
            step.get("source"),
        )

    def register_world_state(self, step, executed_action):
        action_name = step["action"]
        if action_name == "PickUpAction":
            self.grounding.clear_storage_memberships(step["object"])
            return
        if action_name not in ("PlaceAction", "TransportAction"):
            return
        self.grounding.record_placement(
            step["object"],
            step["location"],
            step["relation"],
            executed_action.target_location,
        )

    def _transport(self, object_name, location, relation, source):
        if self.pending_pickup is not None:
            pending_name = str(self.pending_pickup.object_designator.name)
            raise GroundingError(
                f"Cannot transport {object_name!r}: "
                f"PickUpAction for {pending_name!r} still needs PlaceAction"
            )
        body = self.grounding.resolve_body(object_name, source)
        place_pose = self.grounding.placement_pose(location, body, relation)
        return TransportAction(
            object_designator=body,
            target_location=place_pose,
            arm=self.arm,
        )

    def _pick_up(self, object_name, _location, _relation, source):
        if self.pending_pickup is not None:
            pending_name = str(self.pending_pickup.object_designator.name)
            raise GroundingError(
                f"Cannot pick up {object_name!r}: "
                f"PickUpAction for {pending_name!r} still needs PlaceAction"
            )
        body = self.grounding.resolve_body(object_name, source)
        pickup_pose, arm = self._resolve_reachable(
            body.global_pose,
            excluded_arms=self._occupied_arms(),
        )
        self.arm = arm
        pickup_action = PickUpAction(
            object_designator=body,
            arm=arm,
            grasp_description=pickup_pose.grasp_description,
        )
        self.pending_pickup = pickup_action
        return [
            NavigateAction(target_location=pickup_pose),
            pickup_action,
        ]

    def _place(self, object_name, location, relation, _source):
        if self.pending_pickup is None:
            raise GroundingError(
                f"Cannot place {object_name!r}: No preceding PickUpAction in this plan"
            )
        pickup_action = self.pending_pickup
        body = pickup_action.object_designator
        arm = pickup_action.arm
        grasp_description = pickup_action.grasp_description
        pending_name = self.grounding.planner_name_for(body)
        if pending_name != object_name:
            raise GroundingError(
                f"Cannot place {object_name!r}: "
                f"The preceding PickUpAction selected {pending_name!r}"
            )
        place_poses = self.grounding.placement_poses(location, body, relation)
        last_error = None
        for place_pose in place_poses:
            try:
                # Only the arm holding the object can place it, using the pickup grasp.
                base_pose, _arm = self._resolve_reachable(
                    place_pose,
                    excluded_arms=set(self.arms) - {arm},
                    grasp_description=grasp_description,
                )
            except GroundingError as error:
                # A surface provides several samples. An unreachable sample does
                # not invalidate the remaining pyCRAM candidates.
                last_error = error
                continue

            self.arm = arm
            self.pending_pickup = None
            return [
                NavigateAction(target_location=base_pose),
                PlaceAction(
                    object_designator=body,
                    target_location=place_pose,
                    arm=arm,
                ),
            ]

        raise GroundingError(
            f"Cannot place {object_name!r} at {location!r}: "
            "No reachable placement pose found"
        ) from last_error

    def _open(self, object_name, _location, _relation, _source):
        return self._container_action(OpenAction, object_name)

    def _close(self, object_name, _location, _relation, _source):
        return self._container_action(CloseAction, object_name)

    def _container_action(self, action_type, object_name):
        # pyCRAM's AccessingLocation rotates its targets about the world origin,
        # which puts the goal underground for furniture at negative x. A front
        # grasp on the handle reaches the same poses without that.
        handle = self.grounding.resolve_handle(object_name)
        base_pose, arm = self._resolve_reachable(
            handle.global_pose,
            excluded_arms=self._occupied_arms(),
            front_grasp=True,
        )
        self.arm = arm
        return [
            NavigateAction(target_location=base_pose),
            action_type(object_designator=handle, arm=arm),
        ]

    def _navigate(self, _object_name, location, _relation, _source):
        label = location
        annotation = self.grounding.resolve_annotation(label)
        if annotation is None:
            raise GroundingError(f"Cannot navigate to {label!r}: Not found in world")

        try:
            base_pose = CostmapLocation(
                target=self.grounding.navigation_pose(label, annotation),
                reachable=False,
                context=self.context,
            ).ground()
        except StopIteration as error:
            # CostmapLocation signals an empty candidate generator this way.
            raise GroundingError(
                f"No collision-free navigation pose for {label!r}"
            ) from error
        return NavigateAction(target_location=base_pose)

    @staticmethod
    def _park(_object_name, _location, _relation, _source):
        return ParkArmsAction(arm=Arms.BOTH)

    def _occupied_arms(self):
        if self.pending_pickup is None:
            return set()
        return {self.pending_pickup.arm}

    def _resolve_reachable(
        self,
        target_pose,
        excluded_arms=None,
        grasp_description=None,
        front_grasp=False,
    ):
        excluded_arms = excluded_arms or set()
        arms_to_try = []
        if self.arm not in excluded_arms:
            arms_to_try.append(self.arm)

        for arm in self.arms:
            if arm == self.arm or arm in excluded_arms:
                continue
            arms_to_try.append(arm)

        if not arms_to_try:
            raise GroundingError("No free arm available for the requested action")

        last_error = None
        for arm in arms_to_try:
            grasp = grasp_description
            if front_grasp:
                end_effector = ViewManager.get_end_effector_view(
                    arm,
                    self.context.robot,
                )
                grasp = GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    end_effector,
                )
            try:
                base_pose = CostmapLocation(
                    target=target_pose,
                    reachable=True,
                    reachable_arm=arm,
                    context=self.context,
                    grasp_description=grasp,
                ).ground()
                return base_pose, base_pose.arm
            except StopIteration as error:
                last_error = error
        raise GroundingError(
            f"No reachable arm for target pose {target_pose!r}: {last_error!r}"
        ) from last_error


def _map_step(mapper, step, step_index):
    try:
        mapped_action = mapper.map(step)
    except GroundingError as error:
        error.attach_step(step_index, step)
        raise
    except StopIteration as error:
        # Keep pyCRAM's exhausted location iterator out of the planner API.
        raise GroundingError(
            "No reachable pose found while grounding the step",
            step_index=step_index,
            action=step.get("action"),
        ) from error

    if isinstance(mapped_action, list):
        return mapped_action
    return [mapped_action]


def _navigation_is_combined_with_next_action(steps, step_index):
    step = steps[step_index]
    if step.get("action") != "NavigateAction":
        return False

    next_step_index = step_index + 1
    if next_step_index >= len(steps):
        return False

    next_step = steps[next_step_index]
    return next_step.get("action") in ACTIONS_WITH_INTERNAL_NAVIGATION


@dataclass
class _ExecutedStep:
    step_index: int
    step: dict
    nodes: list = field(default_factory=list)
    conditions: list = field(default_factory=list)
    combined_with: str = None


def _complete_observations(mapper, robot, executed_steps):
    observations = {
        "navigation": [],
        "container_states": {},
        "held_objects": [],
        "directional_relations": [],
        "steps": [],
    }

    container_names = []
    picked_object_names = []
    for position, executed in enumerate(executed_steps):
        step = executed.step
        action_name = step.get("action")

        if executed.conditions:
            observations["steps"].append(
                {
                    "action": action_name,
                    "object": step.get("object"),
                    "condition_ok": all(executed.conditions),
                }
            )
        if executed.combined_with is not None:
            # The combined navigation runs as the first action of the next
            # step; its recorded end pose is the navigation observation.
            navigation_node = executed_steps[position + 1].nodes[0]
            observations["navigation"].append(
                _navigation_observation(
                    navigation_node, step.get("location"), executed.combined_with
                )
            )
        elif action_name == "NavigateAction":
            observations["navigation"].append(
                _navigation_observation(executed.nodes[-1], step.get("location"))
            )

        if step.get("relation") in DIRECTIONAL_RELATIONS and executed.nodes:
            viewpoint = executed.nodes[0].execution_data.execution_start_pose
            observations["directional_relations"].append(
                _directional_observation(mapper, step, viewpoint)
            )

        if action_name in {"OpenAction", "CloseAction"}:
            container_name = step.get("object")
            if container_name not in container_names:
                container_names.append(container_name)
        if action_name == "PickUpAction":
            object_name = step.get("object")
            if object_name not in picked_object_names:
                picked_object_names.append(object_name)

    for container_name in container_names:
        observations["container_states"][container_name] = _container_state(
            mapper,
            container_name,
        )

    observations["held_objects"] = _held_objects(
        mapper,
        robot,
        picked_object_names,
    )
    return observations


def run_plan(world, robot, context, steps, step_callback=None, names=None):
    mapper = ActionMapper(world, robot, context, names=names)
    plan_root = sequential([], context=context)
    executed_steps = []
    unreported_steps = []

    with simulated_robot:
        for step_index, step in enumerate(steps):
            if _navigation_is_combined_with_next_action(steps, step_index):
                executed_step = _ExecutedStep(
                    step_index=step_index,
                    step=step,
                    combined_with=steps[step_index + 1]["action"],
                )
                executed_steps.append(executed_step)
                unreported_steps.append(executed_step)
                continue

            nodes = []
            conditions = []
            try:
                actions = _map_step(mapper, step, step_index)
                for action in actions:
                    action_node = make_node(action)
                    plan_root.add_child(action_node)
                    action_node.perform()
                    nodes.append(action_node)
                    conditions.append(_post_condition_ok(action))
            except (TimeoutError, GroundingError):
                # Callers report a grounding failure as its own phase.
                raise
            except StopIteration as error:
                raise RuntimeError(
                    f"Step {step_index} ({step.get('action')}): "
                    "No reachable pose found during execution"
                ) from error
            except Exception as error:
                raise RuntimeError(
                    f"Step {step_index} ({step.get('action')}) failed: "
                    f"{type(error).__name__}: {error}"
                ) from error

            executed_steps.append(_ExecutedStep(step_index, step, nodes, conditions))
            mapper.register_world_state(step, actions[-1])

            if step_callback is not None:
                for unreported_step in unreported_steps:
                    step_callback(unreported_step.step_index, unreported_step.step)
                unreported_steps.clear()
                step_callback(step_index, step)

    return _complete_observations(mapper, robot, executed_steps)
