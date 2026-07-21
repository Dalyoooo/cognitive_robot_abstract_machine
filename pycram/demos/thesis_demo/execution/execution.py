import pycram.alternative_motion_mappings.tiago_motion_mapping
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

from ..validation.schema import DIRECTIONAL_RELATIONS, VALID_PYCRAM_ACTIONS
from .grounding import Grounding, GroundingError
from .observations import (
    _container_state,
    _directional_observation,
    _held_objects,
    _navigation_succeeded,
)

ACTIONS_WITH_NAVIGATION = {
    "OpenAction",
    "CloseAction",
    "PickUpAction",
    "PlaceAction",
}


class ActionMapper:

    def __init__(self, world, robot, context):
        self.grounding = Grounding(world, robot)
        self.context = context
        arm_count = len(robot.get_arms())
        if arm_count == 1:
            self.arms = [Arms.LEFT]
        elif arm_count == 2:
            self.arms = [Arms.RIGHT, Arms.LEFT]
        else:
            raise GroundingError(
                f"The Binder supports one or two robot arms, found {arm_count}"
            )
        self.arm = self.arms[0]
        self.pending_pickup = None
        self._dispatch = dict(
            NavigateAction=self._navigate,
            PickUpAction=self._pick_up,
            PlaceAction=self._place,
            TransportAction=self._transport,
            OpenAction=self._open,
            CloseAction=self._close,
            ParkArmsAction=self._park,
        )

    def map(self, step):
        """Map one planner step to a grounded pyCRAM action or action list."""
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

    def register_placement(self, step):
        """Register a successfully executed placement in semDT storage."""
        if step["action"] not in ("PlaceAction", "TransportAction"):
            return
        self.grounding.register_placement(
            step["object"],
            step["location"],
            step["relation"],
        )

    def register_world_state(self, step):
        """Synchronize semDT storage after a successfully executed step."""
        if step["action"] == "PickUpAction":
            self.grounding.remove_from_storage(step["object"])
            return
        self.register_placement(step)

    # ----- Action builders -----

    def _transport(self, object_name, location, relation, source):
        if self.pending_pickup is not None:
            pending_name = str(self.pending_pickup.object_designator.name)
            raise GroundingError(
                f"Cannot transport {object_name!r}: "
                f"PickUpAction for {pending_name!r} still needs PlaceAction"
            )
        body = self.grounding.body(object_name, source)
        place_pose = self.grounding.place_pose(location, body, relation)
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
        body = self.grounding.body(object_name, source)
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
        pending_name = self.grounding.body_id(body)
        if pending_name != object_name:
            raise GroundingError(
                f"Cannot place {object_name!r}: "
                f"The preceding PickUpAction selected {pending_name!r}"
            )
        place_poses = self.grounding.place_poses(location, body, relation)
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
        handle = self.grounding.handle(object_name)
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
                target=self.grounding.navigate_pose(label, annotation),
                reachable=False,
                context=self.context,
            ).resolve()
        except StopIteration as error:
            raise GroundingError(
                f"No collision-free navigation pose for {label!r}"
            ) from error
        return NavigateAction(target_location=base_pose)

    @staticmethod
    def _park(_object_name, _location, _relation, _source):
        return ParkArmsAction(arm=Arms.BOTH)

    # ----- helpers -----

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
                grasp = self._front_grasp(arm)
            try:
                base_pose = CostmapLocation(
                    target=target_pose,
                    reachable=True,
                    reachable_arm=arm,
                    context=self.context,
                    grasp_description=grasp,
                ).resolve()
                # A reachable costmap yields a GraspPose that carries its arm.
                return base_pose, base_pose.arm
            except StopIteration as error:
                last_error = error
        raise GroundingError(
            f"No reachable arm for target pose {target_pose!r}: {last_error!r}"
        ) from last_error

    def _front_grasp(self, arm):
        """Build the grasp that pyCRAM OpenAction and CloseAction use."""
        end_effector = ViewManager.get_end_effector_view(arm, self.context.robot)
        return GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            end_effector,
        )


def log_world_stats(world, when):
    try:
        bodies = len(list(world.bodies))
        annotations = len(list(world.semantic_annotations))
        print(
            f"[executor] world @ {when}: bodies={bodies} annotations={annotations}",
            flush=True,
        )
    except Exception as error:
        print(f"[executor] world stats @ {when} failed: {error!r}", flush=True)


def _map_step(mapper, step, step_index):
    try:
        mapped_action = mapper.map(step)
    except GroundingError as error:
        error.attach_step(step_index, step)
        raise
    except StopIteration as error:
        raise GroundingError(
            "No reachable pose found while grounding the step",
            step_index=step_index,
            action=step.get("action"),
        ) from error

    if isinstance(mapped_action, list):
        return mapped_action
    return [mapped_action]


def _navigation_is_redundant(steps, step_index):
    step = steps[step_index]
    if step.get("action") != "NavigateAction":
        return False

    next_step_index = step_index + 1
    if next_step_index >= len(steps):
        return False

    next_step = steps[next_step_index]
    return next_step.get("action") in ACTIONS_WITH_NAVIGATION


def _complete_observations(
    mapper,
    robot,
    steps,
    navigation_checks=None,
    directional_checks=None,
):
    navigation_checks = navigation_checks or []
    directional_checks = directional_checks or []
    observations = {
        "navigation": [],
        "container_states": {},
        "held_objects": [],
        "arms_parked": None,
        "directional_relations": [],
    }

    for check in navigation_checks:
        observations["navigation"].append(
            {
                "location": check["location"],
                "success": _navigation_succeeded(check["action"]),
            }
        )

    for check in directional_checks:
        observations["directional_relations"].append(
            _directional_observation(
                mapper,
                check["step"],
                check["viewpoint"],
            )
        )

    container_names = []
    picked_object_names = []
    for step in steps:
        action = step.get("action")
        if action in {"OpenAction", "CloseAction"}:
            container_name = step.get("object")
            if container_name not in container_names:
                container_names.append(container_name)
        if action == "PickUpAction":
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


def run_plan(world, robot, context, steps, step_callback=None):
    log_world_stats(world, "grounding")
    mapper = ActionMapper(world, robot, context)
    plan_root = sequential([], context=context)
    navigation_checks = []
    directional_checks = []
    pending_navigation_location = None

    with simulated_robot:
        for step_index, step in enumerate(steps):
            if _navigation_is_redundant(steps, step_index):
                pending_navigation_location = step.get("location")
                continue

            viewpoint = None
            if step.get("relation") in DIRECTIONAL_RELATIONS:
                viewpoint = robot.root.global_transform

            actions = _map_step(mapper, step, step_index)
            try:
                for action in actions:
                    action_node = make_node(action)
                    plan_root.add_child(action_node)
                    action_node.perform()
                mapper.register_world_state(step)

                if pending_navigation_location is not None:
                    navigation_checks.append(
                        {
                            "location": pending_navigation_location,
                            "action": actions[0],
                        }
                    )
                    pending_navigation_location = None
                elif step.get("action") == "NavigateAction":
                    navigation_checks.append(
                        {
                            "location": step.get("location"),
                            "action": actions[-1],
                        }
                    )
                if viewpoint is not None:
                    directional_checks.append(
                        {
                            "step": step,
                            "viewpoint": viewpoint,
                        }
                    )
            except StopIteration as error:
                raise RuntimeError(
                    "No reachable pose found during execution"
                ) from error

            if step_callback is not None:
                step_callback(step_index, step)

    return _complete_observations(
        mapper,
        robot,
        steps,
        navigation_checks,
        directional_checks,
    )
