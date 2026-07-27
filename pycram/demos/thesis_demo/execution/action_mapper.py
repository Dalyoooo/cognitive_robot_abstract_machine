from dataclasses import dataclass, field

from krrood.entity_query_language.factories import underspecified, variable
from pycram.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.locations.locations import CostmapLocation
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.container import CloseAction, OpenAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from pycram.view_manager import ViewManager
from semantic_digital_twin.world_description.world_entity import Body

from thesis_demo.execution.grounding import Grounding, GroundingError
from thesis_demo.validation.schema import VALID_PYCRAM_ACTIONS


@dataclass
class NavigateAndOperateContainerAction(ActionDescription):
    navigation_target: object
    object_designator: Body
    arm: object
    container_action_class: object

    def execute(self):
        self.add_subplan(
            sequential(
                [
                    NavigateAction(target_location=self.navigation_target),
                    self.container_action_class(
                        object_designator=self.object_designator,
                        arm=self.arm,
                    ),
                    # Pulling a container open leaves the arm stretched over it,
                    # and no stance lets that configuration reach into it
                    # afterwards. Parking here keeps the next grasp possible.
                    ParkArmsAction(arm=Arms.BOTH),
                ]
            )
        ).perform()


@dataclass
class ActionMapper:
    context: object
    grounding: Grounding = field(init=False)
    arms: list = field(init=False)
    arm: object = field(init=False)
    pending_pickup: object = field(init=False, default=None)
    action_handlers: dict = field(init=False)

    def __post_init__(self):
        self.grounding = Grounding(self.context.world, self.context.robot)
        number_of_arms = len(self.context.robot.get_arms())
        if number_of_arms == 1:
            self.arms = [Arms.LEFT]
        elif number_of_arms == 2:
            self.arms = [Arms.RIGHT, Arms.LEFT]
        else:
            raise GroundingError(
                f"The Binder supports one or two robot arms, found {number_of_arms}"
            )
        self.arm = self.arms[0]
        self.action_handlers = {
            "NavigateAction": self._navigate,
            "PickUpAction": self._pick_up,
            "PlaceAction": self._place,
            "TransportAction": self._transport,
            "OpenAction": self._open,
            "CloseAction": self._close,
            "ParkArmsAction": self._park,
        }
        if set(self.action_handlers) != VALID_PYCRAM_ACTIONS:
            raise RuntimeError(
                "Action dispatch does not match the schema's action specs"
            )

    def map(self, step):
        action_name = step["action"]
        action_handler = self.action_handlers.get(action_name)
        if action_handler is None:
            raise ValueError(
                f"unsupported action {action_name!r}; "
                f"valid: {', '.join(sorted(VALID_PYCRAM_ACTIONS))}"
            )
        return action_handler(
            step.get("object"),
            step.get("location"),
            step.get("relation"),
            step.get("source"),
        )

    def register_world_state(self, step, executed_action):
        action_name = step["action"]
        if action_name in ("OpenAction", "CloseAction"):
            self.arm = executed_action.arm
            return
        if action_name == "PickUpAction":
            self.grounding.clear_storage_memberships(step["object"])
            self.pending_pickup = executed_action
            return
        if action_name not in ("PlaceAction", "TransportAction"):
            return
        self.grounding.record_placement(
            step["object"],
            step["location"],
            step["relation"],
            executed_action.target_location,
        )
        if action_name == "PlaceAction":
            self.pending_pickup = None

    def _require_empty_gripper(self, action_verb, object_name):
        if self.pending_pickup is None:
            return
        pending_name = str(self.pending_pickup.object_designator.name)
        raise GroundingError(
            f"Cannot {action_verb} {object_name!r}: "
            f"PickUpAction for {pending_name!r} still needs PlaceAction"
        )

    def _transport(self, object_name, location, relation, source):
        self._require_empty_gripper("transport", object_name)
        object_body = self.grounding.resolve_body(object_name, source)
        placement_pose = self.grounding.placement_poses(
            location, object_body, relation
        )[0]
        return underspecified(TransportAction)(
            object_designator=variable(Body, domain=[object_body]),
            target_location=placement_pose,
            arm=self.arm,
        )

    def _pick_up(self, object_name, _location, _relation, source):
        self._require_empty_gripper("pick up", object_name)
        object_body = self.grounding.resolve_body(object_name, source)
        pickup_base_pose = self._find_reachable_base_pose(
            object_body.global_pose,
            excluded_arms=self._arms_holding_objects(),
        )
        arm = pickup_base_pose.arm
        self.arm = arm
        pickup_match = underspecified(PickUpAction)(
            object_designator=variable(Body, domain=[object_body]),
            arm=arm,
            grasp_description=pickup_base_pose.grasp_description,
        )
        return [
            NavigateAction(target_location=pickup_base_pose),
            pickup_match,
        ]

    def _place(self, object_name, location, relation, _source):
        if self.pending_pickup is None:
            raise GroundingError(
                f"Cannot place {object_name!r}: No preceding PickUpAction in this plan"
            )
        pickup_action = self.pending_pickup
        object_body = pickup_action.object_designator
        arm = pickup_action.arm
        grasp_description = pickup_action.grasp_description
        if self.grounding.resolve_body(object_name) is not object_body:
            raise GroundingError(
                f"Cannot place {object_name!r}: "
                "the preceding PickUpAction picked up something else"
            )
        placement_poses = self.grounding.placement_poses(
            location, object_body, relation
        )
        last_reachability_error = None
        for placement_pose in placement_poses:
            try:
                base_pose = self._find_reachable_base_pose(
                    placement_pose,
                    excluded_arms=set(self.arms) - {arm},
                    grasp_description=grasp_description,
                )
            except GroundingError as error:
                last_reachability_error = error
                continue

            self.arm = arm
            return [
                NavigateAction(target_location=base_pose),
                PlaceAction(
                    object_designator=object_body,
                    target_location=placement_pose,
                    arm=arm,
                ),
            ]

        raise GroundingError(
            f"Cannot place {object_name!r} at {location!r}: "
            "No reachable placement pose found"
        ) from last_reachability_error

    def _open(self, object_name, _location, _relation, _source):
        return self._container_action(OpenAction, object_name)

    def _close(self, object_name, _location, _relation, _source):
        return self._container_action(CloseAction, object_name)

    def _container_action(self, container_action_class, object_name):
        candidate_handles = self.grounding.candidate_handles(object_name)
        handle_navigation_targets = []
        for handle in candidate_handles:
            try:
                navigation_target = self._find_reachable_base_pose(
                    handle.global_pose,
                    excluded_arms=self._arms_holding_objects(),
                    front_grasp=True,
                )
            except GroundingError:
                continue
            handle_navigation_targets.append((handle, navigation_target))

        if not handle_navigation_targets:
            raise GroundingError(
                f"Cannot reach {object_name!r}: no reachable handle found"
            )
        self.arm = handle_navigation_targets[0][1].arm

        def make_container_action(object_designator):
            navigation_target = next(
                candidate_navigation_target
                for handle, candidate_navigation_target in handle_navigation_targets
                if handle is object_designator
            )
            return NavigateAndOperateContainerAction(
                navigation_target=navigation_target,
                object_designator=object_designator,
                arm=navigation_target.arm,
                container_action_class=container_action_class,
            )

        return underspecified(
            make_container_action,
            target_type=NavigateAndOperateContainerAction,
        )(
            object_designator=variable(
                Body,
                domain=[
                    handle for handle, _navigation_target in handle_navigation_targets
                ],
            ),
        )

    def _navigate(self, _object_name, location, _relation, _source):
        target_annotation = self.grounding.resolve_annotation(location)
        if target_annotation is None:
            raise GroundingError(f"Cannot navigate to {location!r}: Not found in world")

        try:
            base_pose = CostmapLocation(
                target=self.grounding.navigation_pose(location, target_annotation),
                reachable=False,
                context=self.context,
            ).ground()
        except StopIteration as error:
            raise GroundingError(
                f"No collision-free navigation pose for {location!r}"
            ) from error
        return NavigateAction(target_location=base_pose)

    @staticmethod
    def _park(_object_name, _location, _relation, _source):
        return ParkArmsAction(arm=Arms.BOTH)

    def _arms_holding_objects(self):
        if self.pending_pickup is None:
            return set()
        return {self.pending_pickup.arm}

    def _find_reachable_base_pose(
        self,
        target_pose,
        excluded_arms=None,
        grasp_description=None,
        front_grasp=False,
    ):
        excluded_arms = excluded_arms or set()
        available_arms = []
        if self.arm not in excluded_arms:
            available_arms.append(self.arm)

        for arm in self.arms:
            if arm == self.arm or arm in excluded_arms:
                continue
            available_arms.append(arm)

        if not available_arms:
            raise GroundingError("No free arm available for the requested action")

        last_reachability_error = None
        for arm in available_arms:
            selected_grasp_description = grasp_description
            if front_grasp:
                end_effector = ViewManager.get_end_effector_view(
                    arm,
                    self.context.robot,
                )
                selected_grasp_description = GraspDescription(
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
                    grasp_description=selected_grasp_description,
                ).ground()
                return base_pose
            except StopIteration as error:
                last_reachability_error = error
        raise GroundingError(
            "No reachable arm for target pose "
            f"{target_pose!r}: {last_reachability_error!r}"
        ) from last_reachability_error
