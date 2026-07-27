from pycram.datastructures.enums import TaskStatus
from pycram.plans.plan_node import DesignatorNode, UnderspecifiedNode
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.view_manager import ViewManager
from semantic_digital_twin.reasoning.predicates import allclose

from thesis_demo.execution.grounding import directional_relation_holds
from thesis_demo.validation.schema import render

NAVIGATION_ARRIVAL_TOLERANCE = 0.03
CONTAINER_OPEN_MIN_FRACTION = 0.75
CONTAINER_CLOSED_MAX_FRACTION = 0.25


def bound_designator(node):
    if not isinstance(node, UnderspecifiedNode):
        return node.designator
    for child in node.children:
        if child.status == TaskStatus.SUCCEEDED:
            return child.designator
    return None


def _navigation_observation(navigation_node, location, combined_with=None):
    designator = bound_designator(navigation_node)
    if not isinstance(designator, NavigateAction):
        nested_navigation_nodes = [
            descendant
            for descendant in designator.plan_node.descendants
            if isinstance(descendant, DesignatorNode)
            and isinstance(descendant.designator, NavigateAction)
        ]
        if not nested_navigation_nodes:
            raise RuntimeError("Executed action contains no navigation")
        navigation_node = nested_navigation_nodes[0]

    observation = {
        "location": render(location),
        "success": bool(
            allclose(
                navigation_node.execution_data.execution_end_pose,
                navigation_node.designator.target_location,
                atol=NAVIGATION_ARRIVAL_TOLERANCE,
            )
        ),
    }
    if combined_with is not None:
        observation["combined_with"] = combined_with
    return observation


def _directional_observation(action_mapper, step, viewpoint, action_node):
    object_body = bound_designator(action_node).object_designator
    reference_body = action_mapper.grounding.resolve_body(step["location"])
    relation_holds = directional_relation_holds(
        object_body.global_pose.position,
        reference_body.global_pose.position,
        viewpoint,
        step["relation"],
    )
    return {
        "object": render(step["object"]),
        "location": render(step["location"]),
        "relation": step["relation"],
        "success": bool(relation_holds),
    }


def _container_state(action_mapper, action_node):
    handle = bound_designator(action_node).object_designator
    opening_mechanism = action_mapper.grounding.opening_mechanism_for_handle(handle)
    position_fraction = opening_mechanism.position_fraction()
    if position_fraction is None:
        return "unknown"
    if position_fraction > CONTAINER_OPEN_MIN_FRACTION:
        return "open"
    if position_fraction < CONTAINER_CLOSED_MAX_FRACTION:
        return "closed"
    return "unknown"


def _held_objects(action_mapper, handled_objects):
    tool_frames = [
        ViewManager.get_end_effector_view(
            arm,
            action_mapper.context.robot,
        ).tool_frame
        for arm in action_mapper.arms
    ]

    held_objects = []
    for object_description, action_node in handled_objects:
        object_body = bound_designator(action_node).object_designator
        if any(
            object_body
            in action_mapper.grounding.world.get_kinematic_structure_entities_of_branch(
                tool_frame
            )
            for tool_frame in tool_frames
        ):
            rendered_description = render(object_description)
            if rendered_description not in held_objects:
                held_objects.append(rendered_description)
    return held_objects
