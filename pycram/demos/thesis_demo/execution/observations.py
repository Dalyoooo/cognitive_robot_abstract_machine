from pycram.exceptions import ConditionNotSatisfied
from pycram.view_manager import ViewManager
from semantic_digital_twin.reasoning.predicates import allclose

from thesis_demo.execution.grounding import directional_relation_holds

NAVIGATION_ARRIVAL_TOLERANCE = 0.03
CONTAINER_OPEN_MIN_FRACTION = 0.75
CONTAINER_CLOSED_MAX_FRACTION = 0.25


def _post_condition_ok(action):
    try:
        return bool(action.evaluate_post_condition())
    except ConditionNotSatisfied:
        return False


def _navigation_succeeded(navigation_node):
    execution_data = navigation_node.execution_data
    return bool(
        allclose(
            execution_data.execution_end_pose,
            navigation_node.action.target_location,
            atol=NAVIGATION_ARRIVAL_TOLERANCE,
        )
    )


def _navigation_observation(navigation_node, location, combined_with=None):
    observation = {
        "location": location,
        "success": _navigation_succeeded(navigation_node),
    }
    if combined_with is not None:
        observation["combined_with"] = combined_with
    return observation


def _directional_observation(mapper, step, viewpoint):
    object_body = mapper.grounding.resolve_body(step["object"])
    reference_body = mapper.grounding.resolve_body(step["location"])
    relation_holds = directional_relation_holds(
        object_body.global_pose.position,
        reference_body.global_pose.position,
        viewpoint,
        step["relation"],
    )
    return {
        "object": step["object"],
        "location": step["location"],
        "relation": step["relation"],
        "success": bool(relation_holds),
    }


def _container_state(mapper, container_name):
    fraction = mapper.grounding.resolve_opening_mechanism(
        container_name
    ).position_fraction()
    if fraction is None:
        return "unknown"
    if fraction > CONTAINER_OPEN_MIN_FRACTION:
        return "open"
    if fraction < CONTAINER_CLOSED_MAX_FRACTION:
        return "closed"
    return "unknown"


def _is_below(world, body, possible_parent):
    return body in world.get_kinematic_structure_entities_of_branch(possible_parent)


def _held_objects(mapper, object_names):
    tool_frames = []
    for arm in mapper.arms:
        end_effector = ViewManager.get_end_effector_view(arm, mapper.context.robot)
        tool_frames.append(end_effector.tool_frame)

    held_objects = []
    for object_name in object_names:
        body = mapper.grounding.resolve_body(object_name)
        if any(
            _is_below(mapper.grounding.world, body, tool_frame)
            for tool_frame in tool_frames
        ):
            held_objects.append(object_name)
    return held_objects
