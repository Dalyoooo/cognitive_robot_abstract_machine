from pycram.exceptions import ConditionNotSatisfied
from pycram.view_manager import ViewManager
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from .grounding import directional_relation_holds


def _navigation_succeeded(action):
    try:
        return bool(action.evaluate_post_condition())
    except ConditionNotSatisfied:
        return False


def _directional_observation(mapper, step, viewpoint):
    object_body = mapper.grounding.body(step["object"])
    reference_body = mapper.grounding.body(step["location"])
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
    handle = mapper.grounding.handle(container_name)
    try:
        connection = handle.get_first_parent_connection_of_type(ActiveConnection1DOF)
    except (AttributeError, ValueError):
        # semDT raises instead of returning None when the chain has no match.
        return "unknown"
    lower = connection.dof.limits.lower.position
    upper = connection.dof.limits.upper.position
    if lower is None or upper is None or upper <= lower:
        return "unknown"
    # Fractions of the connection's own motion range, so doors and drawers
    # share one rule (absolute thresholds were wrong for rotational doors).
    fraction = (connection.position - lower) / (upper - lower)
    if fraction > 0.75:
        return "open"
    if fraction < 0.25:
        return "closed"
    return "unknown"


def _is_below(world, body, possible_parent):
    return body in world.get_kinematic_structure_entities_of_branch(possible_parent)


def _held_objects(mapper, robot, object_names):
    tool_frames = []
    for arm in mapper.arms:
        end_effector = ViewManager.get_end_effector_view(arm, robot)
        tool_frames.append(end_effector.tool_frame)

    held_objects = []
    for object_name in object_names:
        body = mapper.grounding.body(object_name)
        if any(
            _is_below(mapper.grounding.world, body, tool_frame)
            for tool_frame in tool_frames
        ):
            held_objects.append(object_name)
    return held_objects
