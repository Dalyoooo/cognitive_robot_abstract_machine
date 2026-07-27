from dataclasses import dataclass, field

import pycram.alternative_motion_mappings.tiago_motion_mapping as tiago_motion_mapping
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import make_node, sequential

from thesis_demo.execution.action_mapper import ActionMapper
from thesis_demo.execution.grounding import GroundingError
from thesis_demo.execution.navigation_conditions import (
    use_costmap_navigation_collision_check,
)
from thesis_demo.execution.observations import (
    _container_state,
    _directional_observation,
    _held_objects,
    _navigation_observation,
    bound_designator,
)
from thesis_demo.validation.schema import (
    DIRECTIONAL_RELATIONS,
    render,
)

__all__ = ["tiago_motion_mapping"]


def _map_step(action_mapper, step, step_index):
    try:
        mapped_action = action_mapper.map(step)
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


def _navigation_is_combined_with_next_action(steps, step_index):
    step = steps[step_index]
    if step.get("action") != "NavigateAction":
        return False

    next_step_index = step_index + 1
    if next_step_index >= len(steps):
        return False

    next_step = steps[next_step_index]
    return next_step.get("action") in (
        "OpenAction",
        "CloseAction",
        "PickUpAction",
        "PlaceAction",
    )


@dataclass
class _ExecutedStep:
    step_index: int
    step: dict
    action_nodes: list = field(default_factory=list)
    combined_with: str = None


def _complete_observations(action_mapper, executed_steps):
    observations = {
        "navigation": [],
        "container_states": {},
        "held_objects": [],
        "directional_relations": [],
    }

    container_actions = {}
    handled_objects = []
    for position, executed_step in enumerate(executed_steps):
        step = executed_step.step
        action_name = step.get("action")

        if executed_step.combined_with is not None:
            navigation_node = executed_steps[position + 1].action_nodes[0]
            observations["navigation"].append(
                _navigation_observation(
                    navigation_node,
                    step.get("location"),
                    executed_step.combined_with,
                )
            )
        elif action_name == "NavigateAction":
            observations["navigation"].append(
                _navigation_observation(
                    executed_step.action_nodes[-1],
                    step.get("location"),
                )
            )

        if step.get("relation") in DIRECTIONAL_RELATIONS and executed_step.action_nodes:
            action_node = executed_step.action_nodes[-1]
            viewpoint = bound_designator(
                action_node
            ).plan_node.execution_data.execution_start_pose
            observations["directional_relations"].append(
                _directional_observation(
                    action_mapper,
                    step,
                    viewpoint,
                    action_node,
                )
            )

        if action_name in {"OpenAction", "CloseAction"}:
            container_description = step.get("object")
            container_actions[render(container_description)] = (
                executed_step.action_nodes[-1]
            )
        if action_name in {"PickUpAction", "PlaceAction", "TransportAction"}:
            handled_objects.append((step.get("object"), executed_step.action_nodes[-1]))

    for container_name, action_node in container_actions.items():
        observations["container_states"][container_name] = _container_state(
            action_mapper,
            action_node,
        )

    observations["held_objects"] = _held_objects(action_mapper, handled_objects)
    return observations


def run_plan(context, steps, step_callback=None):
    action_mapper = ActionMapper(context)
    plan_root = sequential([], context=context)
    executed_steps = []
    unreported_steps = []

    with simulated_robot, use_costmap_navigation_collision_check():
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

            action_nodes = []
            try:
                mapped_actions = _map_step(action_mapper, step, step_index)
                for action in mapped_actions:
                    action_node = make_node(action)
                    plan_root.add_child(action_node)
                    action_node.perform()
                    if bound_designator(action_node) is None:
                        raise RuntimeError(
                            "no candidate succeeded for the underspecified action"
                        )
                    action_nodes.append(action_node)
            except (TimeoutError, GroundingError):
                raise
            except StopIteration as error:
                raise GroundingError(
                    "No reachable pose found while grounding the step",
                    step_index=step_index,
                    action=step.get("action"),
                ) from error
            except Exception as error:
                raise RuntimeError(
                    f"Step {step_index} ({step.get('action')}) failed: "
                    f"{type(error).__name__}: {error}"
                ) from error

            executed_steps.append(_ExecutedStep(step_index, step, action_nodes))
            action_mapper.register_world_state(
                step,
                bound_designator(action_nodes[-1]),
            )

            if step_callback is not None:
                for unreported_step in unreported_steps:
                    step_callback(unreported_step.step_index, unreported_step.step)
                unreported_steps.clear()
                step_callback(step_index, step)

    return _complete_observations(action_mapper, executed_steps)


def run_plan_as_result(context, steps, step_callback=None):
    try:
        observations = run_plan(context, steps, step_callback=step_callback)
    except TimeoutError:
        raise
    except GroundingError as error:
        return {
            "status": "error",
            "phase": "grounding",
            "error": str(error),
            "grounding_error": error.to_dict(),
        }
    except Exception as error:
        return {
            "status": "error",
            "phase": "execution",
            "error": f"{type(error).__name__}: {error}",
            "error_type": type(error).__name__,
        }
    return {"status": "ok", "phase": "execution", "observations": observations}
