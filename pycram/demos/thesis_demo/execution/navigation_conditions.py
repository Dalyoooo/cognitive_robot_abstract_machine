from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pycram.robot_plans.actions.core.navigation as navigation_actions
from krrood.entity_query_language.predicate import symbolic_function
from pycram.pose_validator import collision_check


@symbolic_function
def is_navigation_pose_collision_free(robot, target_pose):
    copied_world = deepcopy(robot._world)
    copied_robot = copied_world.get_semantic_annotation_by_id(robot.id)
    with copied_world.modify_world():
        copied_robot.root.parent_connection.origin = target_pose
    return not collision_check(copied_robot, copied_world)


@contextmanager
def use_costmap_navigation_collision_check():
    with patch.object(
        navigation_actions,
        "is_pose_free_for_robot",
        is_navigation_pose_collision_free,
    ):
        yield
