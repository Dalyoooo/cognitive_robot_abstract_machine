import os

import rclpy
from pycram.datastructures.dataclasses import Context
from pycram.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Drawer,
    Handle,
    Spoon,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import FixedConnection


def _stl(name):
    return STLParser(
        os.path.join(os.path.dirname(__file__), "..", "resources", "objects", name)
    ).parse()


world = setup_world()

bowl = _stl("bowl.stl")
spoon = _stl("spoon.stl")
cup = _stl("jeroen_cup.stl")
coke = _stl("Static_CokeBottle.stl")

with world.modify_world():
    # Bowl on the countertop
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 1.0, reference_frame=world.root
        ),
    )
    # Spoon inside cabinet10 top drawer
    world.merge_world(
        spoon,
        FixedConnection(
            parent=world.get_body_by_name("cabinet10_drawer_top"),
            child=spoon.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                -0.05, -0.05, 0
            ),
        ),
    )
    # Cup on the island countertop
    world.merge_world_at_pose(
        cup,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            3.0, 2.5, 1.0, reference_frame=world.root
        ),
    )
    # Coke bottle next to the cup
    world.merge_world_at_pose(
        coke,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            3.2, 2.5, 1.0, reference_frame=world.root
        ),
    )

rclpy.init()
node = rclpy.create_node("viz_marker")
VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2)

with world.modify_world():
    WorldReasoner(world).reason()
    world.add_semantic_annotations(
        [
            Bowl(root=world.get_body_by_name("bowl.stl")),
            Spoon(root=world.get_body_by_name("spoon.stl")),
            Drawer(
                root=world.get_body_by_name("cabinet10_drawer_top"),
                handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
            ),
        ]
    )

context.evaluate_conditions = False

rclpy.spin(node)
