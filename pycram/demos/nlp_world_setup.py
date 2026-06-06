import os
import time

import rclpy
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bottle,
    Bowl,
    CounterTop,
    Cup,
    Drawer,
    Handle,
    Spoon,
    Table,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
)

_RESOURCES = os.path.join(os.path.dirname(__file__), "..", "resources")
_OBJECTS_DIR = os.path.join(_RESOURCES, "objects")
_ROBOTS_DIR = os.path.join(_RESOURCES, "robots")

ROBOTS = {
    "pr2": (
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro",
        PR2,
        OmniDrive,
    ),
    "hsrb": (os.path.join(_ROBOTS_DIR, "hsrb.urdf"), HSRB, OmniDrive),
    "stretch": (
        os.path.join(_ROBOTS_DIR, "stretch_description.urdf"),
        Stretch,
        DifferentialDrive,
    ),
    "tiago": (
        "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf",
        Tiago,
        DifferentialDrive,
    ),
}

ENVIRONMENTS = {
    "apartment": os.path.join(_RESOURCES, "worlds", "apartment.urdf"),
}

_START_POSE = (1.5, 2.5, 0)


def _stl(name):
    return STLParser(os.path.join(_OBJECTS_DIR, name)).parse()


def _clear_markers(node, topic="/semworld/viz_marker"):
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import Marker, MarkerArray

    qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(MarkerArray, topic, qos)
    time.sleep(0.3)
    marker = Marker()
    marker.action = Marker.DELETEALL
    arr = MarkerArray()
    arr.markers.append(marker)
    pub.publish(arr)
    time.sleep(0.2)


def build_world(robot_name="pr2", environment="apartment"):
    if robot_name not in ROBOTS:
        raise ValueError(
            f"Unknown robot {robot_name!r}; choose from {sorted(ROBOTS)}"
        )
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment {environment!r}; choose from {sorted(ENVIRONMENTS)}"
        )

    robot_urdf, robot_cls, drive_cls = ROBOTS[robot_name]

    world = URDFParser.from_file(ENVIRONMENTS[environment]).parse()
    robot_world = URDFParser.from_file(robot_urdf).parse()

    with world.modify_world():
        drive = drive_cls.create_with_dofs(
            parent=world.root, child=robot_world.root, world=world
        )
        world.merge_world(robot_world, drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*_START_POSE)

    bowl = _stl("bowl.stl")
    spoon = _stl("spoon.stl")
    cup = _stl("jeroen_cup.stl")
    coke = _stl("Static_CokeBottle.stl")

    with world.modify_world():
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1.0, reference_frame=world.root
            ),
        )
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
        world.merge_world_at_pose(
            cup,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                3.0, 2.5, 1.0, reference_frame=world.root
            ),
        )
        world.merge_world_at_pose(
            coke,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                3.2, 2.5, 1.0, reference_frame=world.root
            ),
        )

    try:
        rclpy.init()
    except RuntimeError:
        pass

    node = rclpy.create_node("viz_marker")
    _clear_markers(node)
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

    robot = robot_cls.from_world(world)
    context = Context(world=world, robot=robot)

    with world.modify_world():
        WorldReasoner(world).reason()
        world.add_semantic_annotations(
            [
                Bowl(root=world.get_body_by_name("bowl.stl")),
                Spoon(root=world.get_body_by_name("spoon.stl")),
                Cup(root=world.get_body_by_name("jeroen_cup.stl")),
                Bottle(root=world.get_body_by_name("Static_CokeBottle.stl")),
                Drawer(
                    root=world.get_body_by_name("cabinet10_drawer_top"),
                    handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
                ),
            ]
        )

        for surf_name, surf_cls in (
            ("island_countertop", CounterTop),
            ("countertop", CounterTop),
            ("table_area_main", Table),
        ):
            try:
                surface = surf_cls(root=world.get_body_by_name(surf_name))
                world.add_semantic_annotation(surface)
                surface.calculate_supporting_surface()
            except Exception as e:
                print(f"[world] surface {surf_name} skipped: {e}", flush=True)

    context.evaluate_conditions = False
    return world, robot, context, node


if __name__ == "__main__":
    world, robot, context, node = build_world()
    rclpy.spin(node)
