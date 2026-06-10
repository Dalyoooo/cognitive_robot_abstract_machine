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
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bottle,
    Bowl,
    Cereal,
    CounterTop,
    Spoon,
    Cup,
    Table,
    Milk,
    Mug,
    DrinkingContainer,
    Fork,
    Knife,
    Apple,
    Book,
    CoffeeTable,
    Sofa,
    Plate,
    Oven,
    Sink,
    Kitchen,
    LivingRoom,
    Floor,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world import World

_RESOURCES = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
_OBJECTS_DIR = os.path.join(_RESOURCES, "objects")

ROBOTS = {
    "pr2": (PR2, OmniDrive),
    "hsrb": (HSRB, OmniDrive),
    "tiago": (Tiago, DifferentialDrive),
}

ENVIRONMENTS = {
    "apartment": os.path.join(_RESOURCES, "worlds", "apartment.urdf"),
}

_START_POSE = (1.5, 2.5, 0)

ROOMS = [
    (Kitchen, "kitchen", 3.0, 2.5, 6.0, 4.5),
    (LivingRoom, "living_room", 17.0, 2.5, 3.5, 4.5),
]


def _stl(name):
    return STLParser(os.path.join(_OBJECTS_DIR, name)).parse()


def _primitive(name, scale):
    body = Body(name=PrefixedName(name))
    shapes = BoundingBoxCollection.from_event(
        body, scale.to_simple_event().as_composite_set()
    ).as_shapes()
    body.collision = shapes
    body.visual = shapes
    sub = World()
    with sub.modify_world():
        sub.add_kinematic_structure_entity(body)
    return sub


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


def place_objects(world):
    PLACED_ON_COUNTER = [
        ("bowl.stl", Bowl, 2.55, 2.25, 0.983),
        ("Static_MilkPitcher.stl", DrinkingContainer, 2.55, 2.55, 1.010),
        ("jeroen_cup.stl", Mug, 2.62, 2.95, 0.950),
        ("breakfast_cereal.stl", Cereal, 2.95, 2.30, 1.052),
        ("milk.stl", Milk, 2.95, 2.65, 1.038),
        ("Static_CokeBottle.stl", Bottle, 0.30, 3.20, 0.920),
    ]
    PLACED_IN_BODY = [
        ("spoon.stl", Spoon, "cabinet10_drawer_top", -0.05, -0.05, 0.0),
    ]
    PRIMITIVE_IN_BODY = [
        (
            Fork,
            "drawer_fork",
            "cabinet10_drawer_top",
            -0.05,
            0.05,
            0.0,
            0.18,
            0.02,
            0.02,
        ),
        (
            Knife,
            "drawer_knife",
            "cabinet10_drawer_top",
            -0.05,
            0.12,
            0.0,
            0.18,
            0.015,
            0.02,
        ),
        (
            Knife,
            "cabinet_knife",
            "cabinet5_drawer_top",
            0.0,
            0.0,
            0.0,
            0.20,
            0.02,
            0.02,
        ),
    ]
    PRIMITIVE_OBJECTS = [
        (Plate, "plate", 5.00, 3.70, 0.730, 0.18, 0.18, 0.02),
        (Apple, "apple", 4.80, 4.10, 0.760, 0.08, 0.08, 0.08),
        (Fork, "fork", 5.00, 4.10, 0.730, 0.18, 0.02, 0.02),
        (Book, "book", 16.65, 2.78, 0.415, 0.20, 0.15, 0.03),
    ]

    on_counter = [
        (stl, _stl(stl), cls, x, y, z) for stl, cls, x, y, z in PLACED_ON_COUNTER
    ]
    in_body = [
        (stl, _stl(stl), cls, parent, dx, dy, dz)
        for stl, cls, parent, dx, dy, dz in PLACED_IN_BODY
    ]

    object_annotations = []
    with world.modify_world():
        for stl, sub, cls, x, y, z in on_counter:
            world.merge_world_at_pose(
                sub,
                HomogeneousTransformationMatrix.from_xyz_quaternion(
                    x, y, z, reference_frame=world.root
                ),
            )
            object_annotations.append(cls(root=world.get_body_by_name(stl)))

        for stl, sub, cls, parent, dx, dy, dz in in_body:
            world.merge_world(
                sub,
                FixedConnection(
                    parent=world.get_body_by_name(parent),
                    child=sub.root,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        dx, dy, dz
                    ),
                ),
            )
            object_annotations.append(cls(root=world.get_body_by_name(stl)))

        for cls, name, parent, dx, dy, dz, sx, sy, sz in PRIMITIVE_IN_BODY:
            sub = _primitive(name, Scale(sx, sy, sz))
            world.merge_world(
                sub,
                FixedConnection(
                    parent=world.get_body_by_name(parent),
                    child=sub.root,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        dx, dy, dz
                    ),
                ),
            )
            object_annotations.append(cls(root=world.get_body_by_name(name)))

        for cls, name, x, y, z, sx, sy, sz in PRIMITIVE_OBJECTS:
            cls.create_with_new_body_in_world(
                world=world,
                name=PrefixedName(name),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=x, y=y, z=z
                ),
                scale=Scale(sx, sy, sz),
            )

    return object_annotations


def annotate_world(world, object_annotations):
    with world.modify_world():
        WorldReasoner(world).reason()
        world.add_semantic_annotations(object_annotations)

        for surf_name, surf_cls, sample in (
            ("island_countertop", CounterTop, True),
            ("countertop", CounterTop, True),
            ("table_area_main", Table, False),
            ("coffee_table", CoffeeTable, False),
            ("sofa", Sofa, False),
        ):
            try:
                surface = surf_cls(root=world.get_body_by_name(surf_name))
                world.add_semantic_annotation(surface)
                if sample:
                    surface.calculate_supporting_surface()
            except Exception as e:
                print(f"[world] surface {surf_name} skipped: {e}", flush=True)

        for fixture_name, fixture_cls in (("sink", Sink), ("oven", Oven)):
            try:
                world.add_semantic_annotation(
                    fixture_cls(root=world.get_body_by_name(fixture_name))
                )
            except Exception as e:
                print(f"[world] fixture {fixture_name} skipped: {e}", flush=True)

        for room_cls, room_name, cx, cy, w, d in ROOMS:
            floor = Floor.create_with_new_body_in_world(
                name=PrefixedName(f"{room_name}_floor"),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(cx, cy),
                scale=Scale(w, d, 0.02),
            )
            floor.root.collision = ShapeCollection([])
            world.add_semantic_annotation(
                room_cls(floor=floor, name=PrefixedName(room_name))
            )


def build_world(robot_name="pr2", environment="apartment"):
    if robot_name not in ROBOTS:
        raise ValueError(f"Unknown robot {robot_name!r}; choose from {sorted(ROBOTS)}")
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment {environment!r}; choose from {sorted(ENVIRONMENTS)}"
        )

    robot_cls, drive_cls = ROBOTS[robot_name]

    world = URDFParser.from_file(ENVIRONMENTS[environment]).parse()
    robot_world = URDFParser.from_file(robot_cls.get_ros_file_path()).parse()

    with world.modify_world():
        drive = drive_cls.create_with_dofs(
            parent=world.root, child=robot_world.root, world=world
        )
        world.merge_world(robot_world, drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*_START_POSE)

    object_annotations = place_objects(world)

    try:
        rclpy.init()
    except RuntimeError:
        pass

    node = rclpy.create_node("viz_marker")
    _clear_markers(node)
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

    robot = robot_cls.from_world(world)
    context = Context(world=world, robot=robot)

    annotate_world(world, object_annotations)

    context.evaluate_conditions = False
    return world, robot, context, node


if __name__ == "__main__":
    world, robot, context, node = build_world()
    rclpy.spin(node)
