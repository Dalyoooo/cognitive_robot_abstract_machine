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
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bottle,
    Bowl,
    Cereal,
    CounterTop,
    Dishwasher,
    Spoon,
    Cup,
    Table,
    Milk,
    Mug,
    DrinkingContainer,
    Fork,
    Fridge,
    Knife,
    Kettle,
    MustardBottle,
    SoapBottle,
    WineBottle,
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
    Point3,
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
    "stretch": (Stretch, DifferentialDrive),
    "tiago": (Tiago, DifferentialDrive),
    "g1": (UnitreeG1, OmniDrive),
    "justin": (Justin, OmniDrive),
}

ENVIRONMENTS = {
    "apartment": os.path.join(_RESOURCES, "worlds", "apartment.urdf"),
    "kitchen": os.path.join(_RESOURCES, "worlds", "kitchen.urdf"),
}

_START_POSE = (0, 0, 0)

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

    object_annotations = []
    with world.modify_world():
        for stl, cls, x, y, z in PLACED_ON_COUNTER:
            sub = _stl(stl)
            world.merge_world_at_pose(
                sub,
                HomogeneousTransformationMatrix.from_xyz_quaternion(
                    x, y, z, reference_frame=world.root
                ),
            )
            object_annotations.append(cls(root=world.get_body_by_name(stl)))

        for stl, cls, parent, dx, dy, dz in PLACED_IN_BODY:
            try:
                sub = _stl(stl)
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
            except Exception as e:
                print(f"[world] in-body {stl} skipped (parent {parent}): {e}", flush=True)

        for cls, name, parent, dx, dy, dz, sx, sy, sz in PRIMITIVE_IN_BODY:
            try:
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
            except Exception as e:
                print(f"[world] primitive {name} skipped (parent {parent}): {e}", flush=True)

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


def _add_room(world, room_cls, name, cx, cy, w, d):
    hw, hd = w / 2.0, d / 2.0
    polytope = [
        Point3(-hw, -hd, 0.0),
        Point3(-hw, hd, 0.0),
        Point3(hw, hd, 0.0),
        Point3(hw, -hd, 0.0),
    ]
    floor = Floor.create_with_new_body_from_polytope_in_world(
        name=PrefixedName(f"{name}_floor"),
        world=world,
        floor_polytope=polytope,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(cx, cy),
    )
    floor.root.collision = ShapeCollection([])
    world.add_semantic_annotation(room_cls(floor=floor, name=PrefixedName(name)))


def annotate_world(world, object_annotations):
    with world.modify_world():
        WorldReasoner(world).reason()
        world.add_semantic_annotations(object_annotations)

        for surf_name, surf_cls in (
            ("island_countertop", CounterTop),
            ("countertop", CounterTop),
            ("table_area_main", Table),
            ("coffee_table", CoffeeTable),
            ("sofa", Sofa),
        ):
            try:
                surface = surf_cls(root=world.get_body_by_name(surf_name))
                world.add_semantic_annotation(surface)
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
            _add_room(world, room_cls, room_name, cx, cy, w, d)


def annotate_kitchen(world):
    with world.modify_world():
        WorldReasoner(world).reason()

        for surf_name in ("kitchen_island_surface", "sink_area_surface"):
            try:
                surface = CounterTop(root=world.get_body_by_name(surf_name))
                world.add_semantic_annotation(surface)
                surface.calculate_supporting_surface()
            except Exception as e:
                print(f"[world] surface {surf_name} skipped: {e}", flush=True)

        try:
            table = Table(root=world.get_body_by_name("table_area_main"))
            world.add_semantic_annotation(table)
            table.calculate_supporting_surface()
        except Exception as e:
            print(f"[world] table table_area_main skipped: {e}", flush=True)

        for fixture_name, fixture_cls in (
            ("sink_area_sink", Sink),
            ("oven_area_oven_main", Oven),
            ("iai_fridge_main", Fridge),
            ("sink_area_dish_washer_main", Dishwasher),
        ):
            try:
                world.add_semantic_annotation(
                    fixture_cls(root=world.get_body_by_name(fixture_name))
                )
            except Exception as e:
                print(f"[world] fixture {fixture_name} skipped: {e}", flush=True)

        _add_room(world, Kitchen, "kitchen", 2.0, 2.0, 5.0, 5.0)


def _surface_point(world, surface_name):
    """Sample a collision-aware point on an annotated surface; return world (x, y, top_z)."""
    body = world.get_body_by_name(surface_name)
    surface = next(
        (
            a
            for a in world.semantic_annotations
            if getattr(a, "root", None) is body
            and hasattr(a, "sample_points_from_surface")
        ),
        None,
    )
    if surface is None:
        raise RuntimeError(f"surface {surface_name!r} is not annotated")
    point = surface.sample_points_from_surface()[0]
    p = surface.supporting_surface.global_transform @ point
    return float(p.x), float(p.y), float(p.z)


_KITCHEN_STL = [
    ("bowl.stl", Bowl, "kitchen_island_surface"),
    ("jeroen_cup.stl", Mug, "kitchen_island_surface"),
    ("breakfast_cereal.stl", Cereal, "kitchen_island_surface"),
    ("milk.stl", Milk, "kitchen_island_surface"),
    ("spoon.stl", Spoon, "kitchen_island_surface"),
    ("Static_CokeBottle.stl", Bottle, "sink_area_surface"),
]
_KITCHEN_PRIMITIVES = [
    (Plate, "plate", "table_area_main", 0.18, 0.18, 0.02),
    (Apple, "apple", "kitchen_island_surface", 0.08, 0.08, 0.08),
    (Fork, "fork", "table_area_main", 0.18, 0.02, 0.02),
    (Knife, "knife", "table_area_main", 0.18, 0.015, 0.02),
    (MustardBottle, "mustard_bottle", "kitchen_island_surface", 0.06, 0.06, 0.18),
    (WineBottle, "wine_bottle", "table_area_main", 0.07, 0.07, 0.25),
    (SoapBottle, "soap_bottle", "sink_area_surface", 0.06, 0.08, 0.15),
    (Kettle, "kettle", "table_area_main", 0.12, 0.12, 0.18),
]


def place_objects_kitchen(world):
    """Place each object resting on its target surface, at a collision-aware sampled
    point, so is_supported_by holds and classify_world records its location. Requires
    the surfaces to be annotated first (annotate_kitchen)."""
    with world.modify_world():
        for stl, cls, surf_name in _KITCHEN_STL:
            try:
                sub = _stl(stl)
                half = sub.root.combined_mesh.extents[2] / 2.0
                x, y, top = _surface_point(world, surf_name)
                world.merge_world_at_pose(
                    sub,
                    HomogeneousTransformationMatrix.from_xyz_quaternion(
                        x,
                        y,
                        top + half - min(0.05, 0.5 * half),
                        reference_frame=world.root,
                    ),
                )
                world.add_semantic_annotation(cls(root=world.get_body_by_name(stl)))
            except Exception as e:
                print(f"[world] kitchen object {stl} skipped: {e}", flush=True)

        for cls, name, surf_name, sx, sy, sz in _KITCHEN_PRIMITIVES:
            try:
                half = sz / 2.0
                x, y, top = _surface_point(world, surf_name)
                cls.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(name),
                    world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=x, y=y, z=top + half - min(0.05, 0.5 * half)
                    ),
                    scale=Scale(sx, sy, sz),
                )
            except Exception as e:
                print(f"[world] kitchen primitive {name} skipped: {e}", flush=True)


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

    if environment == "kitchen":
        annotate_kitchen(world)
        place_objects_kitchen(world)
    else:
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

    if environment != "kitchen":
        annotate_world(world, object_annotations)

    context.evaluate_conditions = False
    return world, robot, context, node


if __name__ == "__main__":
    world, robot, context, node = build_world()
    rclpy.spin(node)
