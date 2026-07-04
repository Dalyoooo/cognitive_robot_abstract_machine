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
    Apple,
    Bottle,
    Bowl,
    Cereal,
    CheezeIt,
    CoffeeTable,
    CounterTop,
    Cup,
    Dishwasher,
    Floor,
    Fork,
    Fridge,
    GelatinBox,
    Kettle,
    Kitchen,
    Knife,
    LivingRoom,
    Milk,
    Mug,
    MustardBottle,
    Oven,
    Plate,
    Pringles,
    SaltContainer,
    SideTable,
    Sink,
    SoapBottle,
    Spoon,
    Table,
    TomatoSoup,
    TunaCan,
    WineBottle,
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
from semantic_digital_twin.world_description.geometry import Color, Scale
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
}

ENVIRONMENTS = {
    "apartment": os.path.join(_RESOURCES, "worlds", "apartment.urdf"),
    "kitchen": os.path.join(_RESOURCES, "worlds", "kitchen.urdf"),
}

_START_POSES = {
    "apartment": (1.5, 2.5, 0),
    "kitchen": (0.3, 0.8, 0),
}
_DEFAULT_START_POSE = (1.5, 2.5, 0)

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


def annotate_kitchen(world):
    with world.modify_world():
        WorldReasoner(world).reason()

        for surf_name in (
            "kitchen_island_surface",
            "sink_area_surface",
        ):
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

        _add_room(world, Kitchen, "kitchen", -1.0, 0.56, 5.6, 4.9)


def annotate_apartment(world):
    with world.modify_world():
        WorldReasoner(world).reason()

        for surf_name, surf_cls in (
            ("island_countertop", CounterTop),
            ("countertop", CounterTop),
            ("table_area_main", Table),
            ("coffee_table", CoffeeTable),
            ("bedside_table", SideTable),
        ):
            try:
                surface = surf_cls(root=world.get_body_by_name(surf_name))
                world.add_semantic_annotation(surface)
                surface.calculate_supporting_surface()
            except Exception as e:
                print(f"[world] surface {surf_name} skipped: {e}", flush=True)

        for fixture_name, fixture_cls in (
            ("sink", Sink),
            ("oven", Oven),
            ("cabinet7", Dishwasher),
        ):
            try:
                world.add_semantic_annotation(
                    fixture_cls(root=world.get_body_by_name(fixture_name))
                )
            except Exception as e:
                print(f"[world] fixture {fixture_name} skipped: {e}", flush=True)

        for room_cls, room_name, cx, cy, w, d in ROOMS:
            _add_room(world, room_cls, room_name, cx, cy, w, d)


def _surface_point(world, surface_name):
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

    world_mesh = body.combined_mesh.copy()
    world_mesh.apply_transform(body.global_transform.to_np())
    pose = body.global_pose
    cx, cy = float(pose.position.x), float(pose.position.y)
    top = float(world_mesh.bounds[1][2])
    return cx, cy, top, surface


def _infer_surface_objects(surfaces):
    """Update semDT surface membership once after all objects were added."""
    seen = set()
    for surface in surfaces:
        if id(surface) in seen:
            continue
        seen.add(id(surface))
        try:
            surface.infer_objects_on_surface()
        except Exception as error:
            print(
                f"[world] surface membership update skipped: {error}",
                flush=True,
            )


def _validate_placed_objects(world, body_names):
    """Reject a demo world that silently lost an expected object."""
    missing = []
    for name in body_names:
        try:
            body = world.get_body_by_name(name)
        except Exception:
            missing.append(name)
            continue
        if not any(
            getattr(annotation, "root", None) is body
            for annotation in world.semantic_annotations
        ):
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Demo world is missing expected annotated objects: "
            + ", ".join(sorted(missing))
        )


_KITCHEN_STL = [
    ("bowl.stl", Bowl, "kitchen_island_surface", 0.0, -0.10),
    ("jeroen_cup.stl", Mug, "kitchen_island_surface", 0.20, -0.10),
    ("breakfast_cereal.stl", Cereal, "oven_area_area", 0.0, 0.0),
    ("milk.stl", Milk, "fridge_area", -0.15, -0.10),
    ("Static_CokeBottle.stl", Bottle, "sink_area_surface", -0.15, 0.0),
]
_KITCHEN_PRIMITIVES = [
    (Plate, "plate", "table_area_main", -0.15, -0.10, 0.18, 0.18, 0.02),
    (Apple, "apple", "kitchen_island_surface", -0.20, 0.10, 0.08, 0.08, 0.08),
    (MustardBottle, "mustard_bottle", "fridge_area", 0.15, -0.10, 0.06, 0.06, 0.18),
    (WineBottle, "wine_bottle", "table_area_main", -0.15, 0.15, 0.07, 0.07, 0.25),
    (SoapBottle, "soap_bottle", "sink_area_surface", 0.15, 0.0, 0.06, 0.08, 0.15),
    (Kettle, "kettle", "table_area_main", 0.20, 0.15, 0.12, 0.12, 0.18),
    (CheezeIt, "cheezeit", "kitchen_island_surface", 0.20, 0.10, 0.06, 0.06, 0.12),
    (Pringles, "pringles", "fridge_area", -0.15, 0.10, 0.07, 0.07, 0.20),
    (GelatinBox, "gelatinbox", "fridge_area", 0.15, 0.10, 0.06, 0.06, 0.08),
]
_KITCHEN_IN_DRAWER_STL = [
    ("spoon.stl", Spoon, "kitchen_island_left_upper_drawer_main", -0.05, 0.0, 0.0),
]
_KITCHEN_IN_DRAWER_PRIMITIVE = [
    (
        Fork,
        "fork",
        "kitchen_island_right_upper_drawer_main",
        -0.05,
        0.0,
        0.0,
        0.18,
        0.02,
        0.02,
    ),
    (
        Knife,
        "knife",
        "kitchen_island_middle_upper_drawer_main",
        -0.05,
        0.0,
        0.0,
        0.18,
        0.015,
        0.02,
    ),
    (
        TunaCan,
        "tunacan",
        "sink_area_left_upper_drawer_main",
        0.0,
        0.0,
        0.0,
        0.06,
        0.06,
        0.08,
    ),
    (
        SaltContainer,
        "saltcontainer",
        "sink_area_left_middle_drawer_main",
        0.0,
        0.0,
        0.0,
        0.05,
        0.05,
        0.12,
    ),
    (TomatoSoup, "tomatosoup", "iai_fridge_main", 0.05, 0.0, 0.05, 0.06, 0.06, 0.10),
]

_APARTMENT_STL = [
    ("bowl.stl", Bowl, "island_countertop", -0.20, -0.10),
    ("breakfast_cereal.stl", Cereal, "island_countertop", 0.20, -0.10),
    ("milk.stl", Milk, "island_countertop", -0.20, 0.10),
    ("Static_CokeBottle.stl", Bottle, "countertop", 0.0, 0.0),
    ("jeroen_cup.stl", Mug, "table_area_main", 0.0, 0.0),
]
_APARTMENT_PRIMITIVES = [
    (Plate, "plate", "table_area_main", -0.15, -0.10, 0.18, 0.18, 0.02),
    (Plate, "plate_counter", "countertop", 0.15, -0.10, 0.18, 0.18, 0.02),
    (Apple, "apple", "table_area_main", -0.15, 0.10, 0.08, 0.08, 0.08),
    (Apple, "apple_island", "island_countertop", 0.15, 0.10, 0.08, 0.08, 0.08),
]
_APARTMENT_IN_DRAWER_STL = [
    ("spoon.stl", Spoon, "cabinet10_drawer_top", -0.05, -0.10, 0.0),
]
_APARTMENT_IN_DRAWER_PRIMITIVE = [
    (Fork, "fork", "cabinet10_drawer_top", -0.05, 0.0, 0.0, 0.18, 0.02, 0.02),
    (Knife, "knife", "cabinet10_drawer_top", -0.05, 0.10, 0.0, 0.18, 0.015, 0.02),
    (Mug, "mug_sink", "sink", 0.0, 0.0, 0.0, 0.08, 0.08, 0.10),
]


_OBJECT_COLORS = {
    "Bowl": (0.20, 0.40, 0.80),
    "Mug": (0.80, 0.20, 0.20),
    "Cereal": (0.80, 0.70, 0.20),
    "Milk": (0.92, 0.92, 0.92),
    "Spoon": (0.75, 0.75, 0.78),
    "Bottle": (0.70, 0.10, 0.10),
    "Plate": (0.92, 0.92, 0.92),
    "Apple": (0.80, 0.15, 0.15),
    "Fork": (0.75, 0.75, 0.78),
    "Knife": (0.75, 0.75, 0.78),
    "MustardBottle": (0.85, 0.72, 0.10),
    "WineBottle": (0.45, 0.10, 0.12),
    "SoapBottle": (0.20, 0.70, 0.30),
    "Kettle": (0.20, 0.20, 0.22),
    "TunaCan": (0.70, 0.50, 0.30),
    "CheezeIt": (0.90, 0.60, 0.05),
    "Pringles": (0.90, 0.10, 0.10),
    "GelatinBox": (0.60, 0.30, 0.70),
    "TomatoSoup": (0.80, 0.10, 0.10),
    "SaltContainer": (0.85, 0.85, 0.85),
}


def _apply_color(body, cls):
    """Set visual shape colours from the per-class colour map."""
    rgb = _OBJECT_COLORS.get(cls.__name__)
    if rgb is None:
        return
    color = Color(rgb[0], rgb[1], rgb[2], 1.0)
    for shape in getattr(body.visual, "shapes", []):
        shape.color = color


def _excluded_near_robot(x, y, start_pose):
    """Return True if (x, y) is within 0.6 m of the robot start pose."""
    sx, sy, _ = start_pose
    return (x - sx) ** 2 + (y - sy) ** 2 < 0.36


def place_objects_kitchen(world):
    placed_xy = []
    surfaces = []
    with world.modify_world():
        for stl, cls, surf_name, x_off, y_off in _KITCHEN_STL:
            try:
                sub = _stl(stl)
                _apply_color(sub.root, cls)
                half = sub.root.combined_mesh.extents[2] / 2.0
                bottom_offset = sub.root.combined_mesh.bounds[0][2]
                cx, cy, top, surface = _surface_point(world, surf_name)
                px, py = cx + x_off, cy + y_off
                placed_xy.append((px, py, max(half, 0.05)))
                world.merge_world_at_pose(
                    sub,
                    HomogeneousTransformationMatrix.from_xyz_quaternion(
                        px,
                        py,
                        top - bottom_offset - min(0.05, 0.5 * half),
                        reference_frame=world.root,
                    ),
                )
                world.add_semantic_annotation(cls(root=world.get_body_by_name(stl)))
            except Exception as e:
                print(f"[world] kitchen object {stl} skipped: {e}", flush=True)
            else:
                surfaces.append(surface)

        for cls, name, surf_name, x_off, y_off, sx, sy, sz in _KITCHEN_PRIMITIVES:
            try:
                half = sz / 2.0
                cx, cy, top, surface = _surface_point(world, surf_name)
                px, py = cx + x_off, cy + y_off
                placed_xy.append((px, py, max(half, 0.05)))
                cls.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(name),
                    world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=px, y=py, z=top + half - min(0.05, 0.5 * half)
                    ),
                    scale=Scale(sx, sy, sz),
                )
                _apply_color(world.get_body_by_name(name), cls)
            except Exception as e:
                print(f"[world] kitchen primitive {name} skipped: {e}", flush=True)
            else:
                surfaces.append(surface)

        for stl, cls, parent, dx, dy, dz in _KITCHEN_IN_DRAWER_STL:
            try:
                sub = _stl(stl)
                _apply_color(sub.root, cls)
                world.merge_world(
                    sub,
                    FixedConnection(
                        parent=world.get_body_by_name(parent),
                        child=sub.root,
                        parent_T_connection_expression=(
                            HomogeneousTransformationMatrix.from_xyz_rpy(dx, dy, dz)
                        ),
                    ),
                )
                world.add_semantic_annotation(cls(root=world.get_body_by_name(stl)))
            except Exception as e:
                print(f"[world] kitchen in-drawer {stl} skipped: {e}", flush=True)

        for cls, name, parent, dx, dy, dz, sx, sy, sz in _KITCHEN_IN_DRAWER_PRIMITIVE:
            try:
                sub = _primitive(name, Scale(sx, sy, sz))
                _apply_color(sub.root, cls)
                world.merge_world(
                    sub,
                    FixedConnection(
                        parent=world.get_body_by_name(parent),
                        child=sub.root,
                        parent_T_connection_expression=(
                            HomogeneousTransformationMatrix.from_xyz_rpy(dx, dy, dz)
                        ),
                    ),
                )
                world.add_semantic_annotation(cls(root=world.get_body_by_name(name)))
            except Exception as e:
                print(f"[world] kitchen in-drawer {name} skipped: {e}", flush=True)

        _infer_surface_objects(surfaces)

    expected_names = (
        [entry[0] for entry in _KITCHEN_STL]
        + [entry[1] for entry in _KITCHEN_PRIMITIVES]
        + [entry[0] for entry in _KITCHEN_IN_DRAWER_STL]
        + [entry[1] for entry in _KITCHEN_IN_DRAWER_PRIMITIVE]
    )
    _validate_placed_objects(world, expected_names)

    # Validate: no surface-placed object in the excluded-near-robot zone
    kitchen_start = _START_POSES.get("kitchen", _DEFAULT_START_POSE)
    for px, py, pr in placed_xy:
        if _excluded_near_robot(px, py, kitchen_start):
            print(
                f"[world] WARNING: kitchen object at ({px:.3f}, {py:.3f}) "
                f"is within 0.6m of robot start pose",
                flush=True,
            )


def place_objects_apartment(world):
    surfaces = []
    with world.modify_world():
        for stl, cls, surf_name, x_off, y_off in _APARTMENT_STL:
            try:
                sub = _stl(stl)
                _apply_color(sub.root, cls)
                half = sub.root.combined_mesh.extents[2] / 2.0
                bottom_offset = sub.root.combined_mesh.bounds[0][2]
                cx, cy, top, surface = _surface_point(world, surf_name)
                world.merge_world_at_pose(
                    sub,
                    HomogeneousTransformationMatrix.from_xyz_quaternion(
                        cx + x_off,
                        cy + y_off,
                        top - bottom_offset - min(0.05, 0.5 * half),
                        reference_frame=world.root,
                    ),
                )
                world.add_semantic_annotation(cls(root=world.get_body_by_name(stl)))
            except Exception as e:
                print(f"[world] apartment object {stl} skipped: {e}", flush=True)
            else:
                surfaces.append(surface)

        for cls, name, surf_name, x_off, y_off, sx, sy, sz in _APARTMENT_PRIMITIVES:
            try:
                half = sz / 2.0
                cx, cy, top, surface = _surface_point(world, surf_name)
                cls.create_with_new_body_in_world(
                    world=world,
                    name=PrefixedName(name),
                    world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=cx + x_off, y=cy + y_off, z=top + half - min(0.05, 0.5 * half)
                    ),
                    scale=Scale(sx, sy, sz),
                )
                _apply_color(world.get_body_by_name(name), cls)
            except Exception as e:
                print(f"[world] apartment primitive {name} skipped: {e}", flush=True)
            else:
                surfaces.append(surface)

        for stl, cls, parent, dx, dy, dz in _APARTMENT_IN_DRAWER_STL:
            try:
                sub = _stl(stl)
                _apply_color(sub.root, cls)
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
                world.add_semantic_annotation(cls(root=world.get_body_by_name(stl)))
            except Exception as e:
                print(f"[world] apartment in-drawer {stl} skipped: {e}", flush=True)

        for cls, name, parent, dx, dy, dz, sx, sy, sz in _APARTMENT_IN_DRAWER_PRIMITIVE:
            try:
                sub = _primitive(name, Scale(sx, sy, sz))
                _apply_color(sub.root, cls)
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
                world.add_semantic_annotation(cls(root=world.get_body_by_name(name)))
            except Exception as e:
                print(f"[world] apartment in-drawer {name} skipped: {e}", flush=True)

        _infer_surface_objects(surfaces)

    expected_names = (
        [entry[0] for entry in _APARTMENT_STL]
        + [entry[1] for entry in _APARTMENT_PRIMITIVES]
        + [entry[0] for entry in _APARTMENT_IN_DRAWER_STL]
        + [entry[1] for entry in _APARTMENT_IN_DRAWER_PRIMITIVE]
    )
    _validate_placed_objects(world, expected_names)


def build_world(robot_name="pr2", environment="apartment"):
    if robot_name not in ROBOTS:
        raise ValueError(f"Unknown robot {robot_name!r}; choose from {sorted(ROBOTS)}")
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment {environment!r}; choose from {sorted(ENVIRONMENTS)}"
        )

    robot_cls, drive_cls = ROBOTS[robot_name]

    world = URDFParser.from_file(ENVIRONMENTS[environment]).parse()
    if robot_name == "pr2":
        robot_world = URDFParser.from_file(
            "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
        ).parse()
    else:
        robot_world = URDFParser.from_file(robot_cls.get_ros_file_path()).parse()

    with world.modify_world():
        drive = drive_cls.create_with_dofs(
            parent=world.root, child=robot_world.root, world=world
        )
        world.merge_world(robot_world, drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            *_START_POSES.get(environment, _DEFAULT_START_POSE)
        )

    if environment == "kitchen":
        annotate_kitchen(world)
        place_objects_kitchen(world)
    else:
        annotate_apartment(world)
        place_objects_apartment(world)

    try:
        rclpy.init()
    except RuntimeError:
        pass

    node = rclpy.create_node("viz_marker")
    _clear_markers(node)
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

    robot = robot_cls.from_world(world)
    context = Context(world=world, robot=robot)

    context.evaluate_conditions = False
    return world, robot, context, node


if __name__ == "__main__":
    world, robot, context, node = build_world()
    rclpy.spin(node)
