import os
import time
import xml.etree.ElementTree as ET
from itertools import combinations

from .nlp_demo_config import (
    ENVIRONMENTS,
    OBJECT_COLORS,
    OBJECTS_DIR,
    SURFACE_ANNOTATION_TYPES,
)
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.mixins import (
    HasDoors,
    HasDrawers,
    HasRootBody,
    HasStorageSpace,
    HasSupportingSurface,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Drawer,
    Floor,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import Body


# A tiny overlap avoids numerical gaps while keeping objects visually on the surface.
_SUPPORT_OVERLAP = 0.005
_GEOMETRY_TOLERANCE = 1e-4

ROBOTS = {
    "pr2": (PR2, OmniDrive),
    "hsrb": (HSRB, OmniDrive),
    "tiago": (Tiago, DifferentialDrive),
}


class FurnitureAnnotationError(RuntimeError):
    pass


def annotate_furniture(world, furniture):
    for declared in furniture:
        annotation = _resolve_or_create_annotation(world, declared)
        _require_door(declared, annotation)
        _ensure_supporting_region(declared, annotation, world)


def _resolve_or_create_annotation(world, declared):
    body = world.get_body_by_name(declared.body)
    matches = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(
            declared.annotation_type
        )
        if type(annotation) is declared.annotation_type and annotation.root is body
    ]
    if len(matches) > 1:
        raise FurnitureAnnotationError(
            f"Multiple {declared.annotation_type.__name__} annotations "
            f"on {declared.body!r}"
        )
    if matches:
        return matches[0]
    return _create_annotation(world, body, declared.annotation_type)


def _create_annotation(world, body, annotation_type):
    attributes = {"root": body}
    if issubclass(annotation_type, HasDoors):
        attributes["doors"] = _branch_annotations(world, body, Door)
    if issubclass(annotation_type, HasDrawers):
        attributes["drawers"] = _branch_annotations(world, body, Drawer)
    annotation = annotation_type(**attributes)

    with world.modify_world():
        if isinstance(annotation, (HasDoors, HasDrawers)):
            world.add_semantic_annotation_recursively(annotation)
        else:
            world.add_semantic_annotation(annotation)
    return annotation


def _branch_annotations(world, root, annotation_type):
    branch = set(world.get_kinematic_structure_entities_of_branch(root))
    return [
        annotation
        for annotation in world.get_semantic_annotations_by_type(annotation_type)
        if annotation.root in branch
    ]


def _require_door(declared, annotation):
    if not isinstance(annotation, HasDoors):
        return
    if annotation.doors:
        return
    raise FurnitureAnnotationError(
        f"WorldReasoner did not infer a door for "
        f"{declared.annotation_type.__name__} {declared.body!r}"
    )


def _ensure_supporting_region(declared, annotation, world):
    if type(annotation) not in SURFACE_ANNOTATION_TYPES:
        return
    if annotation.supporting_surface is not None:
        return
    with world.modify_world():
        region = annotation.calculate_supporting_surface()
    if region is None:
        raise FurnitureAnnotationError(
            f"Could not calculate a supporting surface for {declared.body!r}"
        )


def _primitive(name, scale):
    body = Body(name=PrefixedName(name))
    shapes = BoundingBoxCollection.from_event(
        body, scale.to_simple_event().as_composite_set()
    ).as_shapes()
    body.collision = shapes
    body.visual = shapes
    subworld = World()
    with subworld.modify_world():
        subworld.add_kinematic_structure_entity(body)
    return subworld


def _object_world(placement):
    if (placement.mesh is None) == (placement.scale is None):
        raise ValueError(f"{placement.name!r} must define exactly one of mesh or scale")
    if placement.mesh is not None:
        return STLParser(os.path.join(OBJECTS_DIR, placement.mesh)).parse()
    return _primitive(placement.name, Scale(*placement.scale))


def _apply_color(body, annotation_type):
    rgb = OBJECT_COLORS.get(annotation_type.__name__)
    if rgb is None:
        return
    color = Color(*rgb, 1.0)
    for shape in getattr(body.visual, "shapes", []):
        shape.color = color


def get_annotation(world, body_name, annotation_type, *, usable_surface=False):
    """Return the single annotation of a type rooted at the named body."""
    body = world.get_body_by_name(body_name)
    matches = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(annotation_type)
        if annotation.root is body
        and (not usable_surface or annotation.supporting_surface is not None)
    ]
    if len(matches) != 1:
        qualifier = " usable" if usable_surface else ""
        raise RuntimeError(
            f"Expected exactly one{qualifier} {annotation_type.__name__} "
            f"annotation on {body_name!r}, found {len(matches)}"
        )
    return matches[0]


def _add_room(world, spec):
    width, depth = spec.size
    half_width, half_depth = width / 2.0, depth / 2.0
    floor_polytope = [
        Point3(-half_width, -half_depth, 0.0),
        Point3(-half_width, half_depth, 0.0),
        Point3(half_width, half_depth, 0.0),
        Point3(half_width, -half_depth, 0.0),
    ]
    with world.modify_world():
        floor = Floor.create_with_new_body_from_polytope_in_world(
            name=PrefixedName(f"{spec.name}_floor"),
            world=world,
            floor_polytope=floor_polytope,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                *spec.center
            ),
        )
        # The room floor is semantic only. The URDF already has collision floors.
        floor.root.collision = ShapeCollection([])
        world.add_semantic_annotation(
            spec.annotation_type(floor=floor, name=PrefixedName(spec.name))
        )


def _surface_pose(world, surface, object_body, offset):
    region = surface.supporting_surface
    if region is None or region.combined_mesh is None:
        raise RuntimeError(f"Surface {surface.root.name!s} has no supporting region")
    if surface.root.combined_mesh is None:
        raise RuntimeError(f"Surface {surface.root.name!s} has no geometry")
    if object_body.combined_mesh is None:
        raise RuntimeError(f"Object {object_body.name!s} has no geometry")

    # Use collision bounds because URDF geometry can be offset from its link frame.
    lower, upper = surface.root.combined_mesh.bounds
    object_lower = object_body.combined_mesh.bounds[0]
    local_point = Point3(
        x=float((lower[0] + upper[0]) / 2.0 + offset[0]),
        y=float((lower[1] + upper[1]) / 2.0 + offset[1]),
        z=float(upper[2] - object_lower[2] - _SUPPORT_OVERLAP),
        reference_frame=surface.root,
    )
    world_point = world.transform(local_point, world.root)
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(world_point.x),
        y=float(world_point.y),
        z=float(world_point.z),
        reference_frame=world.root,
    )


def _place_surface_objects(world, placements):
    if not placements:
        return

    with world.modify_world():
        for placement in placements:
            surface = get_annotation(
                world, placement.surface, HasSupportingSurface, usable_surface=True
            )
            object_world = _object_world(placement)
            _apply_color(object_world.root, placement.annotation_type)
            pose = _surface_pose(world, surface, object_world.root, placement.offset)
            world.merge_world_at_pose(object_world, pose)
            body = world.get_body_by_name(placement.name)
            object_annotation = placement.annotation_type(root=body)
            world.add_semantic_annotation(object_annotation)
            surface.add_object(object_annotation)


def _place_contained_objects(world, placements):
    if not placements:
        return

    with world.modify_world():
        for placement in placements:
            object_world = _object_world(placement)
            _apply_color(object_world.root, placement.annotation_type)
            parent = world.get_body_by_name(placement.container)
            world.merge_world(
                object_world,
                FixedConnection(
                    parent=parent,
                    child=object_world.root,
                    parent_T_connection_expression=(
                        HomogeneousTransformationMatrix.from_xyz_rpy(
                            *placement.offset, reference_frame=parent
                        )
                    ),
                ),
            )
            body = world.get_body_by_name(placement.name)
            world.add_semantic_annotation(placement.annotation_type(root=body))

    # Storage registration also reparents through semDT's public storage API.
    for placement in placements:
        container = get_annotation(world, placement.container, placement.container_type)
        if not isinstance(container, HasStorageSpace):
            continue
        stored_object = get_annotation(world, placement.name, placement.annotation_type)
        with world.modify_world():
            container.add_object(stored_object)


def _body_bounds_in_frame(world, body, frame):
    if body.combined_mesh is None:
        raise RuntimeError(f"Body {body.name!s} has no collision geometry")
    mesh = body.combined_mesh.copy()
    frame_transform = world.compute_forward_kinematics(root=frame, tip=body)
    mesh.apply_transform(frame_transform.to_np())
    return mesh.bounds


def _validate_surface_geometry(world, body, surface):
    object_lower, object_upper = _body_bounds_in_frame(world, body, surface.root)
    surface_lower, surface_upper = surface.root.combined_mesh.bounds
    for axis in (0, 1):
        if (
            object_lower[axis] < surface_lower[axis] - _GEOMETRY_TOLERANCE
            or object_upper[axis] > surface_upper[axis] + _GEOMETRY_TOLERANCE
        ):
            raise RuntimeError(
                f"{body.name!s} footprint exceeds surface {surface.root.name!s}"
            )

    penetration = float(surface_upper[2] - object_lower[2])
    if not (
        -_GEOMETRY_TOLERANCE <= penetration <= _SUPPORT_OVERLAP + _GEOMETRY_TOLERANCE
    ):
        raise RuntimeError(
            f"{body.name!s} penetrates {surface.root.name!s} by {penetration:.4f} m"
        )


def _validate_contained_geometry(world, body, container):
    if container.combined_mesh is None:
        raise RuntimeError(f"Container {container.name!s} has no collision geometry")
    object_lower, object_upper = _body_bounds_in_frame(world, body, container)
    container_lower, container_upper = container.combined_mesh.bounds
    for axis in range(3):
        if (
            object_lower[axis] < container_lower[axis] - _GEOMETRY_TOLERANCE
            or object_upper[axis] > container_upper[axis] + _GEOMETRY_TOLERANCE
        ):
            raise RuntimeError(
                f"{body.name!s} lies outside the bounds of {container.name!s}"
            )


def _validate_object_separation(world, spec):
    placements = (*spec.surface_objects, *spec.contained_objects)
    bodies = [world.get_body_by_name(placement.name) for placement in placements]
    bounds = {body: _body_bounds_in_frame(world, body, world.root) for body in bodies}
    for first, second in combinations(bodies, 2):
        first_lower, first_upper = bounds[first]
        second_lower, second_upper = bounds[second]
        overlap = [
            min(first_upper[axis], second_upper[axis])
            - max(first_lower[axis], second_lower[axis])
            for axis in range(3)
        ]
        if all(amount > _GEOMETRY_TOLERANCE for amount in overlap):
            raise RuntimeError(f"Objects {first.name!s} and {second.name!s} overlap")


def _validate_environment(world, spec):
    world.validate()

    for room_spec in spec.rooms:
        rooms = [
            room
            for room in world.get_semantic_annotations_by_type(
                room_spec.annotation_type
            )
            if str(room.name) == room_spec.name
        ]
        if len(rooms) != 1:
            raise RuntimeError(
                f"Expected exactly one {room_spec.annotation_type.__name__} "
                f"named {room_spec.name!r}, found {len(rooms)}"
            )

    root_annotations = {}
    for annotation in world.get_semantic_annotations_by_type(HasRootBody):
        key = (type(annotation), annotation.root)
        root_annotations[key] = root_annotations.get(key, 0) + 1
    duplicates = [
        (annotation_type.__name__, str(root.name))
        for (annotation_type, root), count in root_annotations.items()
        if count > 1
    ]
    if duplicates:
        raise RuntimeError(f"Duplicate root annotations: {duplicates}")

    for placement in spec.surface_objects:
        body = world.get_body_by_name(placement.name)
        object_annotation = get_annotation(
            world, placement.name, placement.annotation_type
        )
        surface = get_annotation(
            world, placement.surface, HasSupportingSurface, usable_surface=True
        )
        if not is_supported_by(body, surface.root):
            raise RuntimeError(
                f"{placement.name!r} is not supported by {placement.surface!r}"
            )
        if object_annotation not in surface.objects:
            raise RuntimeError(
                f"{placement.name!r} is missing from {placement.surface!r}.objects"
            )
        if body.parent_kinematic_structure_entity is not surface.root:
            raise RuntimeError(
                f"{placement.name!r} is not attached to {placement.surface!r}"
            )
        _validate_surface_geometry(world, body, surface)

    for placement in spec.contained_objects:
        body = world.get_body_by_name(placement.name)
        parent = world.get_body_by_name(placement.container)
        object_annotation = get_annotation(
            world, placement.name, placement.annotation_type
        )
        container = get_annotation(world, placement.container, placement.container_type)
        if body.parent_kinematic_structure_entity is not parent:
            raise RuntimeError(
                f"{placement.name!r} is not attached to {placement.container!r}"
            )
        if (
            isinstance(container, HasStorageSpace)
            and object_annotation not in container.objects
        ):
            raise RuntimeError(
                f"{placement.name!r} is missing from {placement.container!r}.objects"
            )
        _validate_contained_geometry(world, body, parent)

    _validate_object_separation(world, spec)


def _build_environment(spec):
    urdf_root = ET.parse(spec.urdf).getroot()

    # The kitchen URDF contains a duplicate limit that would freeze the drawer.
    left_drawer_joint = urdf_root.find(
        "./joint[@name='oven_area_area_left_drawer_main_joint']"
    )
    if left_drawer_joint is not None:
        for duplicate_limit in left_drawer_joint.findall("limit")[1:]:
            left_drawer_joint.remove(duplicate_limit)

    world = URDFParser(urdf=ET.tostring(urdf_root, encoding="unicode")).parse()

    # Keep household inference isolated from robot links and robot part names.
    WorldReasoner(world).infer_semantic_annotations()
    annotate_furniture(world, spec.furniture)

    for room_spec in spec.rooms:
        _add_room(world, room_spec)

    _place_surface_objects(world, spec.surface_objects)
    _place_contained_objects(world, spec.contained_objects)
    _validate_environment(world, spec)
    return world


def _attach_robot(world, robot_name, start_pose):
    robot_type, drive_type = ROBOTS[robot_name]
    robot_world = URDFParser.from_file(robot_type.get_ros_file_path()).parse()

    with world.modify_world():
        drive = drive_type.create_with_dofs(
            parent=world.root,
            child=robot_world.root,
            world=world,
        )
        world.merge_world(robot_world, drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*start_pose)
        drive.has_hardware_interface = True

    if not drive.has_hardware_interface:
        raise RuntimeError(f"Drive for robot {robot_name!r} is not controlled")
    return robot_type.from_world(world)


def build_world_model(robot_name="hsrb", environment="apartment"):
    """Build and validate the semDT world without starting ROS visualization."""
    if robot_name not in ROBOTS:
        raise ValueError(f"Unknown robot {robot_name!r}; choose from {sorted(ROBOTS)}")
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment {environment!r}; choose from {sorted(ENVIRONMENTS)}"
        )

    spec = ENVIRONMENTS[environment]
    world = _build_environment(spec)
    robot = _attach_robot(world, robot_name, spec.robot_start)
    world.validate()

    context = Context(world=world, robot=robot)
    context.evaluate_conditions = False
    return world, robot, context


def _clear_markers(node, topic="/semworld/viz_marker", timeout=5.0):
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import Marker, MarkerArray

    qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(MarkerArray, topic, qos)

    # Deterministic wait instead of fixed sleeps: only clear once a
    # subscriber (RViz) actually matched. The clear matters only on
    # environment/robot switches with a long-lived RViz. Same-map rebuilds
    # are covered by marker overwrite, headless runs publish nothing.
    deadline = time.monotonic() + timeout
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        time.sleep(0.05)

    marker = Marker()
    marker.action = Marker.DELETEALL
    marker_array = MarkerArray()
    marker_array.markers.append(marker)
    publisher.publish(marker_array)


def _start_visualization(world):
    import rclpy
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("viz_marker")
    _clear_markers(node)
    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()
    return node


def build_world(robot_name="hsrb", environment="apartment", *, visualize=True):
    """Build the Binder demo and optionally start its ROS visualization node."""
    world, robot, context = build_world_model(robot_name, environment)
    node = _start_visualization(world) if visualize else None
    return world, robot, context, node


if __name__ == "__main__":
    import rclpy

    world, robot, context, node = build_world()
    rclpy.spin(node)
