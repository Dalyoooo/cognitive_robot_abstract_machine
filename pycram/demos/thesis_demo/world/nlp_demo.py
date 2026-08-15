import os
import time
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import combinations

from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import InsideOf, is_supported_by
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
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    OmniDrive,
)
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Region

from thesis_demo.planner.world_context import CONTAINMENT_THRESHOLD
from thesis_demo.world.environment import OBJECT_COLORS, OBJECTS_DIR
from thesis_demo.world.environments import ENVIRONMENTS

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
    if not isinstance(annotation, HasSupportingSurface):
        return
    if annotation.supporting_surface is not None:
        return
    with world.modify_world():
        region = annotation.calculate_supporting_surface()
    if region is None:
        raise FurnitureAnnotationError(
            f"Could not calculate a supporting surface for {declared.body!r}"
        )


def _apply_color(body, annotation_type):
    rgb = OBJECT_COLORS.get(annotation_type.__name__)
    if rgb is None:
        return
    color = Color(*rgb, 1.0)
    for shape in body.visual.shapes:
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
    with world.modify_world():
        floor = Floor.create_with_new_body_in_world(
            name=PrefixedName(f"{spec.name}_floor"),
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                *spec.center
            ),
            scale=Scale(width, depth, 0.0),
        )
        # The room floor is semantic only. The URDF already has collision floors.
        floor.root.collision = ShapeCollection([])
        world.add_semantic_annotation(
            spec.annotation_type(floor=floor, name=PrefixedName(spec.name))
        )


def _surface_pose(world, surface, object_lower_z, offset):
    region = surface.supporting_surface
    if region is None or region.combined_mesh is None:
        raise RuntimeError(f"Surface {surface.root.name!s} has no supporting region")

    # The region polygon is the actual usable surface; the furniture bbox also
    # covers basins, legs, and frames.
    lower, upper = _body_bounds_in_frame(region, surface.root)
    local_point = Point3(
        x=float((lower[0] + upper[0]) / 2.0 + offset[0]),
        y=float((lower[1] + upper[1]) / 2.0 + offset[1]),
        z=float(upper[2] - object_lower_z - _SUPPORT_OVERLAP),
        reference_frame=surface.root,
    )
    world_point = world.transform(local_point, world.root)
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(world_point.x),
        y=float(world_point.y),
        z=float(world_point.z),
        reference_frame=world.root,
    )


def _object_lower_z(placement):
    # The pose sits the object on the surface, so its lowest point is needed
    # before the body exists.
    _check_geometry(placement)
    if placement.scale is not None:
        return -placement.scale[2] / 2
    mesh_world = STLParser(os.path.join(OBJECTS_DIR, placement.mesh)).parse()
    return float(mesh_world.root.combined_mesh.bounds[0][2])


def _check_geometry(placement):
    if (placement.mesh is None) == (placement.scale is None):
        raise ValueError(f"{placement.name!r} must define exactly one of mesh or scale")


def _create_object(world, placement, pose):
    _check_geometry(placement)
    if placement.scale is not None:
        annotation = placement.annotation_type.create_with_new_body_in_world(
            name=PrefixedName(placement.name),
            world=world,
            world_root_T_self=pose,
            scale=Scale(*placement.scale),
        )
        _apply_color(annotation.root, placement.annotation_type)
        return annotation
    object_world = STLParser(os.path.join(OBJECTS_DIR, placement.mesh)).parse()
    _apply_color(object_world.root, placement.annotation_type)
    world.merge_world_at_pose(object_world, pose)
    body = world.get_body_by_name(placement.name)
    annotation = placement.annotation_type(root=body)
    world.add_semantic_annotation(annotation)
    return annotation


def _place_surface_objects(world, placements):
    if not placements:
        return

    with world.modify_world():
        for placement in placements:
            surface = get_annotation(
                world, placement.surface, HasSupportingSurface, usable_surface=True
            )
            pose = _surface_pose(
                world, surface, _object_lower_z(placement), placement.offset
            )
            object_annotation = _create_object(world, placement, pose)
            surface.add_object(object_annotation)


def _place_contained_objects(world, placements):
    if not placements:
        return

    with world.modify_world():
        for placement in placements:
            parent = world.get_body_by_name(placement.container)
            pose = world.transform(
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    *placement.offset, reference_frame=parent
                ),
                world.root,
            )
            _create_object(world, placement, pose)

    # Storage registration also reparents through semDT's public storage API.
    for placement in placements:
        container = get_annotation(world, placement.container, placement.container_type)
        if not isinstance(container, HasStorageSpace):
            continue
        stored_object = get_annotation(world, placement.name, placement.annotation_type)
        with world.modify_world():
            container.add_object(stored_object)


def _body_bounds_in_frame(body, frame):
    if body.combined_mesh is None:
        raise RuntimeError(f"Body {body.name!s} has no collision geometry")
    shapes = body.area if isinstance(body, Region) else body.collision
    bounding_box = shapes.as_bounding_box_collection_in_frame(frame).bounding_box()
    return (
        (bounding_box.min_x, bounding_box.min_y, bounding_box.min_z),
        (bounding_box.max_x, bounding_box.max_y, bounding_box.max_z),
    )


def _validate_surface_geometry(body, surface):
    object_lower, object_upper = _body_bounds_in_frame(body, surface.root)
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


def _validate_contained_geometry(body, container):
    if container.combined_mesh is None:
        raise RuntimeError(f"Container {container.name!s} has no collision geometry")
    # The same predicate the planner uses to decide what counts as being inside.
    if InsideOf(body, container).compute_containment_ratio() <= CONTAINMENT_THRESHOLD:
        raise RuntimeError(
            f"{body.name!s} lies outside the bounds of {container.name!s}"
        )


def _validate_object_separation(world, spec):
    # Axis-aligned bounds in the world frame. That is conservative for a rotated
    # object, which is fine because every demo object is an axis-aligned box.
    placements = (*spec.surface_objects, *spec.contained_objects)
    bodies = [world.get_body_by_name(placement.name) for placement in placements]
    bounds = {body: _body_bounds_in_frame(body, world.root) for body in bodies}
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

    root_annotations = Counter(
        (type(annotation), annotation.root)
        for annotation in world.get_semantic_annotations_by_type(HasRootBody)
    )
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
        _validate_surface_geometry(body, surface)

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
        _validate_contained_geometry(body, parent)

    _validate_object_separation(world, spec)


def _build_environment(spec):
    urdf_root = ET.parse(spec.urdf).getroot()
    if spec.urdf_customizer is not None:
        spec.urdf_customizer(urdf_root)

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

    # The publisher keeps its last sample on offer as long as it exists, so a
    # viewer connecting later could still receive this delete after the real
    # markers and wipe them. Drop it once the delete is out.
    time.sleep(0.2)
    node.destroy_publisher(publisher)


REPUBLISH_PERIOD_S = 1.0
"""How often the current marker array is repeated for late subscribers."""

VIEWER_ROOT_FRAME = "map"
"""Frame the viewer resolves everything against."""

VIEWER_CAMERA_FRAME = "demo_camera_target"
"""Frame the viewer's orbit camera turns around."""


def _static_transform(stamp, parent_frame, child_frame, translation=(0.0, 0.0, 0.0)):
    from geometry_msgs.msg import TransformStamped

    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    transform.transform.translation.x = float(translation[0])
    transform.transform.translation.y = float(translation[1])
    transform.transform.translation.z = float(translation[2])
    transform.transform.rotation.w = 1.0
    return transform


def _publish_viewer_frames(
    node, world, camera_target=None, parent_frame=VIEWER_ROOT_FRAME
):
    """Publish the frames a viewer needs to place and aim at the scene.

    Frames carry the name of the body they belong to, so the root of the
    apartment is called something like ``apartment/apartment_root`` and changes
    with the environment. A viewer configured on a fixed frame of its own finds
    no path to those frames and drops every marker, which leaves an empty grid.
    The camera frame gives the orbit view a point to turn around; without it
    the view opens on an arbitrary spot that need not contain the scene.
    """
    from tf2_ros import StaticTransformBroadcaster

    stamp = node.get_clock().now().to_msg()
    transforms = []

    root_frame = str(world.root.name)
    if root_frame != parent_frame:
        transforms.append(_static_transform(stamp, parent_frame, root_frame))
    if camera_target is not None:
        transforms.append(
            _static_transform(stamp, parent_frame, VIEWER_CAMERA_FRAME, camera_target)
        )

    if not transforms:
        return None

    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(transforms)
    return broadcaster


def start_visualization(
    world, camera_target=None, republish_period_s=REPUBLISH_PERIOD_S
):
    import rclpy
    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("viz_marker")
    _clear_markers(node)
    # Held on the node so the broadcaster outlives this call.
    node._viewer_frame_broadcaster = _publish_viewer_frames(node, world, camera_target)
    marker_publisher = VizMarkerPublisher(_world=world, node=node)
    tf_publisher = TFPublisher(_world=world, node=node)

    # Both the shapes and the frames they hang in are sent once: the markers
    # when the world is built, the transforms whenever the world state moves.
    # A viewer that connects later receives neither, because both topics drop
    # what was sent before it subscribed, so it shows an empty scene. Repeating
    # them on a timer means the scene no longer depends on when a viewer joined.
    # Without the transforms the markers alone are useless; every one of them
    # names the body frame it belongs to and is discarded if that frame is
    # unknown.
    def republish_scene():
        marker_publisher.pub.publish(marker_publisher.markers)
        tf_publisher.on_state_change()

    node.create_timer(republish_period_s, republish_scene)
    return node


def build_world(robot_name="hsrb", environment="apartment", *, visualize=True):
    """Build the Binder demo and optionally start its ROS visualization node.

    The context carries the world and the robot, so it is the only handle a
    caller needs.
    """
    world, robot, context = build_world_model(robot_name, environment)
    # Where the robot starts is where the action is, so the view opens there.
    camera_target = ENVIRONMENTS[environment].robot_start
    node = start_visualization(world, camera_target) if visualize else None
    return context, node


if __name__ == "__main__":
    import rclpy

    _context, node = build_world()
    rclpy.spin(node)
