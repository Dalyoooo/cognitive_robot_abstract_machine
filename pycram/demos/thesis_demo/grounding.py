"""Ground planner IDs and relations into semDT bodies, poses, and storage."""

from semantic_digital_twin.reasoning.predicates import (
    Behind,
    InFrontOf,
    LeftOf,
    RightOf,
    is_supported_by,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasStorageSpace,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from .schema import DIRECTIONAL_RELATIONS, INSIDE_RELATIONS
from .world_context import (
    find_openable_handle,
    is_at_location,
    is_inside_or_attached,
    planner_ids_for,
    primary_annotation,
    provides_supporting_surface,
    supporting_surface_of,
)

SUPPORT_DETECTION_OVERLAP = 0.01

_DIRECTIONAL_PREDICATES = {
    "left_of": LeftOf,
    "right_of": RightOf,
    "in_front_of": InFrontOf,
    "behind": Behind,
}
assert set(_DIRECTIONAL_PREDICATES) == set(
    DIRECTIONAL_RELATIONS
), "directional predicates must cover exactly the schema relations"


class GroundingError(Exception):
    def __init__(self, message, *, step_index=None, action=None):
        super().__init__(message)
        self.step_index = step_index
        self.action = action

    def attach_step(self, step_index, step):
        if self.step_index is None:
            self.step_index = step_index
        if self.action is None and isinstance(step, dict):
            self.action = step.get("action")
        return self

    def to_dict(self):
        return {
            "message": str(self),
            "step_index": self.step_index,
            "action": self.action,
        }


def directional_relation_holds(point, other, viewpoint, relation):
    """Check one directional relation with the matching semDT predicate."""
    predicate_type = _DIRECTIONAL_PREDICATES[relation]
    predicate = predicate_type(
        point=point,
        other=other,
        point_of_view=viewpoint,
    )
    return bool(predicate())


class Grounding:

    def __init__(self, world, robot=None):
        """Create a grounding helper for one world."""
        self.world = world
        self.robot = robot

        planner_ids = planner_ids_for(world)
        self.ids_by_body = {body: planner_ids[str(body.name)] for body in world.bodies}
        self.bodies_by_id = {
            planner_id: body for body, planner_id in self.ids_by_body.items()
        }

        rooms = world.get_semantic_annotations_by_type(Room)
        self.rooms_by_id = {planner_ids[str(room.name)]: room for room in rooms}

        self.annotations_by_body = {}
        for annotation in world.get_semantic_annotations_by_type(SemanticAnnotation):
            body = getattr(annotation, "root", None)
            if body is not None:
                self.annotations_by_body.setdefault(body, []).append(annotation)

    def resolve_annotation(self, label):
        """Resolve a body or room ID to its most specific annotation, or None."""
        body = self.bodies_by_id.get(label.strip())
        if body is not None:
            annotations = self._annotations_on(body)
            if annotations:
                return primary_annotation(annotations)
        return self.rooms_by_id.get(label.strip())

    def body(self, label, source=None):
        """Resolve an object label to a body, optionally checked at a source."""
        body_label = label.strip()
        body = self.bodies_by_id.get(body_label)
        if body is None:
            raise GroundingError(f"cannot find canonical body ID {body_label!r}")

        if source:
            source_label = source.strip()
            place = self.body(source_label)
            if not is_at_location(body, place):
                raise GroundingError(
                    f"cannot find {body_label!r} at {source_label!r}: "
                    f"not attached, inside, or supported by that location",
                )
        return body

    def body_id(self, body):
        """Return the planner ID exposed for a semantic body."""
        try:
            return self.ids_by_body[body]
        except KeyError as error:
            raise GroundingError(
                f"body {body.name!s} is not exposed to the planner"
            ) from error

    def handle(self, label):
        """Resolve a container label to its articulated handle body."""
        body = self.body(label)
        for annotation in self._annotations_on(body):
            handle = find_openable_handle(annotation)
            if handle is not None:
                return handle.root
        raise GroundingError(f"Cannot open {label!r}: No articulated handle found.")

    def place_pose(self, target_label, obj_body, relation=None):
        """Resolve a surface or container to one object placement pose."""
        return self.place_poses(target_label, obj_body, relation)[0]

    def place_poses(self, target_label, obj_body, relation=None):
        """Resolve a destination to ordered placement pose candidates.

        Surface poses preserve the order provided by semDT.
        """
        if relation in DIRECTIONAL_RELATIONS:
            return [self.directional_pose(target_label, obj_body, relation)]

        if relation in INSIDE_RELATIONS:
            return [self._inside_pose(self.body(target_label), obj_body)]

        surface = self._supporting_surface_annotation(target_label)
        if surface is None:
            raise GroundingError(
                f"cannot place 'on' {target_label!r}: no supporting surface"
            )

        points = self._surface_points(surface, obj_body)
        if not points:
            raise GroundingError(
                f"no free placement point on {str(surface.root.name)!r}"
            )
        return [self._placement_pose(point) for point in points]

    def register_placement(self, object_label, target_label, relation):
        """Register a successful placement in the destination storage space.

        Failures raise RuntimeError on purpose: the object was already placed
        physically, so the executor reports them as execution failures, not
        grounding failures.
        """
        object_body = self.body(object_label)
        object_annotation = self._annotation_on(object_body, HasRootBody)

        if object_annotation is None:
            raise RuntimeError(f"No storage annotation found for {object_label!r}")

        if relation in DIRECTIONAL_RELATIONS:
            target_body = self.body(target_label)
            storage = supporting_surface_of(self.world, target_body)
        elif relation in INSIDE_RELATIONS:
            target_body = self.body(target_label)
            storage = self._annotation_on(target_body, HasStorageSpace)
        else:
            storage = self._supporting_surface_annotation(target_label)

        if storage is None:
            raise RuntimeError(f"No storage space found for {target_label!r}")

        if relation in INSIDE_RELATIONS:
            relation_holds = is_inside_or_attached(object_body, storage.root)
        else:
            relation_holds = is_supported_by(object_body, storage.root)
        if not relation_holds:
            raise RuntimeError(
                f"Placed object {object_label!r} does not satisfy "
                f"{relation!r} at {target_label!r}"
            )

        with self.world.modify_world():
            self._remove_storage_memberships(object_annotation)
            storage.add_object(object_annotation)

    def remove_from_storage(self, object_label):
        """Remove a successfully picked object from all semDT storage lists.

        Failures raise RuntimeError on purpose: the object was already picked
        up physically, so the executor reports them as execution failures.
        """
        object_body = self.body(object_label)
        object_annotation = self._annotation_on(object_body, HasRootBody)
        if object_annotation is None:
            raise RuntimeError(f"No storage annotation found for {object_label!r}")

        with self.world.modify_world():
            self._remove_storage_memberships(object_annotation)

    def navigate_pose(self, label, annotation=None):
        """Resolve a label to a planar navigation target pose."""
        if annotation is None:
            annotation = self.resolve_annotation(label)
        if isinstance(annotation, Room) and annotation.floor is not None:
            body = annotation.floor.root
        else:
            body = self.body(label)
        body_pose = body.global_pose
        return self._world_pose(float(body_pose.x), float(body_pose.y), 0.0)

    # Placement helpers

    def directional_pose(self, reference_label, obj_body, relation):
        """Find the closest free placement point in a view direction."""
        if self.robot is None:
            raise GroundingError("directional placement needs a robot viewpoint")

        reference_body = self.body(reference_label)
        surface = supporting_surface_of(self.world, reference_body)
        if surface is None:
            raise GroundingError(f"{reference_label!r} is not on a supporting surface")

        reference_point = reference_body.global_pose.position
        viewpoint = self.robot.root.global_transform
        matching_points = []
        for point in self._surface_points(surface, obj_body):
            if directional_relation_holds(
                point,
                reference_point,
                viewpoint,
                relation,
            ):
                matching_points.append(point)

        if not matching_points:
            raise GroundingError(
                f"no spot {relation} {reference_label!r} from the robot's view"
            )

        reference_x = float(reference_point.x)
        reference_y = float(reference_point.y)
        point = min(
            matching_points,
            key=lambda candidate: (
                (float(candidate.x) - reference_x) ** 2
                + (float(candidate.y) - reference_y) ** 2
            ),
        )
        return self._placement_pose(point)

    def _inside_pose(self, container_body, obj_body):
        # we just place obj in the middle
        lower, upper = container_body.combined_mesh.bounds
        object_height = obj_body.combined_mesh.extents[2]
        clearance = 0.02
        local_point = Point3(
            x=float((lower[0] + upper[0]) / 2),
            y=float((lower[1] + upper[1]) / 2),
            z=float(lower[2] + object_height / 2 + clearance),
            reference_frame=container_body,
        )
        point = self.world.transform(local_point, self.world.root)
        return self._world_pose(float(point.x), float(point.y), float(point.z))

    def _surface_points(self, surface, obj_body):
        """Return semDT surface samples in the world frame."""
        object_annotation = self._annotation_on(obj_body, HasRootBody)
        points = surface.sample_points_from_surface(
            body_to_sample_for=object_annotation
        )
        return [self.world.transform(point, self.world.root) for point in points]

    def _placement_pose(self, point):
        """Create a sampled placement pose with the demo support overlap."""
        return self._world_pose(
            float(point.x),
            float(point.y),
            float(point.z) - SUPPORT_DETECTION_OVERLAP,
        )

    def _world_pose(self, x, y, z):
        """Create a position-only pose in the world root frame."""
        return Pose.from_xyz_rpy(
            x=x,
            y=y,
            z=z,
            reference_frame=self.world.root,
        )

    def _supporting_surface_annotation(self, label):
        """Return the rooted annotation that provides a usable surface."""
        body = self.body(label)
        for annotation in self._annotations_on(body):
            if provides_supporting_surface(annotation):
                return annotation
        return None

    # Lookup helpers

    def _annotations_on(self, body):
        """Return semantic annotations rooted at a body."""
        return self.annotations_by_body.get(body, [])

    def _annotation_on(self, body, annotation_type):
        """Return the first rooted annotation of a requested type."""
        for annotation in self._annotations_on(body):
            if isinstance(annotation, annotation_type):
                return annotation
        return None

    def _remove_storage_memberships(self, object_annotation):
        storages = self.world.get_semantic_annotations_by_type(HasStorageSpace)
        for storage in storages:
            while object_annotation in storage.objects:
                storage.objects.remove(object_annotation)
