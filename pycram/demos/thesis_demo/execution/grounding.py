from dataclasses import dataclass, field

from semantic_digital_twin.reasoning.predicates import (
    Behind,
    InFrontOf,
    LeftOf,
    RightOf,
    allclose,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasStorageSpace,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose

from thesis_demo.planner.world_context import (
    PlannerNames,
    find_opening_mechanism,
    find_supporting_surface,
    has_supporting_surface,
    is_at_location,
    is_inside_or_attached,
    most_specific_annotation,
)
from thesis_demo.validation.schema import DIRECTIONAL_RELATIONS, INSIDE_RELATIONS

PLACEMENT_TOLERANCE = 0.03
CONTAINER_PLACEMENT_CLEARANCE = 0.02

_DIRECTIONAL_PREDICATES = {
    "left_of": LeftOf,
    "right_of": RightOf,
    "in_front_of": InFrontOf,
    "behind": Behind,
}
if set(_DIRECTIONAL_PREDICATES) != set(DIRECTIONAL_RELATIONS):
    raise RuntimeError("Directional predicate mapping does not match the schema")


@dataclass(eq=False)
class GroundingError(Exception):
    message: str
    step_index: object = None
    action: object = None

    def __post_init__(self):
        Exception.__init__(self, self.message)

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
    predicate_type = _DIRECTIONAL_PREDICATES[relation]
    predicate = predicate_type(
        point=point,
        other=other,
        point_of_view=viewpoint,
    )
    return bool(predicate())


@dataclass
class Grounding:
    world: object
    robot: object = None
    names: object = None
    names_by_body: dict = field(init=False)
    bodies_by_name: dict = field(init=False)
    rooms_by_name: dict = field(init=False)
    annotations_by_body: dict = field(init=False)

    def __post_init__(self):
        names = self.names or PlannerNames.build(self.world)
        self.names_by_body = names.by_body
        self.bodies_by_name = names.bodies_by_name
        self.rooms_by_name = names.rooms_by_name
        self.annotations_by_body = names.annotations_by_body

    # ----- Entity lookup -----

    def resolve_annotation(self, label):
        body = self.bodies_by_name.get(label.strip())
        if body is not None:
            annotations = self._annotations_for(body)
            if annotations:
                return most_specific_annotation(annotations)
        return self.rooms_by_name.get(label.strip())

    def resolve_body(self, label, source=None):
        body_label = label.strip()
        body = self.bodies_by_name.get(body_label)
        if body is None:
            raise GroundingError(f"cannot find canonical body name {body_label!r}")

        if source:
            source_label = source.strip()
            place = self.resolve_body(source_label)
            if not is_at_location(body, place):
                raise GroundingError(
                    f"cannot find {body_label!r} at {source_label!r}: "
                    f"not attached, inside, or supported by that location",
                )
        return body

    def planner_name_for(self, body):
        planner_name = self.names_by_body.get(body)
        if planner_name is None:
            raise GroundingError(f"body {body.name!s} is not exposed to the planner")
        return planner_name

    def resolve_handle(self, label):
        return self.resolve_opening_mechanism(label).handle

    def resolve_opening_mechanism(self, label):
        body = self.resolve_body(label)
        for annotation in self._annotations_for(body):
            mechanism = find_opening_mechanism(annotation)
            if mechanism is not None:
                return mechanism
        raise GroundingError(f"Cannot open {label!r}: No opening mechanism found.")

    def _find_supporting_surface(self, label):
        body = self.resolve_body(label)
        for annotation in self._annotations_for(body):
            if has_supporting_surface(annotation):
                return annotation
        return None

    def _require_root_annotation(self, object_body, object_label):
        annotation = self._find_annotation(object_body, HasRootBody)
        if annotation is None:
            raise RuntimeError(f"No storage annotation found for {object_label!r}")
        return annotation

    def _annotations_for(self, body):
        return self.annotations_by_body.get(body, [])

    def _find_annotation(self, body, annotation_type):
        for annotation in self._annotations_for(body):
            if isinstance(annotation, annotation_type):
                return annotation
        return None

    # ----- Placement poses -----

    def navigation_pose(self, label, annotation=None):
        if annotation is None:
            annotation = self.resolve_annotation(label)
        if isinstance(annotation, Room) and annotation.floor is not None:
            body = annotation.floor.root
        else:
            body = self.resolve_body(label)
        body_pose = body.global_pose
        return self._world_pose(float(body_pose.x), float(body_pose.y), 0.0)

    def placement_pose(self, target_label, object_body, relation=None):
        return self.placement_poses(target_label, object_body, relation)[0]

    def placement_poses(self, target_label, object_body, relation=None):
        if relation in DIRECTIONAL_RELATIONS:
            return [
                self.directional_placement_pose(target_label, object_body, relation)
            ]

        if relation in INSIDE_RELATIONS:
            return [
                self._inside_placement_pose(
                    self.resolve_body(target_label),
                    object_body,
                )
            ]

        surface = self._find_supporting_surface(target_label)
        if surface is None:
            raise GroundingError(
                f"cannot place 'on' {target_label!r}: no supporting surface"
            )

        points = self._sample_surface_points(surface, object_body)
        if not points:
            raise GroundingError(
                f"no free placement point on {str(surface.root.name)!r}"
            )
        return [self._pose_at(point) for point in points]

    def directional_placement_pose(self, reference_label, object_body, relation):
        if self.robot is None:
            raise GroundingError("directional placement needs a robot viewpoint")

        reference_body = self.resolve_body(reference_label)
        surface = find_supporting_surface(self.world, reference_body)
        if surface is None:
            raise GroundingError(f"{reference_label!r} is not on a supporting surface")

        reference_point = reference_body.global_pose.position
        viewpoint = self.robot.root.global_transform
        matching_points = [
            point
            for point in self._sample_surface_points(surface, object_body)
            if directional_relation_holds(
                point,
                reference_point,
                viewpoint,
                relation,
            )
        ]

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
        return self._pose_at(point)

    def _inside_placement_pose(self, container_body, object_body):
        # just placing in the middle for the demo
        lower, upper = container_body.combined_mesh.bounds
        object_height = object_body.combined_mesh.extents[2]
        local_point = Point3(
            x=float((lower[0] + upper[0]) / 2),
            y=float((lower[1] + upper[1]) / 2),
            z=float(lower[2] + object_height / 2 + CONTAINER_PLACEMENT_CLEARANCE),
            reference_frame=container_body,
        )
        point = self.world.transform(local_point, self.world.root)
        return self._world_pose(float(point.x), float(point.y), float(point.z))

    def _sample_surface_points(self, surface, object_body):
        object_annotation = self._find_annotation(object_body, HasRootBody)
        points = surface.sample_points_from_surface(
            body_to_sample_for=object_annotation
        )
        return [self.world.transform(point, self.world.root) for point in points]

    def _pose_at(self, point):
        return self._world_pose(float(point.x), float(point.y), float(point.z))

    def _world_pose(self, x, y, z):
        return Pose.from_xyz_rpy(
            x=x,
            y=y,
            z=z,
            reference_frame=self.world.root,
        )

    # ----- Storage  -----

    def record_placement(self, object_label, target_label, relation, target_pose):
        object_body = self.resolve_body(object_label)
        object_annotation = self._require_root_annotation(
            object_body,
            object_label,
        )
        storage = self._resolve_storage(target_label, relation)
        self._validate_storage_relation(
            object_body,
            storage,
            object_label,
            target_label,
            relation,
        )

        with self.world.modify_world():
            self._remove_from_all_storage(object_annotation)
            self._add_to_storage(
                object_annotation,
                storage,
                object_label,
                target_label,
                relation,
                target_pose,
            )

    def clear_storage_memberships(self, object_label):
        object_body = self.resolve_body(object_label)
        object_annotation = self._require_root_annotation(
            object_body,
            object_label,
        )

        with self.world.modify_world():
            self._remove_from_all_storage(object_annotation)

    def _resolve_storage(self, target_label, relation):
        if relation in DIRECTIONAL_RELATIONS:
            target_body = self.resolve_body(target_label)
            storage = find_supporting_surface(self.world, target_body)
        elif relation in INSIDE_RELATIONS:
            target_body = self.resolve_body(target_label)
            storage = self._find_annotation(target_body, HasStorageSpace)
        else:
            storage = self._find_supporting_surface(target_label)

        if storage is None:
            raise RuntimeError(f"No storage space found for {target_label!r}")
        return storage

    def _validate_storage_relation(
        self,
        object_body,
        storage,
        object_label,
        target_label,
        relation,
    ):
        if relation not in INSIDE_RELATIONS:
            return
        if is_inside_or_attached(object_body, storage.root):
            return
        raise self._unsatisfied_relation_error(
            object_label,
            target_label,
            relation,
        )

    def _add_to_storage(
        self,
        object_annotation,
        storage,
        object_label,
        target_label,
        relation,
        target_pose,
    ):
        if relation in INSIDE_RELATIONS:
            storage.add_object(object_annotation)
            return

        # The object reached the surface if it ended up where pyCRAM was told to put it.
        if not allclose(
            object_annotation.root.global_pose,
            target_pose,
            atol=PLACEMENT_TOLERANCE,
        ):
            raise self._unsatisfied_relation_error(
                object_label,
                target_label,
                relation,
            )
        if object_annotation not in storage.objects:
            storage.add_object(object_annotation)

    @staticmethod
    def _unsatisfied_relation_error(object_label, target_label, relation):
        return RuntimeError(
            f"Placed object {object_label!r} does not satisfy "
            f"{relation!r} at {target_label!r}"
        )

    def _remove_from_all_storage(self, object_annotation):
        storages = self.world.get_semantic_annotations_by_type(HasStorageSpace)
        for storage in storages:
            while object_annotation in storage.objects:
                storage.objects.remove(object_annotation)
