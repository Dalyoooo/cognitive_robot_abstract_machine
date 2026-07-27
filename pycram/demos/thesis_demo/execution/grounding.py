from dataclasses import dataclass, field

from krrood.entity_query_language.exceptions import (
    MultipleSolutionFound,
    NoSolutionFound,
)
from krrood.entity_query_language.factories import an, entity, the, variable
from krrood.entity_query_language.predicate import symbolic_function
from semantic_digital_twin.reasoning.predicates import (
    Behind,
    ContainsType,
    InFrontOf,
    LeftOf,
    RightOf,
    allclose,
    compute_euclidean_planar_distance,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasStorageSpace,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose, Vector3
from semantic_digital_twin.world_description.world_entity import Body

from thesis_demo.planner.world_context import (
    content_types_of,
    group_annotations_by_body,
    find_opening_mechanism,
    find_supporting_surface,
    has_supporting_surface,
    is_at_location,
    is_inside_or_attached,
)
from thesis_demo.validation.schema import (
    DIRECTIONAL_RELATIONS,
    INSIDE_RELATIONS,
    description_key,
)

PLACEMENT_TOLERANCE = 0.03
CONTAINER_PLACEMENT_CLEARANCE = 0.02


@symbolic_function
def at_location(object_body: Body, location_body: Body) -> bool:
    return is_at_location(object_body, location_body)


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


def directional_relation_holds(object_point, reference_point, viewpoint, relation):
    predicate_type = _DIRECTIONAL_PREDICATES[relation]
    predicate = predicate_type(
        point=object_point,
        other=reference_point,
        point_of_view=viewpoint,
    )
    return bool(predicate())


@dataclass
class Grounding:
    world: object
    robot: object = None
    annotations_by_body: dict = field(init=False)
    resolved_bodies_by_description: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.annotations_by_body = group_annotations_by_body(self.world)

    def _annotation_types_by_name(self):
        annotation_types = {}
        for annotation in self.world.get_semantic_annotations_by_type(HasRootBody):
            annotation_types[type(annotation).__name__] = type(annotation)
        for room in self.world.get_semantic_annotations_by_type(Room):
            annotation_types[type(room).__name__] = type(room)
        return annotation_types

    def resolve_annotation(self, description):
        annotations = self._candidate_annotations(description)
        return annotations[0] if annotations else None

    def candidates(self, description):
        description_identifier = description_key(description)
        resolved_body = self.resolved_bodies_by_description.get(description_identifier)
        if resolved_body is not None:
            return [resolved_body]

        bodies = [
            annotation.root for annotation in self._candidate_annotations(description)
        ]
        if len(bodies) == 1:
            self.resolved_bodies_by_description[description_identifier] = bodies[0]
        return bodies

    def the_body(self, description):
        description_identifier = description_key(description)
        resolved_body = self.resolved_bodies_by_description.get(description_identifier)
        if resolved_body is not None:
            return resolved_body

        query = self._description_query(description)
        if query is None:
            raise GroundingError(f"nothing in this world matches {description!r}")
        try:
            # `the()` yields before it counts, so the solutions must be drained
            # for uniqueness to be enforced at all.
            annotations = list(the(query).evaluate())
        except NoSolutionFound as error:
            raise GroundingError(
                f"nothing in this world matches {description!r}"
            ) from error
        except MultipleSolutionFound as error:
            raise GroundingError(
                f"{description!r} matches several entities; the description is ambiguous"
            ) from error
        resolved_body = annotations[0].root
        self.resolved_bodies_by_description[description_identifier] = resolved_body
        return resolved_body

    def resolve_body(self, description, source=None):
        body = self.the_body(description)

        if source is not None:
            source_bodies = self.candidates(source)
            if not source_bodies:
                raise GroundingError(f"nothing in this world matches {source!r}")
            if not any(
                is_at_location(body, source_body) for source_body in source_bodies
            ):
                raise GroundingError(
                    f"cannot find {description!r} at {source!r}: "
                    "not attached, inside, or supported by that location",
                )
        return body

    def _candidate_annotations(self, description):
        query = self._description_query(description)
        if query is None:
            return []
        return list(an(query).evaluate())

    def _description_query(self, description):
        annotation_type = self._annotation_types_by_name().get(description["type"])
        if annotation_type is None:
            return None

        instances = self._exact_instances_of(annotation_type)
        if not instances:
            # Ordering a query over an empty domain raises instead of yielding
            # nothing, so a type with no instances is answered before asking.
            return None

        annotation_variable = variable(annotation_type, domain=instances)
        conditions = self._description_conditions(
            annotation_variable,
            annotation_type,
            description,
        )
        query = (
            entity(annotation_variable).where(*conditions)
            if conditions
            else entity(annotation_variable)
        )
        return self._ordered_by_distance(
            query,
            annotation_variable,
            annotation_type,
        )

    def _ordered_by_distance(self, query, annotation_variable, annotation_type):
        if self.robot is None:
            return query
        if not issubclass(annotation_type, HasRootBody):
            # A room is reached through its floor and has no body to measure.
            return query
        return query.ordered_by(
            compute_euclidean_planar_distance(
                annotation_variable.root,
                self.robot.root,
                ignore_dimension=Vector3(z=1),
            )
        )

    def _exact_instances_of(self, annotation_type):
        return [
            annotation
            for annotation in self.world.get_semantic_annotations_by_type(
                annotation_type
            )
            if type(annotation) is annotation_type
        ]

    def _content_types_by_name(self):
        content_types = {}
        for annotation in self.world.get_semantic_annotations_by_type(HasRootBody):
            for content_type in content_types_of(annotation):
                content_types[content_type.__name__] = content_type
        return content_types

    def _description_conditions(
        self, annotation_variable, annotation_type, description
    ):
        conditions = []
        place_type_name = description.get("at")
        if place_type_name is not None:
            conditions.append(
                self._location_condition(annotation_variable, place_type_name)
            )

        content_type_name = description.get("contains")
        if content_type_name is not None:
            conditions.append(
                self._contains_condition(
                    annotation_variable,
                    annotation_type,
                    content_type_name,
                )
            )
        return conditions

    def _contains_condition(
        self,
        annotation_variable,
        annotation_type,
        content_type_name,
    ):
        if not issubclass(annotation_type, HasStorageSpace):
            # Asking this of a thing that holds nothing is a broken description,
            # not an empty world: without this the query just yields nothing and
            # the demo would blame the world for it.
            raise GroundingError(
                f"a {annotation_type.__name__} has no storage space, "
                f"so it cannot contain a {content_type_name!r}"
            )
        content_type = self._content_types_by_name().get(content_type_name)
        if content_type is None:
            raise GroundingError(f"nothing in this world is a {content_type_name!r}")
        return ContainsType(annotation_variable.objects, content_type)

    def _location_condition(self, annotation_variable, place_type_name):
        place_type = self._annotation_types_by_name().get(place_type_name)
        if place_type is None:
            raise GroundingError(f"nothing in this world is a {place_type_name!r}")
        place_variable = variable(
            place_type,
            domain=self._exact_instances_of(place_type),
        )
        return at_location(annotation_variable.root, place_variable.root)

    def candidate_handles(self, description):
        candidate_handles = []
        for body in self.candidates(description):
            for annotation in self._annotations_for(body):
                mechanism = find_opening_mechanism(annotation)
                if mechanism is not None:
                    candidate_handles.append(mechanism.handle)
                    break
        if not candidate_handles:
            raise GroundingError(f"Cannot open {description!r}: no opening mechanism.")
        return candidate_handles

    def opening_mechanism_for_handle(self, handle):
        for annotations in self.annotations_by_body.values():
            for annotation in annotations:
                mechanism = find_opening_mechanism(annotation)
                if mechanism is not None and mechanism.handle is handle:
                    return mechanism
        raise GroundingError(f"Cannot find an opening mechanism for handle {handle!r}.")

    def _supporting_surface_of(self, body):
        for annotation in self._annotations_for(body):
            if has_supporting_surface(annotation):
                return annotation
        return None

    def _require_root_annotation(self, object_body, object_description):
        annotation = self._find_annotation(object_body, HasRootBody)
        if annotation is None:
            raise RuntimeError(
                f"No annotation found for the body of {object_description!r}"
            )
        return annotation

    def _annotations_for(self, body):
        return self.annotations_by_body.get(body, [])

    def _find_annotation(self, body, annotation_type):
        for annotation in self._annotations_for(body):
            if isinstance(annotation, annotation_type):
                return annotation
        return None

    def navigation_pose(self, description, annotation=None):
        if annotation is None:
            annotation = self.resolve_annotation(description)
        if isinstance(annotation, Room) and annotation.floor is not None:
            body = annotation.floor.root
        else:
            body = self.resolve_body(description)
        body_pose = body.global_pose
        return Pose.from_xyz_rpy(
            x=float(body_pose.x),
            y=float(body_pose.y),
            reference_frame=self.world.root,
        )

    def placement_poses(self, target, object_body, relation=None):
        if relation in DIRECTIONAL_RELATIONS:
            return [self.directional_placement_pose(target, object_body, relation)]

        targets = self.candidates(target)
        if not targets:
            raise GroundingError(f"nothing in this world matches {target!r}")

        if relation in INSIDE_RELATIONS:
            return [
                self._inside_placement_pose(container, object_body)
                for container in targets
            ]

        poses = []
        for target_body in targets:
            surface = self._supporting_surface_of(target_body)
            if surface is None:
                continue
            poses.extend(
                Pose(position=point, reference_frame=self.world.root)
                for point in self._sample_surface_points(surface, object_body)
            )

        if not poses:
            raise GroundingError(f"no free placement point on {target!r}")
        return poses

    def directional_placement_pose(self, reference, object_body, relation):
        if self.robot is None:
            raise GroundingError("directional placement needs a robot viewpoint")

        reference_body = self.resolve_body(reference)
        surface = find_supporting_surface(self.world, reference_body)
        if surface is None:
            raise GroundingError(f"{reference!r} is not on a supporting surface")

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
                f"no spot {relation} {reference!r} from the robot's view"
            )

        reference_x_coordinate = float(reference_point.x)
        reference_y_coordinate = float(reference_point.y)
        point = min(
            matching_points,
            key=lambda candidate: (
                (float(candidate.x) - reference_x_coordinate) ** 2
                + (float(candidate.y) - reference_y_coordinate) ** 2
            ),
        )
        return Pose(position=point, reference_frame=self.world.root)

    def _inside_placement_pose(self, container_body, object_body):
        lower_bound, upper_bound = container_body.combined_mesh.bounds
        object_height = object_body.combined_mesh.extents[2]
        local_point = Point3(
            x=float((lower_bound[0] + upper_bound[0]) / 2),
            y=float((lower_bound[1] + upper_bound[1]) / 2),
            z=float(lower_bound[2] + object_height / 2 + CONTAINER_PLACEMENT_CLEARANCE),
            reference_frame=container_body,
        )
        world_point = self.world.transform(local_point, self.world.root)
        return Pose(position=world_point, reference_frame=self.world.root)

    def _sample_surface_points(self, supporting_surface, object_body):
        object_annotation = self._find_annotation(object_body, HasRootBody)
        surface_points = supporting_surface.sample_points_from_surface(
            body_to_sample_for=object_annotation
        )
        return [
            self.world.transform(surface_point, self.world.root)
            for surface_point in surface_points
        ]

    def record_placement(self, object_description, target, relation, target_pose):
        object_body = self.resolve_body(object_description)
        object_annotation = self._require_root_annotation(
            object_body,
            object_description,
        )
        storage = self._resolve_storage(target, relation, object_body)
        if relation in INSIDE_RELATIONS and not is_inside_or_attached(
            object_body, storage.root
        ):
            raise self._unsatisfied_relation_error(object_description, target, relation)

        with self.world.modify_world():
            self._remove_from_all_storage(object_annotation)
            self._add_to_storage(
                object_annotation,
                storage,
                object_description,
                target,
                relation,
                target_pose,
            )

    def clear_storage_memberships(self, object_description):
        object_body = self.resolve_body(object_description)
        object_annotation = self._require_root_annotation(
            object_body,
            object_description,
        )

        with self.world.modify_world():
            self._remove_from_all_storage(object_annotation)

    def _resolve_storage(self, target, relation, object_body):
        if relation in DIRECTIONAL_RELATIONS:
            storage = find_supporting_surface(self.world, self.resolve_body(target))
        elif relation in INSIDE_RELATIONS:
            storage = self._storage_reached_by_object(
                target,
                object_body,
                lambda body: self._find_annotation(body, HasStorageSpace),
                is_inside_or_attached,
            )
        else:
            storage = self._storage_reached_by_object(
                target,
                object_body,
                self._supporting_surface_of,
                is_at_location,
            )

        if storage is None:
            raise RuntimeError(f"No storage space found for {target!r}")
        return storage

    def _storage_reached_by_object(
        self,
        target,
        object_body,
        storage_for_body,
        relation_holds,
    ):
        candidates = self.candidates(target)
        for candidate in candidates:
            storage = storage_for_body(candidate)
            if storage is not None and relation_holds(object_body, candidate):
                return storage
        for candidate in candidates:
            storage = storage_for_body(candidate)
            if storage is not None:
                return storage
        return None

    def _add_to_storage(
        self,
        object_annotation,
        storage,
        object_description,
        target,
        relation,
        target_pose,
    ):
        if relation in INSIDE_RELATIONS:
            storage.add_object(object_annotation)
            return

        # Only where the object came to rest decides this, not how it is turned:
        # the gripper keeps the yaw it grasped with, so comparing whole poses
        # rejects a correct placement for being rotated.
        if not allclose(
            object_annotation.root.global_pose.to_position(),
            target_pose.to_position(),
            atol=PLACEMENT_TOLERANCE,
        ):
            raise self._unsatisfied_relation_error(
                object_description,
                target,
                relation,
            )
        storage.add_object(object_annotation)

    @staticmethod
    def _unsatisfied_relation_error(object_description, target, relation):
        return RuntimeError(
            f"Placed object {object_description!r} does not satisfy "
            f"{relation!r} at {target!r}"
        )

    def _remove_from_all_storage(self, object_annotation):
        storages = self.world.get_semantic_annotations_by_type(HasStorageSpace)
        for storage in storages:
            while object_annotation in storage.objects:
                storage.objects.remove(object_annotation)
