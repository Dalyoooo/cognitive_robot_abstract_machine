from dataclasses import dataclass, field

from semantic_digital_twin.reasoning.predicates import InsideOf, is_supported_by
from semantic_digital_twin.semantic_annotations.mixins import (
    HasApertures,
    HasCaseAsRootBody,
    HasDoors,
    HasDrawers,
    HasHandle,
    HasHinge,
    HasRootBody,
    HasSlider,
    HasStorageSpace,
    HasSupportingSurface,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Furniture,
    Handle,
    Hinge,
    Plate,
    Room,
    Slider,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from thesis_demo.world.environments import FURNITURE_ANNOTATION_TYPES

CONTAINMENT_THRESHOLD = 0.9


def content_types_of(annotation):
    return [
        ancestor
        for ancestor in type(annotation).mro()
        if ancestor is not HasRootBody and issubclass(ancestor, HasRootBody)
    ]


def build_world_context(world, robot):
    rooms = world.get_semantic_annotations_by_type(Room)
    grouped_annotations = group_annotations_by_body(world)
    hidden_roots = _hidden_body_roots(rooms, grouped_annotations)
    robot_bodies = set(robot.bodies)
    task_annotations = {}
    for root, annotations in grouped_annotations.items():
        if root in robot_bodies or root in hidden_roots or root.combined_mesh is None:
            continue
        annotations = [
            annotation
            for annotation in annotations
            if not isinstance(annotation, (HasApertures, Handle, Hinge, Slider))
        ]
        if annotations:
            task_annotations[root] = annotations

    task_annotations = dict(
        sorted(task_annotations.items(), key=lambda item: str(item[0].name))
    )

    planner_names = build_type_names(world)

    entities = _WorldContextEntities.from_annotations(
        rooms,
        task_annotations,
        planner_names,
    )
    location_index = ObjectLocations(
        entities.containers,
        entities.surfaces,
        task_annotations,
    )
    object_locations = entities.build_object_locations(location_index)
    context = entities.to_context(rooms, planner_names, object_locations)
    context["instances"] = _distinguishing_qualifiers(
        entities,
        task_annotations,
        location_index,
    )
    return context


def build_type_names(world):
    grouped_annotations = group_annotations_by_body(world)
    names = {}
    for body in world.bodies:
        annotations = grouped_annotations.get(body, [])
        if annotations:
            names[str(body.name)] = planner_type_name(annotations)
    for room in world.get_semantic_annotations_by_type(Room):
        names[str(room.name)] = type(room).__name__
    return names


def _stored_objects(root, task_annotations):
    return [
        stored_object
        for annotation in task_annotations.get(root, [])
        if isinstance(annotation, HasStorageSpace)
        for stored_object in annotation.objects
        if stored_object.root is not root
    ]


def _common_content_type_name(stored_objects):
    shared = set(content_types_of(stored_objects[0]))
    for stored_object in stored_objects[1:]:
        shared &= set(content_types_of(stored_object))
    for content_type in content_types_of(stored_objects[0]):
        if content_type in shared:
            return content_type.__name__
    return None


def _qualifiers_for_root(root, task_annotations, location_index):
    stored_objects = _stored_objects(root, task_annotations)
    if stored_objects:
        content_type_name = _common_content_type_name(stored_objects)
        if content_type_name is not None:
            return {"contains": content_type_name}
    places = location_index.locations_for(root)
    return {"at": places[0]} if places else {}


def _matches_qualifiers(root, qualifiers, task_annotations, location_index):
    content_type_name = qualifiers.get("contains")
    if content_type_name is not None:
        held_type_names = {
            content_type.__name__
            for stored_object in _stored_objects(root, task_annotations)
            for content_type in content_types_of(stored_object)
        }
        if content_type_name not in held_type_names:
            return False
    place_name = qualifiers.get("at")
    if place_name is not None and place_name not in location_index.locations_for(root):
        return False
    return True


def _distinguishing_qualifiers(entities, task_annotations, location_index):
    roots_by_type = {}
    for type_name, root in (
        *entities.objects,
        *entities.surfaces,
        *entities.containers,
    ):
        roots = roots_by_type.setdefault(type_name, [])
        if not any(known is root for known in roots):
            roots.append(root)

    instances = {}
    for type_name, roots in roots_by_type.items():
        if len(roots) < 2:
            continue
        qualifier_sets = []
        for root in roots:
            qualifiers = _qualifiers_for_root(
                root,
                task_annotations,
                location_index,
            )
            if not qualifiers or qualifiers in qualifier_sets:
                continue
            named = [
                candidate
                for candidate in roots
                if _matches_qualifiers(
                    candidate,
                    qualifiers,
                    task_annotations,
                    location_index,
                )
            ]
            if len(named) == 1:
                qualifier_sets.append(qualifiers)
        if qualifier_sets:
            instances[type_name] = qualifier_sets
    return instances


def planner_type_name(annotations):
    for annotation in annotations:
        if type(annotation) in FURNITURE_ANNOTATION_TYPES:
            return type(annotation).__name__
    return type(most_specific_annotation(annotations)).__name__


def most_specific_annotation(annotations):
    def priority(annotation):
        annotation_type = type(annotation)
        return len(annotation_type.mro()), annotation_type.__name__

    return max(annotations, key=priority)


@dataclass
class _WorldContextEntities:
    objects: list = field(default_factory=list)
    surfaces: list = field(default_factory=list)
    containers: list = field(default_factory=list)
    openables: list = field(default_factory=list)
    furniture: list = field(default_factory=list)
    types: dict = field(default_factory=dict)

    @classmethod
    def from_annotations(cls, rooms, grouped_annotations, planner_names):
        entities = cls()
        for room in rooms:
            entities.types[planner_names[str(room.name)]] = type(room).__name__

        for root, annotations in grouped_annotations.items():
            entities._classify_body(root, annotations, planner_names)
        return entities

    def build_object_locations(self, location_index):
        locations = {}
        for type_name, root in self.objects:
            for place in location_index.locations_for(root):
                places = locations.setdefault(type_name, [])
                if place not in places:
                    places.append(place)
        return locations

    def to_context(self, rooms, planner_names, object_locations):
        return {
            "objects": _deduplicate_names(name for name, _ in self.objects),
            "object_locations": object_locations,
            "surfaces": _deduplicate_names(name for name, _ in self.surfaces),
            "containers": _deduplicate_names(name for name, _ in self.containers),
            "openables": _deduplicate_names(self.openables),
            "furniture": _deduplicate_names(self.furniture),
            "rooms": _deduplicate_names(
                planner_names[str(room.name)] for room in rooms
            ),
            "types": self.types,
        }

    def _classify_body(self, root, annotations, planner_names):
        name = planner_names[str(root.name)]
        self.types[name] = planner_type_name(annotations)

        is_container = any(
            isinstance(annotation, HasCaseAsRootBody) for annotation in annotations
        )
        is_surface = any(
            has_supporting_surface(annotation) for annotation in annotations
        )
        is_furniture = any(
            isinstance(annotation, Furniture)
            or type(annotation) in FURNITURE_ANNOTATION_TYPES
            for annotation in annotations
        )
        is_openable = any(
            find_opening_mechanism(annotation) is not None for annotation in annotations
        )
        is_object = any(
            isinstance(annotation, (Plate, Bowl)) for annotation in annotations
        ) or not (
            is_container
            or is_surface
            or is_furniture
            or is_openable
            or any(isinstance(annotation, HasHinge) for annotation in annotations)
        )

        if is_object:
            self.objects.append((name, root))
        if is_container:
            self.containers.append((name, root))
        if is_surface:
            self.surfaces.append((name, root))
        if is_furniture:
            self.furniture.append(name)
        if is_openable:
            self.openables.append(name)


def _deduplicate_names(values):
    return list(dict.fromkeys(values))


def _hidden_body_roots(rooms, annotations_by_root):
    hidden_roots = {room.floor.root for room in rooms if room.floor is not None}
    hidden_annotations = set()
    for root, annotations in annotations_by_root.items():
        has_declared_furniture = any(
            type(annotation) in FURNITURE_ANNOTATION_TYPES for annotation in annotations
        )
        for annotation in annotations:
            if isinstance(annotation, HasDoors):
                hidden_annotations.update(annotation.doors)
            if isinstance(annotation, HasHandle) and annotation.handle is not None:
                hidden_roots.add(annotation.handle.root)
            if isinstance(annotation, HasHinge) and annotation.hinge is not None:
                hidden_roots.add(annotation.hinge.root)
            if isinstance(annotation, HasSlider) and annotation.slider is not None:
                hidden_roots.add(annotation.slider.root)
            if (
                isinstance(annotation, HasDrawers)
                and annotation.drawers
                and not has_declared_furniture
            ):
                hidden_roots.add(root)
    hidden_roots.update(
        root
        for root, annotations in annotations_by_root.items()
        if any(annotation in hidden_annotations for annotation in annotations)
    )
    return hidden_roots


@dataclass
class ObjectLocations:
    containers: list
    surfaces: list
    annotations_by_root: dict
    container_roots: list = field(init=False)
    surface_roots: list = field(init=False)
    place_names: dict = field(init=False)
    storage_locations: dict = field(init=False)

    def __post_init__(self):
        self.container_roots = [root for _, root in self.containers]
        self.surface_roots = [root for _, root in self.surfaces]
        self.place_names = {
            root: name for name, root in (*self.containers, *self.surfaces)
        }
        self.storage_locations = {}
        for place_name, place_root in (*self.containers, *self.surfaces):
            for annotation in self.annotations_by_root.get(place_root, []):
                if not isinstance(annotation, HasStorageSpace):
                    continue
                for stored_object in annotation.objects:
                    if stored_object.root is place_root:
                        continue
                    place_names = self.storage_locations.setdefault(
                        stored_object.root, []
                    )
                    if place_name not in place_names:
                        place_names.append(place_name)

    def locations_for(self, object_root):
        places = self.storage_locations.get(object_root, [])
        if places:
            return list(places)
        place = self._infer_location(object_root)
        return [place] if place is not None else []

    def name_for(self, place_root):
        return self.place_names.get(place_root, str(place_root.name))

    def _infer_location(self, object_root):
        container = _find_parent_container(object_root, self.container_roots)
        if container is not None:
            return self.name_for(container)

        container = self._find_tightest_container(object_root)
        if container is not None:
            return self.name_for(container)

        surface = _find_most_specific_supporting_body(
            object_root,
            self.surface_roots,
        )
        if surface is not None:
            return self.name_for(surface)
        return None

    def _find_tightest_container(self, object_root):
        best_ratio = CONTAINMENT_THRESHOLD
        best_container = None
        for container_root in self.container_roots:
            if container_root is object_root:
                continue
            ratio = InsideOf(object_root, container_root).compute_containment_ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_container = container_root
        return best_container


def group_annotations_by_body(world):
    grouped_annotations = {}
    for annotation in world.get_semantic_annotations_by_type(HasRootBody):
        grouped_annotations.setdefault(annotation.root, []).append(annotation)
    return grouped_annotations


@dataclass(frozen=True)
class OpeningMechanism:
    handle: object
    connection: ActiveConnection1DOF

    def position_fraction(self):
        # Reading connection.dof copies the degree of freedom, so read it once.
        limits = self.connection.dof.limits
        lower = limits.lower.position
        upper = limits.upper.position
        if lower is None or upper is None or upper <= lower:
            return None
        return (self.connection.position - lower) / (upper - lower)


def find_opening_mechanism(annotation):
    handles = []
    if isinstance(annotation, HasHandle) and annotation.handle is not None:
        handles.append(annotation.handle)
    if isinstance(annotation, HasDoors):
        handles.extend(
            door.handle for door in annotation.doors if door.handle is not None
        )
    if isinstance(annotation, HasDrawers):
        handles.extend(
            drawer.handle for drawer in annotation.drawers if drawer.handle is not None
        )

    for handle in handles:
        try:
            connection = handle.root.get_first_parent_connection_of_type(
                ActiveConnection1DOF
            )
        except ValueError:
            # semDT raises when no movable parent connection leads to the handle.
            continue
        limits = connection.dof.limits
        lower = limits.lower.position
        upper = limits.upper.position
        if lower is not None and upper is not None and lower >= upper:
            continue
        return OpeningMechanism(handle.root, connection)
    return None


def has_supporting_surface(annotation):
    return (
        isinstance(annotation, HasSupportingSurface)
        and annotation.supporting_surface is not None
    )


def is_at_location(object_body, location_body):
    return is_inside_or_attached(object_body, location_body) or is_supported_by(
        object_body, location_body
    )


def is_inside_or_attached(object_body, container_body):
    if _find_parent_container(object_body, [container_body]) is not None:
        return True
    return (
        InsideOf(object_body, container_body).compute_containment_ratio()
        > CONTAINMENT_THRESHOLD
    )


def _find_parent_container(object_body, container_roots):
    parent = object_body.parent_kinematic_structure_entity
    while parent is not None:
        if any(parent is container for container in container_roots):
            return parent
        parent = parent.parent_kinematic_structure_entity
    return None


def find_supporting_surface(world, object_body):
    surfaces = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(HasSupportingSurface)
        if has_supporting_surface(annotation)
    ]
    surface_body = _find_most_specific_supporting_body(
        object_body,
        [surface.root for surface in surfaces],
    )
    for surface in surfaces:
        if surface.root is surface_body:
            return surface
    return None


def _find_most_specific_supporting_body(object_body, surface_roots):
    best_surface = None
    best_key = None
    for surface_body in surface_roots:
        if surface_body is object_body or not is_supported_by(
            object_body, surface_body
        ):
            continue

        depth = 0
        seen = set()
        parent = surface_body.parent_kinematic_structure_entity
        while parent is not None and id(parent) not in seen:
            seen.add(id(parent))
            depth += 1
            parent = parent.parent_kinematic_structure_entity

        key = (depth, str(surface_body.name))
        if best_key is None or key > best_key:
            best_surface = surface_body
            best_key = key
    return best_surface
