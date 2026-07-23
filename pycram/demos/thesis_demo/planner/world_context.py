import math
import re
from dataclasses import dataclass, field

from semantic_digital_twin.reasoning.predicates import InsideOf, is_supported_by
from semantic_digital_twin.semantic_annotations.mixins import (
    HasCaseAsRootBody,
    HasDoors,
    HasDrawers,
    HasHandle,
    HasRootBody,
    HasStorageSpace,
    HasSupportingSurface,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Door,
    DoubleDoor,
    Floor,
    Furniture,
    Handle,
    Hinge,
    Plate,
    Room,
    Slider,
    Wall,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from thesis_demo.world.nlp_demo_config import FURNITURE_ANNOTATION_TYPES

SKIPPED_ANNOTATION_TYPES = (Handle, Hinge, Slider, Wall, Floor)
CONTAINER_TYPES = (HasCaseAsRootBody,)
TABLEWARE_TYPES = (Plate, Bowl)
DOOR_TYPES = (Door, DoubleDoor)
CONTAINMENT_THRESHOLD = 0.9


def classify_world(world, robot):
    rooms = world.get_semantic_annotations_by_type(Room)
    grouped_annotations = _task_annotations_by_body(world, robot)
    planner_names = planner_names_for(world)

    entities = _ClassifiedEntities.from_annotations(
        rooms,
        grouped_annotations,
        planner_names,
    )
    object_locations = entities.object_locations(grouped_annotations)
    return entities.as_context(rooms, planner_names, object_locations)


def annotations_by_body(world):
    grouped_annotations = {}
    for annotation in world.get_semantic_annotations_by_type(HasRootBody):
        grouped_annotations.setdefault(annotation.root, []).append(annotation)
    return grouped_annotations


@dataclass(frozen=True)
class _EntityCapabilities:
    is_object: bool
    is_container: bool
    is_surface: bool
    is_furniture: bool
    is_openable: bool


@dataclass
class _ClassifiedEntities:
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

        hidden_roots = _hidden_task_roots(grouped_annotations)
        for root, annotations in grouped_annotations.items():
            if root in hidden_roots:
                continue
            entities._add(root, annotations, planner_names)
        return entities

    def object_locations(self, grouped_annotations):
        place_index = PlaceIndex(
            self.containers,
            self.surfaces,
            grouped_annotations,
        )
        locations = {}
        for name, root in self.objects:
            object_places = place_index.locations_of(root)
            if object_places:
                locations[name] = object_places
        return locations

    def as_context(self, rooms, planner_names, object_locations):
        return {
            "objects": _unique_names(name for name, _ in self.objects),
            "object_locations": object_locations,
            "surfaces": _unique_names(name for name, _ in self.surfaces),
            "containers": _unique_names(name for name, _ in self.containers),
            "openables": _unique_names(self.openables),
            "furniture": _unique_names(self.furniture),
            "rooms": _unique_names(planner_names[str(room.name)] for room in rooms),
            "types": self.types,
        }

    def _add(self, root, annotations, planner_names):
        name = planner_names[str(root.name)]
        self.types[name] = annotation_type_name(annotations)
        capabilities = _entity_capabilities(annotations)

        if capabilities.is_object:
            self.objects.append((name, root))
        if capabilities.is_container:
            self.containers.append((name, root))
        if capabilities.is_surface:
            self.surfaces.append((name, root))
        if capabilities.is_furniture:
            self.furniture.append(name)
        if capabilities.is_openable:
            self.openables.append(name)


def _unique_names(values):
    return list(dict.fromkeys(values))


def _handles(annotation):
    handles = []
    if isinstance(annotation, HasHandle) and annotation.handle is not None:
        handles.append(annotation.handle)

    children = []
    if isinstance(annotation, HasDoors):
        children.extend(annotation.doors)
    if isinstance(annotation, HasDrawers):
        children.extend(annotation.drawers)
    for child in children:
        if child.handle is not None:
            handles.append(child.handle)

    return handles


def _is_locked(connection):
    # semDT exposes joint limits but no lock predicate. A finite zero-width
    # range is the only reliable indication that the connection cannot move.
    lower = connection.dof.limits.lower.position
    upper = connection.dof.limits.upper.position
    if lower is None or upper is None:
        return False
    if not math.isfinite(lower) or not math.isfinite(upper):
        return False
    return abs(upper - lower) < 1e-9


def find_openable_handle(annotation):
    for handle in _handles(annotation):
        try:
            connection = handle.root.get_first_parent_connection_of_type(
                ActiveConnection1DOF
            )
        except ValueError:
            # semDT raises when no articulated connection leads to the handle.
            continue
        if not _is_locked(connection):
            return handle
    return None


def primary_annotation(annotations):
    # semDT permits several annotations per root but has no primary selector.
    return max(
        annotations,
        key=lambda item: (len(type(item).mro()), type(item).__name__),
    )


def provides_supporting_surface(annotation):
    return (
        isinstance(annotation, HasSupportingSurface)
        and annotation.supporting_surface is not None
    )


def _snake_token(class_name):
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", class_name)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", words)
    return "_".join(words.lower().split())


def annotation_type_name(annotations):
    for annotation in annotations:
        if type(annotation) in FURNITURE_ANNOTATION_TYPES:
            return type(annotation).__name__
    return type(primary_annotation(annotations)).__name__


def _structural_body_name(name):
    # semDT exposes the URDF body name but no planner label for repeated components.
    if "/" not in name:
        return None
    local_name = name.rsplit("/", 1)[-1].removesuffix("_main")
    return _snake_token(local_name) if local_name else None


def _descriptive_duplicate_names(names, reserved_names):
    descriptive_names = [_structural_body_name(name) for name in names]
    if not all(descriptive_names):
        return None
    if len(descriptive_names) != len(set(descriptive_names)):
        return None
    if set(descriptive_names).intersection(reserved_names):
        return None
    return descriptive_names


def planner_names_for(world):
    # semDT names preserve model prefixes. Planner names need stable, short tokens.
    rooms = world.get_semantic_annotations_by_type(Room)
    grouped_annotations = annotations_by_body(world)

    names_by_token = {}
    for body in world.bodies:
        annotations = grouped_annotations.get(body, [])
        if annotations:
            token = _snake_token(annotation_type_name(annotations))
        else:
            token = str(body.name).rsplit("/", 1)[-1]
        names_by_token.setdefault(token, []).append(str(body.name))

    for room in rooms:
        token = _snake_token(type(room).__name__)
        names_by_token.setdefault(token, []).append(str(room.name))

    planner_names = {}
    reserved_names = set(names_by_token)
    for token, names in names_by_token.items():
        if len(names) == 1:
            planner_names[names[0]] = token
            continue
        sorted_names = sorted(names)
        descriptive_names = _descriptive_duplicate_names(sorted_names, reserved_names)
        if descriptive_names is not None:
            planner_names.update(zip(sorted_names, descriptive_names, strict=True))
            reserved_names.update(descriptive_names)
            continue
        for index, name in enumerate(sorted_names, start=1):
            planner_names[name] = f"{token}_{index}"
    return planner_names


def _task_annotations_by_body(world, robot):
    robot_bodies = set(robot.bodies)
    grouped_annotations = annotations_by_body(world)
    annotations_by_root = {}

    for root, annotations in grouped_annotations.items():
        if root in robot_bodies:
            continue
        kept = [
            annotation
            for annotation in annotations
            if not isinstance(annotation, (Room, *SKIPPED_ANNOTATION_TYPES))
        ]
        if kept:
            annotations_by_root[root] = kept

    return annotations_by_root


def _entity_capabilities(annotations):
    is_container = any(
        isinstance(annotation, CONTAINER_TYPES) for annotation in annotations
    )
    is_surface = any(
        provides_supporting_surface(annotation) for annotation in annotations
    )
    is_furniture = any(
        isinstance(annotation, Furniture)
        or type(annotation) in FURNITURE_ANNOTATION_TYPES
        for annotation in annotations
    )
    is_openable = any(
        find_openable_handle(annotation) is not None for annotation in annotations
    )
    is_tableware = any(
        isinstance(annotation, TABLEWARE_TYPES) for annotation in annotations
    )
    is_door = any(isinstance(annotation, DOOR_TYPES) for annotation in annotations)
    is_object = is_tableware or not (
        is_container or is_surface or is_furniture or is_openable or is_door
    )

    return _EntityCapabilities(
        is_object=is_object,
        is_container=is_container,
        is_surface=is_surface,
        is_furniture=is_furniture,
        is_openable=is_openable,
    )


def _hidden_task_roots(annotations_by_root):
    hidden_annotations = set()
    hidden_roots = set()
    for root, annotations in annotations_by_root.items():
        for annotation in annotations:
            if isinstance(annotation, HasDoors):
                hidden_annotations.update(annotation.doors)
            if (
                isinstance(annotation, HasDrawers)
                and annotation.drawers
                and type(annotation) not in FURNITURE_ANNOTATION_TYPES
            ):
                # semDT can label a cabinet shell as Wardrobe. Its concrete
                # Drawer children are the task-relevant containers.
                hidden_roots.add(root)

    hidden_roots.update(
        root
        for root, annotations in annotations_by_root.items()
        if any(annotation in hidden_annotations for annotation in annotations)
    )
    return hidden_roots


@dataclass
class PlaceIndex:
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
        self.storage_locations = _storage_locations(
            self.containers, self.surfaces, self.annotations_by_root
        )

    def locations_of(self, object_root):
        places = self.storage_locations.get(object_root, [])
        if places:
            return list(places)
        place = self._fallback_location(object_root)
        return [place] if place is not None else []

    def place_name(self, place_root):
        return self.place_names.get(place_root, str(place_root.name))

    def _fallback_location(self, object_root):
        container = _structural_container(object_root, self.container_roots)
        if container is not None:
            return self.place_name(container)

        container = self._tightest_container(object_root)
        if container is not None:
            return self.place_name(container)

        surface = _best_supporting_body(object_root, self.surface_roots)
        if surface is not None:
            return self.place_name(surface)
        return None

    def _tightest_container(self, object_root):
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


def _storage_locations(containers, surfaces, annotations_by_root):
    locations = {}
    for place_name, place_root in (*containers, *surfaces):
        for annotation in annotations_by_root.get(place_root, []):
            if not isinstance(annotation, HasStorageSpace):
                continue
            for stored_object in annotation.objects:
                object_root = stored_object.root
                if object_root is place_root:
                    continue
                place_names = locations.setdefault(object_root, [])
                if place_name not in place_names:
                    place_names.append(place_name)
    return locations


def _structural_container(object_body, container_roots):
    parent = object_body.parent_kinematic_structure_entity
    while parent is not None:
        if any(parent is container for container in container_roots):
            return parent
        parent = parent.parent_kinematic_structure_entity
    return None


def is_inside_or_attached(object_body, container_body):
    if _structural_container(object_body, [container_body]) is not None:
        return True
    return (
        InsideOf(object_body, container_body).compute_containment_ratio()
        > CONTAINMENT_THRESHOLD
    )


def is_at_location(object_body, location_body):
    return is_inside_or_attached(object_body, location_body) or is_supported_by(
        object_body, location_body
    )


def _kinematic_depth(body):
    depth = 0
    seen = set()
    parent = body.parent_kinematic_structure_entity
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        depth += 1
        parent = parent.parent_kinematic_structure_entity
    return depth


def _best_supporting_body(object_body, surface_roots):
    # is_supported_by may match nested bodies. The deepest match is the concrete
    # surface directly below the object.
    best_surface = None
    best_key = None
    for surface_body in surface_roots:
        if surface_body is object_body or not is_supported_by(
            object_body, surface_body
        ):
            continue

        key = (_kinematic_depth(surface_body), str(surface_body.name))
        if best_key is None or key > best_key:
            best_surface = surface_body
            best_key = key
    return best_surface


def supporting_surface_of(world, object_body):
    surfaces = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(HasSupportingSurface)
        if provides_supporting_surface(annotation)
    ]
    surface_body = _best_supporting_body(
        object_body,
        [surface.root for surface in surfaces],
    )
    for surface in surfaces:
        if surface.root is surface_body:
            return surface
    return None
