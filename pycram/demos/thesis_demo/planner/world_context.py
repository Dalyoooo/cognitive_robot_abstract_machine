import math
from collections import Counter
from dataclasses import dataclass

from semantic_digital_twin.reasoning.predicates import InsideOf, is_supported_by
from semantic_digital_twin.semantic_annotations.mixins import (
    HasCaseAsRootBody,
    HasDoors,
    HasDrawers,
    HasHandle,
    HasStorageSpace,
    HasSupportingSurface,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Dishwasher,
    Door,
    DoubleDoor,
    Floor,
    Fridge,
    Furniture,
    Handle,
    Hinge,
    Oven,
    Plate,
    Room,
    Sink,
    Slider,
    Wall,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from ..validation.schema import CONTEXT_KEYS


def unique(values):
    return list(dict.fromkeys(values))


SKIP_TYPES = (Handle, Hinge, Slider, Wall, Floor)
CONTAINER_TYPES = (HasCaseAsRootBody,)
FURNITURE_TYPES = (Furniture, Sink, Oven)
TABLEWARE_TYPES = (Plate, Bowl)
DOOR_TYPES = (Door, DoubleDoor)
CASE_APPLIANCE_TYPES = (Fridge, Oven, Dishwasher)
CONTAINMENT_THRESHOLD = 0.9


@dataclass(frozen=True)
class _EntityCapabilities:
    is_object: bool
    is_container: bool
    is_surface: bool
    is_furniture: bool
    is_openable: bool


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
    try:
        lower = connection.dof.limits.lower.position
        upper = connection.dof.limits.upper.position
        if lower is None or upper is None:
            return False
        if not math.isfinite(lower) or not math.isfinite(upper):
            return False
        return abs(upper - lower) < 1e-9
    except (AttributeError, TypeError):
        return False


def find_openable_handle(annotation):
    for handle in _handles(annotation):
        root = getattr(handle, "root", None)
        if root is None:
            continue
        try:
            connection = root.get_first_parent_connection_of_type(ActiveConnection1DOF)
        except (AttributeError, TypeError, ValueError):
            continue
        if not _is_locked(connection):
            return handle
    return None


def primary_annotation(annotations):
    """Return the most derived annotation, breaking ties by class name."""
    return max(
        annotations,
        key=lambda item: (len(type(item).mro()), type(item).__name__),
    )


def provides_supporting_surface(annotation):
    """Return whether an annotation carries a usable supporting surface."""
    return (
        isinstance(annotation, HasSupportingSurface)
        and annotation.supporting_surface is not None
    )


def planner_id_map(names):
    raw_names = [str(name) for name in names]
    short_names = [name.rsplit("/", 1)[-1] for name in raw_names]
    short_name_counts = Counter(short_names)
    planner_ids = {}
    for raw_name, short_name in zip(raw_names, short_names):
        if short_name_counts[short_name] == 1:
            planner_ids[raw_name] = short_name
        else:
            planner_ids[raw_name] = raw_name
    return planner_ids


def planner_ids_for(world):
    rooms = world.get_semantic_annotations_by_type(Room)
    names = [body.name for body in world.bodies] + [room.name for room in rooms]
    return planner_id_map(names)


def _collect_annotations(world, robot):
    """Group task-relevant annotations by root body.

    Rooms, structural parts (handles, hinges, sliders, walls, floors) and
    anything on the robot are skipped.
    """
    robot_bodies = set(robot.bodies)
    annotations_by_root = {}

    for annotation in world.semantic_annotations:
        if isinstance(annotation, (Room, *SKIP_TYPES)):
            continue

        root = getattr(annotation, "root", None)
        if root is None or root in robot_bodies:
            continue
        annotations_by_root.setdefault(root, []).append(annotation)

    return annotations_by_root


def _entity_capabilities(annotations):
    is_container = any(
        isinstance(annotation, CONTAINER_TYPES) for annotation in annotations
    )
    is_surface = any(
        provides_supporting_surface(annotation) for annotation in annotations
    )
    is_furniture = any(
        isinstance(annotation, FURNITURE_TYPES) for annotation in annotations
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


def classify_world(world, robot):
    """Project semDT annotations into the flat planner context schema."""
    rooms = world.get_semantic_annotations_by_type(Room)
    annotations_by_root = _collect_annotations(world, robot)
    planner_ids = planner_ids_for(world)

    hidden_roots = _appliance_door_roots(annotations_by_root)
    containers = []
    surfaces = []
    furniture = []
    openables = []
    objects = []
    types = {}

    for room in rooms:
        types[planner_ids[str(room.name)]] = type(room).__name__

    for root, annotations in annotations_by_root.items():
        if root in hidden_roots:
            continue

        name = planner_ids[str(root.name)]
        types[name] = type(primary_annotation(annotations)).__name__
        capabilities = _entity_capabilities(annotations)

        if capabilities.is_object:
            objects.append((name, root))
        if capabilities.is_container:
            containers.append((name, root))
        if capabilities.is_surface:
            surfaces.append((name, root))
        if capabilities.is_furniture:
            furniture.append(name)
        if capabilities.is_openable:
            openables.append(name)

    place_index = PlaceIndex(containers, surfaces, annotations_by_root)
    object_locations = {}
    for name, root in objects:
        places = place_index.locations_of(root)
        if places:
            object_locations[name] = places

    result = dict(
        objects=unique([name for name, _ in objects]),
        object_locations=object_locations,
        surfaces=unique([name for name, _ in surfaces]),
        containers=unique([name for name, _ in containers]),
        openables=unique(openables),
        furniture=unique(furniture),
        rooms=unique([planner_ids[str(room.name)] for room in rooms]),
        types=types,
    )
    assert set(result) == set(CONTEXT_KEYS), "classify_world keys {} != {}".format(
        set(result), set(CONTEXT_KEYS)
    )
    return result


def _appliance_door_roots(annotations_by_root):
    appliance_roots = set()
    for root, annotations in annotations_by_root.items():
        for annotation in annotations:
            if isinstance(annotation, CASE_APPLIANCE_TYPES):
                appliance_roots.add(root)
                break

    door_roots = set()
    for root, annotations in annotations_by_root.items():
        if _parent_body(root) not in appliance_roots:
            continue
        for annotation in annotations:
            if isinstance(annotation, DOOR_TYPES):
                door_roots.add(root)
                break
    return door_roots


class PlaceIndex:
    def __init__(self, containers, surfaces, annotations_by_root):
        self.container_roots = [root for _, root in containers]
        self.surface_roots = [root for _, root in surfaces]
        self.place_names = {root: name for name, root in (*containers, *surfaces)}
        self.storage_locations = _verified_storage_locations(
            containers, surfaces, annotations_by_root
        )

    def locations_of(self, object_root):
        """Return the planner IDs of the places holding one object."""
        places = self.storage_locations.get(object_root, [])
        if places:
            return list(places)
        place = self._fallback_location(object_root)
        return [place] if place is not None else []

    def place_name(self, place_root):
        """Return the planner ID of one container or surface body."""
        return self.place_names.get(place_root, str(place_root.name))

    def _fallback_location(self, object_root):
        """Return the best place for an object without a storage entry."""
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
        """Return the container holding the largest share of the object."""
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


def _verified_storage_locations(containers, surfaces, annotations_by_root):
    """Index semDT storage members that still satisfy their spatial relation.

    A place's ``HasStorageSpace`` annotation lists the objects it stores; each
    is kept only if it is still inside/attached (containers) or supported
    (surfaces).
    """
    locations = {}
    relation_checks = (
        (containers, is_inside_or_attached),
        (surfaces, is_supported_by),
    )
    for places, relation_holds in relation_checks:
        for place_name, place_root in places:
            for annotation in annotations_by_root.get(place_root, []):
                if not isinstance(annotation, HasStorageSpace):
                    continue
                for stored_object in annotation.objects:
                    object_root = getattr(stored_object, "root", None)
                    if object_root is None or object_root is place_root:
                        continue
                    if not relation_holds(object_root, place_root):
                        continue
                    place_names = locations.setdefault(object_root, [])
                    if place_name not in place_names:
                        place_names.append(place_name)
    return locations


def _structural_container(obj_body, container_roots):
    """Return the nearest container ancestor in the kinematic tree."""
    parent = _parent_body(obj_body)
    while parent is not None:
        if any(parent is container for container in container_roots):
            return parent
        parent = _parent_body(parent)
    return None


def is_inside_or_attached(obj_body, container_body):
    """Return whether structural attachment or geometric containment holds."""
    if _structural_container(obj_body, [container_body]) is not None:
        return True
    return (
        InsideOf(obj_body, container_body).compute_containment_ratio()
        > CONTAINMENT_THRESHOLD
    )


def is_at_location(obj_body, location_body):
    """Return whether an object is inside, attached to, or supported by a place."""
    return is_inside_or_attached(obj_body, location_body) or is_supported_by(
        obj_body, location_body
    )


def _parent_body(body):
    """Return the parent kinematic body when one exists."""
    try:
        return body.parent_kinematic_structure_entity
    except (AttributeError, TypeError):
        return None


def _kinematic_depth(body):
    """Return a body's depth in the kinematic structure."""
    depth = 0
    seen = set()
    parent = _parent_body(body)
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        depth += 1
        parent = _parent_body(parent)
    return depth


def _best_supporting_body(obj_body, surface_roots):
    """Return the deepest supporting surface body below an object."""
    best_surface = None
    best_key = None
    for surface_body in surface_roots:
        if surface_body is obj_body or not is_supported_by(obj_body, surface_body):
            continue

        key = (_kinematic_depth(surface_body), str(surface_body.name))
        if best_key is None or key > best_key:
            best_surface = surface_body
            best_key = key
    return best_surface


def supporting_surface_of(world, obj_body):
    """Return the most specific surface annotation supporting an object."""
    surfaces = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(HasSupportingSurface)
        if provides_supporting_surface(annotation)
    ]
    surface_body = _best_supporting_body(
        obj_body,
        [surface.root for surface in surfaces],
    )
    for surface in surfaces:
        if surface.root is surface_body:
            return surface
    return None
