import importlib
import sys
import types
from enum import Enum
from pathlib import Path

import pytest

_DEMOS_DIR = Path(__file__).resolve().parents[1] / "pycram" / "demos"
if str(_DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMOS_DIR))


class _PrefixedName:
    def __init__(self, name, prefix):
        self.name = name
        self.prefix = prefix

    def __str__(self):
        return f"{self.prefix}/{self.name}"


class _Body:
    """Attribute container that hashes by identity, like semDT world entities."""

    def __init__(self, **attributes):
        for key, value in attributes.items():
            setattr(self, key, value)


def _fake_world(bodies, rooms=()):
    return types.SimpleNamespace(
        bodies=bodies,
        get_semantic_annotations_by_type=lambda _type: list(rooms),
    )


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_grounding(monkeypatch):
    Room = type("Room", (), {})

    class SemanticAnnotation:
        pass

    class HasRootBody:
        pass

    class HasStorageSpace(HasRootBody):
        pass

    class HasSupportingSurface(HasStorageSpace):
        pass

    class Point3:
        def __init__(self, x, y, z, reference_frame=None):
            self.x = x
            self.y = y
            self.z = z
            self.reference_frame = reference_frame

    def find_handle(annotation):
        """Return a handle from the fake annotation."""
        handle = getattr(annotation, "handle", None)
        if handle is not None:
            return handle
        for child_attr in ("doors", "drawers"):
            for child in getattr(annotation, child_attr, []) or []:
                handle = getattr(child, "handle", None)
                if handle is not None:
                    return handle
        return None

    def planner_ids_for(world):
        """Return stable planner IDs for fake bodies and rooms."""
        rooms = world.get_semantic_annotations_by_type(Room)
        names = [str(body.name) for body in world.bodies] + [
            str(room.name) for room in rooms
        ]
        short_names = [name.rsplit("/", 1)[-1] for name in names]
        return {
            name: short_name if short_names.count(short_name) == 1 else name
            for name, short_name in zip(names, short_names)
        }

    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.reasoning.predicates",
        _module(
            "semantic_digital_twin.reasoning.predicates",
            Behind=object,
            InFrontOf=object,
            InsideOf=object,
            LeftOf=object,
            RightOf=object,
            is_supported_by=lambda _object, _place: False,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.semantic_annotations.mixins",
        _module(
            "semantic_digital_twin.semantic_annotations.mixins",
            HasRootBody=HasRootBody,
            HasStorageSpace=HasStorageSpace,
            HasSupportingSurface=HasSupportingSurface,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.semantic_annotations.semantic_annotations",
        _module(
            "semantic_digital_twin.semantic_annotations.semantic_annotations",
            Room=Room,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.spatial_types.spatial_types",
        _module(
            "semantic_digital_twin.spatial_types.spatial_types",
            Point3=Point3,
            Pose=type("Pose", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.world_description.world_entity",
        _module(
            "semantic_digital_twin.world_description.world_entity",
            SemanticAnnotation=SemanticAnnotation,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "thesis_demo.world_context",
        _module(
            "thesis_demo.world_context",
            find_openable_handle=find_handle,
            is_at_location=lambda _object, _place: False,
            is_inside_or_attached=lambda _object, _place: False,
            planner_ids_for=planner_ids_for,
            primary_annotation=lambda annotations: annotations[0],
            provides_supporting_surface=lambda annotation: (
                getattr(annotation, "supporting_surface", None) is not None
            ),
            supporting_surface_of=lambda _world, _object: None,
        ),
    )
    sys.modules.pop("thesis_demo.grounding", None)
    return importlib.import_module("thesis_demo.grounding")


def test_source_mismatch_does_not_fall_back_to_first_object(monkeypatch):
    """Verify that source mismatch does not fall back to first object."""
    grounding_module = _load_grounding(monkeypatch)
    candidate = types.SimpleNamespace(name="cup_1")
    place = types.SimpleNamespace(name="table")
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {"cup_1": candidate, "table": place}

    with pytest.raises(grounding_module.GroundingError) as error:
        grounding.body("cup_1", "table")

    assert "not attached, inside, or supported" in str(error.value)


def test_source_uses_the_same_location_check_as_world_context(monkeypatch):
    """Verify that source uses the same location check as world context."""
    grounding_module = _load_grounding(monkeypatch)
    candidate = types.SimpleNamespace(name="large_box")
    drawer = types.SimpleNamespace(name="drawer")
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {"large_box": candidate, "drawer": drawer}
    monkeypatch.setattr(
        grounding_module,
        "is_at_location",
        lambda obj, place: obj is candidate and place is drawer,
    )

    assert grounding.body("large_box", "drawer") is candidate


def test_unknown_canonical_body_is_rejected(monkeypatch):
    """Verify that unknown canonical body is rejected."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {}

    with pytest.raises(grounding_module.GroundingError) as error:
        grounding.body("cup")

    assert "canonical body ID" in str(error.value)


def test_body_prefers_an_exact_canonical_id(monkeypatch):
    """Verify that body prefers an exact canonical id."""
    grounding_module = _load_grounding(monkeypatch)
    left = _Body(name=_PrefixedName("cup", "left"))
    right = _Body(name=_PrefixedName("cup", "right"))
    world = _fake_world([left, right])
    grounding = grounding_module.Grounding(world)

    assert grounding.body("right/cup") is right


def test_body_uses_short_id_for_a_unique_prefixed_name(monkeypatch):
    """Verify that body uses short id for a unique prefixed name."""
    grounding_module = _load_grounding(monkeypatch)
    drawer = _Body(name=_PrefixedName("drawer", "apartment"))
    grounding = grounding_module.Grounding(_fake_world([drawer]))

    assert grounding.body("drawer") is drawer
    assert grounding.body_id(drawer) == "drawer"

    with pytest.raises(grounding_module.GroundingError):
        grounding.body("apartment/drawer")


def test_unprefixed_name_is_not_used_as_a_legacy_fallback(monkeypatch):
    """Verify that unprefixed name is not used as a legacy fallback."""
    grounding_module = _load_grounding(monkeypatch)
    left = _Body(name=_PrefixedName("cup", "left"))
    right = _Body(name=_PrefixedName("cup", "right"))
    world = _fake_world([left, right])
    grounding = grounding_module.Grounding(world)

    with pytest.raises(grounding_module.GroundingError) as error:
        grounding.body("cup")

    assert "canonical body ID" in str(error.value)


def test_resolve_annotation_prefers_exact_canonical_body(monkeypatch):
    """Verify that resolve annotation prefers exact canonical body."""
    grounding_module = _load_grounding(monkeypatch)
    body = types.SimpleNamespace(name=_PrefixedName("drawer", "kitchen"))
    annotation = types.SimpleNamespace(root=body)
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {"kitchen/drawer": body}
    grounding.rooms_by_id = {}
    grounding._annotations_on = lambda candidate: (
        [annotation] if candidate is body else []
    )

    assert grounding.resolve_annotation("kitchen/drawer") is annotation


def test_resolve_annotation_accepts_a_short_unique_room_id(monkeypatch):
    """Verify that resolve annotation accepts a short unique room id."""
    grounding_module = _load_grounding(monkeypatch)
    room = types.SimpleNamespace(name=_PrefixedName("kitchen", "apartment"))
    grounding = grounding_module.Grounding(_fake_world([], [room]))

    assert grounding.resolve_annotation("kitchen") is room


def test_handle_prefers_exact_canonical_body(monkeypatch):
    """Verify that handle prefers exact canonical body."""
    grounding_module = _load_grounding(monkeypatch)
    body = types.SimpleNamespace(name=_PrefixedName("drawer", "kitchen"))
    handle = types.SimpleNamespace(root="exact_handle")
    annotation = types.SimpleNamespace(root=body, handle=handle)
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {"kitchen/drawer": body}
    grounding._annotations_on = lambda candidate: (
        [annotation] if candidate is body else []
    )

    assert grounding.handle("kitchen/drawer") == "exact_handle"


def test_handle_resolves_container_by_canonical_id(monkeypatch):
    """Verify that handle resolves container by canonical id."""
    grounding_module = _load_grounding(monkeypatch)
    body = types.SimpleNamespace(name="kitchen/fridge")
    handle_body = types.SimpleNamespace(root="fridge_handle_root")
    annotation = types.SimpleNamespace(root=body, handle=handle_body)
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {"kitchen/fridge": body}
    grounding._annotations_on = lambda candidate: (
        [annotation] if candidate is body else []
    )

    assert grounding.handle("kitchen/fridge") == "fridge_handle_root"


def test_handle_resolves_container_by_instance_name(monkeypatch):
    """Verify that handle resolves container by instance name."""
    grounding_module = _load_grounding(monkeypatch)
    handle_body = types.SimpleNamespace(root="fridge_handle_root")
    body = types.SimpleNamespace(name="fridge_main")
    annotation = types.SimpleNamespace(root=body, handle=handle_body)
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {"fridge_main": body}
    grounding._annotations_on = lambda queried: [annotation] if queried is body else []

    assert grounding.handle("fridge_main") == "fridge_handle_root"


def test_annotations_on_reads_world_annotations_rooted_at_the_body(monkeypatch):
    """Verify that annotations on reads world annotations rooted at the body."""
    grounding_module = _load_grounding(monkeypatch)

    class Body:
        name = "cup"

        @staticmethod
        def get_semantic_annotations_by_type(_type):
            """Return fake semantic annotations for a requested type."""
            return []

    body = Body()
    annotation = types.SimpleNamespace(root=body)
    calls = []

    def annotations(annotation_type):
        """Return the fake annotations rooted at a body."""
        calls.append(annotation_type)
        if annotation_type is grounding_module.Room:
            return []
        return [annotation]

    world = types.SimpleNamespace(
        bodies=[body],
        get_semantic_annotations_by_type=annotations,
    )
    grounding = grounding_module.Grounding(world)

    assert (
        body.get_semantic_annotations_by_type(grounding_module.SemanticAnnotation) == []
    )
    assert grounding._annotations_on(body) == [annotation]
    assert calls == [
        grounding_module.Room,
        grounding_module.Room,
        grounding_module.SemanticAnnotation,
    ]


def test_handle_rejects_container_without_articulated_handle(monkeypatch):
    """Verify that handle rejects container without articulated handle."""
    grounding_module = _load_grounding(monkeypatch)
    body = types.SimpleNamespace(name="drawer_left")
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {"drawer_left": body}
    grounding._annotations_on = lambda _body: [types.SimpleNamespace(root=body)]

    with pytest.raises(grounding_module.GroundingError) as error:
        grounding.handle("drawer_left")

    assert "No articulated handle" in str(error.value)


def _pose_grounding(grounding_module):
    """A Grounding with identity world transform and tuple poses for pose tests."""
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding._world_pose = lambda x, y, z: (x, y, z)
    grounding.world = types.SimpleNamespace(
        root=object(), transform=lambda point, root: point
    )
    return grounding


def test_inside_pose_allows_an_object_larger_than_the_container(monkeypatch):
    """Verify that inside pose allows an object larger than the container."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    container = types.SimpleNamespace(
        name="tiny_drawer",
        combined_mesh=types.SimpleNamespace(bounds=((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))),
    )
    obj = types.SimpleNamespace(
        name="huge_box",
        combined_mesh=types.SimpleNamespace(extents=(1.0, 1.0, 1.0)),
    )

    pose = grounding._inside_pose(container, obj)

    assert pose == (0.05, 0.05, 0.52)


def test_inside_pose_uses_container_local_bounds_for_a_fitting_object(monkeypatch):
    """Verify that inside pose uses container local bounds for a fitting object."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    container = types.SimpleNamespace(
        name="drawer",
        combined_mesh=types.SimpleNamespace(bounds=((0.0, 0.0, 0.0), (0.5, 0.4, 0.3))),
    )
    obj = types.SimpleNamespace(
        name="cup",
        combined_mesh=types.SimpleNamespace(extents=(0.1, 0.1, 0.1)),
    )

    pose = grounding._inside_pose(container, obj)

    assert pose == (0.25, 0.2, 0.07)


def test_place_pose_rejects_a_surface_without_a_free_spot(monkeypatch):
    """Verify that place pose rejects a surface without a free spot."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    grounding._annotations_on = lambda _body: []
    area = types.SimpleNamespace(
        min_point=types.SimpleNamespace(x=0.0, y=0.0, z=1.0),
        max_point=types.SimpleNamespace(x=2.0, y=2.0, z=1.0),
    )
    surface = types.SimpleNamespace(
        supporting_surface=types.SimpleNamespace(area=area),
        sample_points_from_surface=lambda body_to_sample_for=None: [],
    )
    obj = types.SimpleNamespace(
        name="cup", combined_mesh=types.SimpleNamespace(extents=(0.1, 0.1, 0.2))
    )

    surface.root = types.SimpleNamespace(name="full_table")

    grounding._supporting_surface_annotation = lambda _label: surface

    with pytest.raises(grounding_module.GroundingError, match="no free placement"):
        grounding.place_pose("full_table", obj, "on")


def test_place_pose_uses_sampled_point_when_available(monkeypatch):
    """A free sampled spot is used directly, not the centre fallback."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    grounding._annotations_on = lambda _body: []
    surface = types.SimpleNamespace(
        sample_points_from_surface=lambda body_to_sample_for=None: [
            types.SimpleNamespace(x=0.5, y=0.7, z=1.0)
        ]
    )
    obj = types.SimpleNamespace(
        name="cup", combined_mesh=types.SimpleNamespace(extents=(0.1, 0.1, 0.2))
    )

    grounding._supporting_surface_annotation = lambda _label: surface

    pose = grounding.place_pose("table", obj, "on")

    assert grounding_module.SUPPORT_DETECTION_OVERLAP == 0.01
    assert pose == (0.5, 0.7, 1.0 - grounding_module.SUPPORT_DETECTION_OVERLAP)


def test_place_poses_preserves_all_sampled_points(monkeypatch):
    """All semDT samples remain available as ordered placement candidates."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    grounding._annotations_on = lambda _body: []
    surface = types.SimpleNamespace(
        sample_points_from_surface=lambda body_to_sample_for=None: [
            types.SimpleNamespace(x=0.5, y=0.7, z=1.0),
            types.SimpleNamespace(x=0.8, y=0.9, z=1.0),
        ]
    )
    obj = types.SimpleNamespace(
        name="cup", combined_mesh=types.SimpleNamespace(extents=(0.1, 0.1, 0.2))
    )
    grounding._supporting_surface_annotation = lambda _label: surface

    poses = grounding.place_poses("table", obj, "on")

    overlap = grounding_module.SUPPORT_DETECTION_OVERLAP
    assert poses == [(0.5, 0.7, 1.0 - overlap), (0.8, 0.9, 1.0 - overlap)]


def test_place_pose_without_supporting_surface_raises_grounding_error(monkeypatch):
    """
    Verify that a missing supporting surface raises a GroundingError.
    """
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    grounding._annotations_on = lambda _body: []
    surface = types.SimpleNamespace(
        root=types.SimpleNamespace(name="broken_table"),
        supporting_surface=None,
        sample_points_from_surface=lambda body_to_sample_for=None: [],
    )
    obj = types.SimpleNamespace(
        name="cup", combined_mesh=types.SimpleNamespace(extents=(0.1, 0.1, 0.2))
    )

    grounding._supporting_surface_annotation = lambda _label: surface

    with pytest.raises(grounding_module.GroundingError):
        grounding.place_pose("broken_table", obj, "on")


def test_place_pose_dispatches_directional_relation(monkeypatch):
    """Verify that place pose dispatches directional relation."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    seen = {}
    grounding.directional_pose = lambda label, body, relation: seen.setdefault(
        "args", (label, body, relation)
    )
    body = object()

    grounding.place_pose("winebottle", body, "left_of")

    assert seen["args"] == ("winebottle", body, "left_of")


def test_place_pose_selects_capability_on_multi_annotation_body(monkeypatch):
    """Verify that place pose selects capability on multi annotation body."""
    grounding_module = _load_grounding(monkeypatch)
    target_body = types.SimpleNamespace(name=_PrefixedName("table", "kitchen"))
    unusable_surface = types.SimpleNamespace(root=target_body, supporting_surface=None)
    surface_annotation = types.SimpleNamespace(
        root=target_body, supporting_surface=object()
    )
    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.bodies_by_id = {"kitchen/table": target_body}
    grounding._annotations_on = lambda _body: [
        unusable_surface,
        surface_annotation,
    ]
    grounding._surface_points = lambda surface, _obj: [surface]
    grounding._placement_pose = lambda point: point

    result = grounding.place_pose("kitchen/table", object(), "on")

    assert result is surface_annotation


def test_directional_pose_picks_nearest_matching_point(monkeypatch):
    """Verify that directional pose picks nearest matching point."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    grounding.robot = types.SimpleNamespace(
        root=types.SimpleNamespace(global_transform="view")
    )
    reference_body = types.SimpleNamespace(
        name="winebottle",
        global_pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=1.0, y=1.0, z=0.5)
        ),
    )
    grounding.body = lambda _label: reference_body
    grounding._annotations_on = lambda _body: []
    monkeypatch.setattr(
        grounding_module,
        "supporting_surface_of",
        lambda _world, _body: types.SimpleNamespace(
            sample_points_from_surface=lambda body_to_sample_for=None: [
                types.SimpleNamespace(x=5.0, y=1.0, z=0.6),
                types.SimpleNamespace(x=2.0, y=1.0, z=0.6),
                types.SimpleNamespace(x=0.0, y=1.0, z=0.6),
            ]
        ),
    )

    predicate_calls = []

    class RightOf:
        def __init__(self, point, other, point_of_view):
            self.point = point
            predicate_calls.append((point, other, point_of_view))

        def __call__(self):
            return self.point.x > 1.0

    monkeypatch.setitem(grounding_module._DIRECTIONAL_PREDICATES, "right_of", RightOf)

    pose = grounding.directional_pose("winebottle", object(), "right_of")

    assert pose == (2.0, 1.0, 0.6 - grounding_module.SUPPORT_DETECTION_OVERLAP)
    assert all(call[2] == "view" for call in predicate_calls)


def test_directional_pose_raises_without_matching_point(monkeypatch):
    """Verify that directional pose raises without matching point."""
    grounding_module = _load_grounding(monkeypatch)
    grounding = _pose_grounding(grounding_module)
    grounding.robot = types.SimpleNamespace(
        root=types.SimpleNamespace(global_transform="view")
    )
    grounding.body = lambda _label: types.SimpleNamespace(
        global_pose=types.SimpleNamespace(
            position=types.SimpleNamespace(x=1.0, y=1.0, z=0.5)
        )
    )
    grounding._annotations_on = lambda _body: []
    monkeypatch.setattr(
        grounding_module,
        "supporting_surface_of",
        lambda _world, _body: types.SimpleNamespace(
            sample_points_from_surface=lambda body_to_sample_for=None: [
                types.SimpleNamespace(x=2.0, y=1.0, z=0.6)
            ]
        ),
    )

    class NeverMatches:
        def __init__(self, point, other, point_of_view):
            pass

        def __call__(self):
            return False

    monkeypatch.setitem(
        grounding_module._DIRECTIONAL_PREDICATES, "right_of", NeverMatches
    )

    with pytest.raises(grounding_module.GroundingError, match="no spot"):
        grounding.directional_pose("winebottle", object(), "right_of")


def test_register_placement_adds_object_to_supporting_surface(monkeypatch):
    """Verify that a successful placement updates semDT storage."""
    grounding_module = _load_grounding(monkeypatch)
    object_body = object()
    surface_body = object()
    object_annotation = grounding_module.HasRootBody()
    old_storage = grounding_module.HasStorageSpace()
    old_storage.objects = [object_annotation]
    storage = grounding_module.HasStorageSpace()
    storage.root = surface_body
    storage.objects = [object_annotation, object_annotation]
    storage.supporting_surface = object()
    added = []

    def add_object(annotation):
        added.append(annotation)
        storage.objects.append(annotation)

    storage.add_object = add_object

    class Modification:
        def __enter__(self):
            return self

        def __exit__(self, error_type, error, traceback):
            return False

    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.world = types.SimpleNamespace(
        modify_world=Modification,
        get_semantic_annotations_by_type=lambda _type: [old_storage, storage],
    )
    grounding.body = lambda label: object_body if label == "cup" else surface_body
    grounding._annotations_on = lambda body: (
        [object_annotation] if body is object_body else [storage]
    )
    monkeypatch.setattr(grounding_module, "is_supported_by", lambda _obj, _place: True)

    grounding.register_placement("cup", "table", "on")

    assert added == [object_annotation]
    assert old_storage.objects == []
    assert storage.objects == [object_annotation]


def test_remove_from_storage_clears_all_memberships(monkeypatch):
    """Verify that a successful pickup clears stale semDT storage lists."""
    grounding_module = _load_grounding(monkeypatch)
    object_body = object()
    object_annotation = grounding_module.HasRootBody()
    first_storage = grounding_module.HasStorageSpace()
    first_storage.objects = [object_annotation, object_annotation]
    second_storage = grounding_module.HasStorageSpace()
    second_storage.objects = [object_annotation]

    class Modification:
        def __enter__(self):
            return self

        def __exit__(self, error_type, error, traceback):
            return False

    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.world = types.SimpleNamespace(
        modify_world=Modification,
        get_semantic_annotations_by_type=lambda _type: [
            first_storage,
            second_storage,
        ],
    )
    grounding.body = lambda _label: object_body
    grounding._annotations_on = lambda _body: [object_annotation]

    grounding.remove_from_storage("cup")

    assert first_storage.objects == []
    assert second_storage.objects == []


def test_register_placement_rejects_unsatisfied_relation(monkeypatch):
    """Verify that storage is unchanged when placement validation fails."""
    grounding_module = _load_grounding(monkeypatch)
    object_body = object()
    drawer_body = object()
    object_annotation = grounding_module.HasRootBody()
    storage = grounding_module.HasStorageSpace()
    storage.root = drawer_body
    storage.objects = []

    grounding = grounding_module.Grounding.__new__(grounding_module.Grounding)
    grounding.body = lambda label: object_body if label == "cup" else drawer_body
    grounding._annotations_on = lambda body: (
        [object_annotation] if body is object_body else [storage]
    )
    monkeypatch.setattr(
        grounding_module,
        "is_inside_or_attached",
        lambda _obj, _place: False,
    )

    with pytest.raises(RuntimeError, match="does not satisfy"):
        grounding.register_placement("cup", "drawer", "inside")

    assert storage.objects == []


