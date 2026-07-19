"""Unit tests for semDT capability-based world classification."""

import importlib
import sys
import types
from pathlib import Path

import pytest

_DEMOS_DIR = Path(__file__).resolve().parents[1] / "pycram" / "demos"
if str(_DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMOS_DIR))


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _Namespace:
    def __init__(self, **attributes):
        for key, value in attributes.items():
            setattr(self, key, value)


class _PrefixedName:
    def __init__(self, name, prefix):
        self.name = name
        self.prefix = prefix

    def __str__(self):
        return f"{self.prefix}/{self.name}"


def _load_world_context(monkeypatch):
    """Import world_context with small semDT substitutes."""
    HasStorageSpace = type("HasStorageSpace", (), {})
    HasSupportingSurface = type("HasSupportingSurface", (HasStorageSpace,), {})
    HasCaseAsRootBody = type("HasCaseAsRootBody", (), {})
    HasDoors = type("HasDoors", (), {"doors": ()})
    HasDrawers = type("HasDrawers", (), {"drawers": ()})
    HasHandle = type("HasHandle", (), {"handle": None})

    Room = type("Room", (), {})
    Furniture = type("Furniture", (), {})
    Handle = type("Handle", (), {})
    Hinge = type("Hinge", (), {})
    Slider = type("Slider", (), {})
    Wall = type("Wall", (), {})
    Floor = type("Floor", (), {})

    Sink = type("Sink", (), {})
    Oven = type("Oven", (), {})
    DoubleDoor = type("DoubleDoor", (), {})
    Drawer = type(
        "Drawer", (Furniture, HasCaseAsRootBody, HasHandle, HasStorageSpace), {}
    )
    Door = type("Door", (HasHandle,), {})
    DoorWithType = type("DoorWithType", (HasHandle,), {})
    Fridge = type("Fridge", (Furniture, HasCaseAsRootBody, HasDoors, HasDrawers), {})
    Dishwasher = type("Dishwasher", (HasCaseAsRootBody, HasDoors, HasDrawers), {})
    Plate = type("Plate", (HasSupportingSurface,), {})
    Bowl = type("Bowl", (HasSupportingSurface,), {})

    Wardrobe = type(
        "Wardrobe", (Furniture, HasCaseAsRootBody, HasDoors, HasDrawers), {}
    )
    Cupboard = type("Cupboard", (Furniture, HasCaseAsRootBody, HasDoors), {})
    ShelfLayer = type("ShelfLayer", (HasSupportingSurface,), {})
    CounterTop = type("CounterTop", (Furniture, HasSupportingSurface), {})
    Milk = type("Milk", (), {})

    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.reasoning.predicates",
        _module(
            "semantic_digital_twin.reasoning.predicates",
            InsideOf=lambda _obj, _place: _Namespace(
                compute_containment_ratio=lambda: 0.0
            ),
            is_supported_by=lambda _obj, _place: False,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.semantic_annotations.mixins",
        _module(
            "semantic_digital_twin.semantic_annotations.mixins",
            HasCaseAsRootBody=HasCaseAsRootBody,
            HasDoors=HasDoors,
            HasDrawers=HasDrawers,
            HasHandle=HasHandle,
            HasStorageSpace=HasStorageSpace,
            HasSupportingSurface=HasSupportingSurface,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.semantic_annotations.semantic_annotations",
        _module(
            "semantic_digital_twin.semantic_annotations.semantic_annotations",
            Dishwasher=Dishwasher,
            Door=Door,
            DoorWithType=DoorWithType,
            DoubleDoor=DoubleDoor,
            Drawer=Drawer,
            Floor=Floor,
            Fridge=Fridge,
            Furniture=Furniture,
            Handle=Handle,
            Hinge=Hinge,
            Oven=Oven,
            Room=Room,
            Sink=Sink,
            Slider=Slider,
            Wall=Wall,
            Plate=Plate,
            Bowl=Bowl,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_digital_twin.world_description.connections",
        _module(
            "semantic_digital_twin.world_description.connections",
            ActiveConnection1DOF=type("ActiveConnection1DOF", (), {}),
        ),
    )
    sys.modules.pop("thesis_demo.world_context", None)
    module = importlib.import_module("thesis_demo.world_context")
    classes = {
        "Sink": Sink,
        "Oven": Oven,
        "DoubleDoor": DoubleDoor,
        "Drawer": Drawer,
        "Door": Door,
        "Fridge": Fridge,
        "Dishwasher": Dishwasher,
        "Plate": Plate,
        "Bowl": Bowl,
        "Wardrobe": Wardrobe,
        "Cupboard": Cupboard,
        "ShelfLayer": ShelfLayer,
        "CounterTop": CounterTop,
        "Milk": Milk,
        "Furniture": Furniture,
    }
    return module, classes


def _make(cls, name, surface=None, root=None):
    annotation = cls.__new__(cls)
    annotation.root = root or _Namespace(name=name)
    annotation.supporting_surface = surface
    annotation.objects = []
    return annotation


def _handle():
    root = _Namespace(name="handle")
    root.get_first_parent_connection_of_type = lambda _connection_type: object()
    return _Namespace(root=root)


def _handle_with_position_limits(lower, upper):
    connection = _Namespace(
        dof=_Namespace(
            limits=_Namespace(
                lower=_Namespace(position=lower),
                upper=_Namespace(position=upper),
            )
        )
    )
    root = _Namespace(name="handle")
    root.get_first_parent_connection_of_type = lambda _connection_type: connection
    return _Namespace(root=root)


def _classify(module, annotations):
    world = _Namespace(
        semantic_annotations=annotations,
        bodies=list(dict.fromkeys(annotation.root for annotation in annotations)),
        get_semantic_annotations_by_type=lambda annotation_type: [
            annotation
            for annotation in annotations
            if isinstance(annotation, annotation_type)
        ],
    )
    robot = _Namespace(bodies=[])
    return module.classify_world(world, robot)


def test_furniture_and_container_capabilities_overlap(monkeypatch):
    """Verify that furniture and container capabilities overlap."""
    module, classes = _load_world_context(monkeypatch)
    result = _classify(
        module,
        [
            _make(classes["Cupboard"], "cupboard_1"),
            _make(classes["Wardrobe"], "wardrobe_1"),
        ],
    )

    assert result["containers"] == ["cupboard_1", "wardrobe_1"]
    assert result["furniture"] == ["cupboard_1", "wardrobe_1"]


def test_supporting_surface_is_classified_independently(monkeypatch):
    """Verify that supporting surface is classified independently."""
    module, classes = _load_world_context(monkeypatch)
    result = _classify(
        module,
        [_make(classes["ShelfLayer"], "shelf_1", surface=object())],
    )

    assert result["surfaces"] == ["shelf_1"]


def test_plate_and_bowl_are_objects_and_surfaces(monkeypatch):
    """Verify that plate and bowl are objects and surfaces."""
    module, classes = _load_world_context(monkeypatch)
    result = _classify(
        module,
        [
            _make(classes["Plate"], "plate_1", surface=object()),
            _make(classes["Bowl"], "bowl_1", surface=object()),
        ],
    )

    assert result["objects"] == ["plate_1", "bowl_1"]
    assert result["surfaces"] == ["plate_1", "bowl_1"]


def test_object_surface_does_not_become_its_own_location(monkeypatch):
    """Verify that object surface does not become its own location."""
    module, classes = _load_world_context(monkeypatch)
    plate = _make(classes["Plate"], "plate_1", surface=object())
    table = _make(classes["CounterTop"], "table_1", surface=object())
    monkeypatch.setattr(module, "is_supported_by", lambda _obj, _surface: True)

    result = _classify(module, [plate, table])

    assert result["object_locations"] == {"plate_1": ["table_1"]}


def test_sink_and_oven_are_navigation_only_fixtures(monkeypatch):
    """Verify that sink and oven are furniture-only navigation targets."""
    module, classes = _load_world_context(monkeypatch)
    result = _classify(
        module,
        [
            _make(classes["Sink"], "sink"),
            _make(classes["Oven"], "oven"),
        ],
    )

    assert result["furniture"] == ["sink", "oven"]
    assert result["containers"] == []
    assert result["surfaces"] == []
    assert result["openables"] == []
    assert "sink" not in result["objects"]
    assert "oven" not in result["objects"]


def test_openable_requires_a_populated_handle_relationship(monkeypatch):
    """Verify that openable requires a populated handle relationship."""
    module, classes = _load_world_context(monkeypatch)
    empty_dishwasher = _make(classes["Dishwasher"], "dishwasher_empty")
    empty_dishwasher.doors = []
    empty_dishwasher.drawers = []

    working_dishwasher = _make(classes["Dishwasher"], "dishwasher_working")
    working_dishwasher.doors = [_Namespace(handle=_handle())]
    working_dishwasher.drawers = []

    result = _classify(module, [empty_dishwasher, working_dishwasher])

    assert result["containers"] == ["dishwasher_empty", "dishwasher_working"]
    assert result["openables"] == ["dishwasher_working"]


def test_openable_requires_an_articulated_handle(monkeypatch):
    """Verify that openable requires an articulated handle."""
    module, classes = _load_world_context(monkeypatch)
    door = _make(classes["Door"], "fixed_door")
    door.handle = _Namespace(root=_Namespace(name="fixed_handle"))

    result = _classify(module, [door])

    assert result["openables"] == []


def test_openable_skips_fixed_handles_and_uses_an_articulated_one(monkeypatch):
    """Verify that openable skips fixed handles and uses an articulated one."""
    module, classes = _load_world_context(monkeypatch)
    fridge = _make(classes["Fridge"], "fridge")
    fixed_handle = _Namespace(root=_Namespace(name="fixed_handle"))
    fridge.doors = [_Namespace(handle=fixed_handle), _Namespace(handle=_handle())]
    fridge.drawers = []

    result = _classify(module, [fridge])

    assert result["openables"] == ["fridge"]


def test_openable_skips_a_zero_range_connection(monkeypatch):
    """Verify that openable skips a zero range connection."""
    module, classes = _load_world_context(monkeypatch)
    door = _make(classes["Door"], "locked_door")
    door.handle = _handle_with_position_limits(0.0, 0.0)

    result = _classify(module, [door])

    assert result["openables"] == []


def test_openable_accepts_a_positive_range_connection(monkeypatch):
    """Verify that openable accepts a positive range connection."""
    module, classes = _load_world_context(monkeypatch)
    door = _make(classes["Door"], "moving_door")
    door.handle = _handle_with_position_limits(0.0, 1.0)

    result = _classify(module, [door])

    assert result["openables"] == ["moving_door"]


def test_capabilities_are_computed_for_all_annotations_on_a_root(monkeypatch):
    """Verify that capabilities are computed for all annotations on a root."""
    module, classes = _load_world_context(monkeypatch)
    root = _Namespace(name="shared_fixture")
    furniture = _make(classes["Furniture"], "unused", root=root)
    surface = _make(classes["ShelfLayer"], "unused", surface=object(), root=root)

    result = _classify(module, [furniture, surface])

    assert result["furniture"] == ["shared_fixture"]
    assert result["surfaces"] == ["shared_fixture"]
    assert result["objects"] == []
    assert result["types"]["shared_fixture"] == "ShelfLayer"


def test_appliance_child_door_is_hidden(monkeypatch):
    """Verify that appliance child door is hidden."""
    module, classes = _load_world_context(monkeypatch)
    fridge = _make(classes["Fridge"], "fridge_main")
    door_root = _Namespace(
        name="fridge_door",
        parent_kinematic_structure_entity=fridge.root,
    )
    door = _make(classes["Door"], "unused", root=door_root)
    door.handle = _handle()
    fridge.doors = [door]
    fridge.drawers = []

    result = _classify(module, [fridge, door])

    assert result["openables"] == ["fridge_main"]
    assert "fridge_door" not in result["types"]


def test_planner_ids_shorten_unique_names_and_preserve_ambiguous_names(monkeypatch):
    """Verify that planner ids shorten unique names and preserve ambiguous names."""
    module, classes = _load_world_context(monkeypatch)
    first = _make(classes["Milk"], _PrefixedName("milk", "island"))
    second = _make(classes["Milk"], _PrefixedName("milk", "table"))
    surface = _make(
        classes["ShelfLayer"],
        _PrefixedName("counter", "kitchen"),
        surface=object(),
    )
    monkeypatch.setattr(
        module,
        "is_supported_by",
        lambda obj, candidate: obj is first.root and candidate is surface.root,
    )

    result = _classify(module, [first, second, surface])

    assert result["objects"] == ["island/milk", "table/milk"]
    assert result["surfaces"] == ["counter"]
    assert result["object_locations"] == {"island/milk": ["counter"]}
    assert result["types"]["island/milk"] == "Milk"
    assert result["types"]["counter"] == "ShelfLayer"


def test_object_locations_prefer_semdt_storage_membership(monkeypatch):
    """Verify that object locations prefer semdt storage membership."""
    module, classes = _load_world_context(monkeypatch)
    milk = _make(classes["Milk"], "milk")
    counter = _make(classes["CounterTop"], "counter", surface=object())
    counter.objects = [milk]
    monkeypatch.setattr(module, "is_supported_by", lambda obj, place: True)

    result = _classify(module, [milk, counter])

    assert result["object_locations"] == {"milk": ["counter"]}


def test_storage_location_checks_surface_relation_for_hybrid_place(monkeypatch):
    """Verify that storage location checks surface relation for hybrid place."""
    module, classes = _load_world_context(monkeypatch)
    milk = _make(classes["Milk"], "milk")
    root = _Namespace(name="hybrid")
    container = _make(classes["Drawer"], "unused", root=root)
    surface = _make(classes["ShelfLayer"], "unused", surface=object(), root=root)
    container.objects = [milk]
    surface.objects = [milk]
    monkeypatch.setattr(
        module,
        "is_supported_by",
        lambda obj, place: obj is milk.root and place is root,
    )

    place_index = module.PlaceIndex(
        [("hybrid", root)],
        [("hybrid", root)],
        {root: [container, surface]},
    )

    assert place_index.locations_of(milk.root) == ["hybrid"]


def test_object_locations_ignore_stale_storage_membership(monkeypatch):
    """Verify that object locations ignore stale storage membership."""
    module, classes = _load_world_context(monkeypatch)
    milk = _make(classes["Milk"], "milk")
    old_counter = _make(classes["CounterTop"], "old_counter", surface=object())
    new_counter = _make(classes["CounterTop"], "new_counter", surface=object())
    old_counter.objects = [milk]
    monkeypatch.setattr(
        module,
        "is_supported_by",
        lambda obj, place: place is new_counter.root,
    )

    result = _classify(module, [milk, old_counter, new_counter])

    assert result["object_locations"] == {"milk": ["new_counter"]}


def test_object_locations_prefer_specific_child_surface(monkeypatch):
    """Verify that object locations prefer specific child surface."""
    module, classes = _load_world_context(monkeypatch)
    milk = _make(classes["Milk"], "milk")
    area_root = _Namespace(name="sink_area")
    surface_root = _Namespace(
        name="sink_area_surface",
        parent_kinematic_structure_entity=area_root,
    )
    area = _make(classes["CounterTop"], "unused", surface=object(), root=area_root)
    surface = _make(
        classes["ShelfLayer"],
        "unused",
        surface=object(),
        root=surface_root,
    )
    monkeypatch.setattr(
        module,
        "is_supported_by",
        lambda obj, place: obj is milk.root and place in {area_root, surface_root},
    )

    result = _classify(module, [milk, area, surface])

    assert result["object_locations"] == {"milk": ["sink_area_surface"]}


def test_supporting_surface_of_prefers_specific_child_surface(monkeypatch):
    """Verify grounding and context use the same specific surface choice."""
    module, classes = _load_world_context(monkeypatch)
    milk = _make(classes["Milk"], "milk")
    area_root = _Namespace(name="sink_area")
    surface_root = _Namespace(
        name="sink_area_surface",
        parent_kinematic_structure_entity=area_root,
    )
    area = _make(classes["CounterTop"], "unused", surface=object(), root=area_root)
    surface = _make(
        classes["ShelfLayer"],
        "unused",
        surface=object(),
        root=surface_root,
    )
    monkeypatch.setattr(
        module,
        "is_supported_by",
        lambda obj, place: obj is milk.root and place in {area_root, surface_root},
    )
    world = _Namespace(get_semantic_annotations_by_type=lambda _type: [area, surface])

    result = module.supporting_surface_of(world, milk.root)

    assert result is surface


def test_object_locations_use_nearest_container_ancestor(monkeypatch):
    """Verify that object locations use nearest container ancestor."""
    module, classes = _load_world_context(monkeypatch)
    drawer = _make(classes["Drawer"], "drawer")
    milk_root = _Namespace(
        name="milk",
        parent_kinematic_structure_entity=drawer.root,
    )
    milk = _make(classes["Milk"], "unused", root=milk_root)

    result = _classify(module, [milk, drawer])

    assert result["object_locations"] == {"milk": ["drawer"]}


def test_structurally_attached_oversized_object_is_at_container(monkeypatch):
    """Verify that structurally attached oversized object is at container."""
    module, _classes = _load_world_context(monkeypatch)
    drawer = _Namespace(name="drawer")
    large_box = _Namespace(
        name="large_box",
        parent_kinematic_structure_entity=drawer,
    )

    assert module.is_inside_or_attached(large_box, drawer)
    assert module.is_at_location(large_box, drawer)


class _FakeBody:
    """Named body that hashes by identity, like semDT world entities."""

    def __init__(self, name):
        self.name = name


def test_world_context_prefers_most_contained_container(monkeypatch):
    """When several containers hold the object, the tightest fit wins."""
    world_context, _ = _load_world_context(monkeypatch)
    obj = _FakeBody("milk")
    fridge = _FakeBody("fridge")
    drawer = _FakeBody("drawer")
    ratios = {"fridge": 0.92, "drawer": 0.98}

    class InsideOf:
        def __init__(self, body, other):
            self.other = other

        def compute_containment_ratio(self):
            """Return the fake containment ratio."""
            return ratios[self.other.name]

    monkeypatch.setattr(world_context, "InsideOf", InsideOf)
    place_index = world_context.PlaceIndex(
        [("fridge", fridge), ("drawer", drawer)], [], {}
    )

    assert place_index.locations_of(obj) == ["drawer"]


def test_world_context_ignores_containers_below_threshold(monkeypatch):
    """A weak containment ratio does not count as holding the object."""
    world_context, _ = _load_world_context(monkeypatch)
    obj = _FakeBody("milk")
    box = _FakeBody("box")

    class InsideOf:
        def __init__(self, body, other):
            self.other = other

        def compute_containment_ratio(self):
            """Return the fake containment ratio."""
            return 0.5

    monkeypatch.setattr(world_context, "InsideOf", InsideOf)
    monkeypatch.setattr(world_context, "is_supported_by", lambda o, s: False)
    place_index = world_context.PlaceIndex([("box", box)], [], {})

    assert place_index.locations_of(obj) == []


def test_world_context_uses_supported_by_for_surface(monkeypatch):
    """With no container, the supporting surface comes from is_supported_by."""
    world_context, _ = _load_world_context(monkeypatch)
    obj = _FakeBody("milk")
    table = _FakeBody("table")

    monkeypatch.setattr(
        world_context, "is_supported_by", lambda o, s: s.name == "table"
    )
    place_index = world_context.PlaceIndex([], [("table", table)], {})

    assert place_index.locations_of(obj) == ["table"]
