import sys
from pathlib import Path

import pytest
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.semantic_annotations.mixins import (
    HasStorageSpace,
    HasSupportingSurface,
)


_DEMOS_DIR = Path(__file__).resolve().parents[1] / "pycram" / "demos"
sys.path.insert(0, str(_DEMOS_DIR))
try:
    import thesis_demo.nlp_demo as nlp_demo
finally:
    sys.path.remove(str(_DEMOS_DIR))


_EXPECTED_SURFACES = {
    "apartment": (
        ("island_countertop", "CounterTop"),
        ("countertop", "CounterTop"),
        ("table_area_main", "Table"),
        ("coffee_table", "CoffeeTable"),
        ("bedside_table", "SideTable"),
    ),
    "kitchen": (
        ("kitchen_island", "CounterTop"),
        ("sink_area", "CounterTop"),
        ("fridge_area", "CounterTop"),
        ("oven_area_area", "CounterTop"),
        ("kitchen_island_surface", "CounterTop"),
        ("sink_area_surface", "CounterTop"),
        ("table_area", "Table"),
    ),
}

_EXPECTED_FIXTURES = {
    "apartment": (
        ("sink", "Sink"),
        ("oven", "Oven"),
        ("cabinet7", "Dishwasher"),
    ),
    "kitchen": (
        ("sink_area_sink", "Sink"),
        ("oven_area_oven_main", "Oven"),
        ("iai_fridge_main", "Fridge"),
    ),
}

_EXPECTED_ROOMS = {
    "apartment": (("kitchen", "Kitchen"), ("living_room", "LivingRoom")),
    "kitchen": (("kitchen", "Kitchen"),),
}

# Object IDs, semantic types, and source relations are part of the Binder data
# contract. Placement offsets intentionally are not fixed here, so they can be
# improved without changing the held-out evaluation inventory.
_EXPECTED_OBJECTS = {
    "apartment": (
        ("supported_by", "bowl.stl", "Bowl", "island_countertop"),
        (
            "supported_by",
            "breakfast_cereal.stl",
            "Cereal",
            "island_countertop",
        ),
        ("supported_by", "milk.stl", "Milk", "island_countertop"),
        ("supported_by", "Static_CokeBottle.stl", "Bottle", "countertop"),
        ("supported_by", "jeroen_cup.stl", "Mug", "table_area_main"),
        ("supported_by", "plate", "Plate", "table_area_main"),
        ("supported_by", "plate_counter", "Plate", "countertop"),
        ("supported_by", "apple", "Apple", "table_area_main"),
        ("supported_by", "apple_island", "Apple", "island_countertop"),
        ("contained_in", "spoon.stl", "Spoon", "cabinet10_drawer_top"),
        ("contained_in", "fork", "Fork", "cabinet10_drawer_top"),
        ("contained_in", "knife", "Knife", "cabinet10_drawer_top"),
        ("contained_in", "mug_sink", "Mug", "sink"),
    ),
    "kitchen": (
        ("supported_by", "wine_bottle", "WineBottle", "table_area"),
        ("supported_by", "soap_bottle", "SoapBottle", "sink_area_surface"),
        ("supported_by", "kettle", "Kettle", "table_area"),
        (
            "supported_by",
            "cheezeit",
            "CheezeIt",
            "kitchen_island_surface",
        ),
        (
            "contained_in",
            "tunacan",
            "TunaCan",
            "sink_area_left_upper_drawer_main",
        ),
        (
            "contained_in",
            "saltcontainer",
            "SaltContainer",
            "sink_area_left_middle_drawer_main",
        ),
        ("contained_in", "mustard_bottle", "MustardBottle", "iai_fridge_main"),
        ("contained_in", "pringles", "Pringles", "iai_fridge_main"),
        ("contained_in", "gelatinbox", "GelatinBox", "iai_fridge_main"),
        ("contained_in", "tomatosoup", "TomatoSoup", "iai_fridge_main"),
    ),
}


def _configured_surfaces(spec):
    return tuple(
        (surface.name, surface.annotation_type.__name__) for surface in spec.surfaces
    )


def _configured_fixtures(spec):
    return tuple(
        (fixture.name, fixture.annotation_type.__name__) for fixture in spec.fixtures
    )


def _configured_objects(spec):
    return tuple(
        (
            "supported_by",
            placement.name,
            placement.annotation_type.__name__,
            placement.surface,
        )
        for placement in spec.surface_objects
    ) + tuple(
        (
            "contained_in",
            placement.name,
            placement.annotation_type.__name__,
            placement.container,
        )
        for placement in spec.contained_objects
    )


def _configured_rooms(spec):
    return tuple((room.name, room.annotation_type.__name__) for room in spec.rooms)


def _annotation_on(world, body_name, annotation_type):
    body = world.get_body_by_name(body_name)
    matches = [
        annotation
        for annotation in world.semantic_annotations
        if type(annotation) is annotation_type
        and getattr(annotation, "root", None) is body
    ]
    assert len(matches) == 1
    return matches[0]


def _surface_on(world, body_name):
    body = world.get_body_by_name(body_name)
    matches = [
        annotation
        for annotation in world.semantic_annotations
        if isinstance(annotation, HasSupportingSurface)
        and getattr(annotation, "root", None) is body
        and annotation.supporting_surface is not None
    ]
    assert len(matches) == 1
    return matches[0]


def test_environment_configuration_preserves_binder_contract():
    assert tuple(nlp_demo.ROBOTS) == ("pr2", "hsrb", "tiago")
    assert tuple(nlp_demo.ENVIRONMENTS) == ("apartment", "kitchen")

    for environment_name, spec in nlp_demo.ENVIRONMENTS.items():
        expected_urdf = {"apartment": "apartment.urdf", "kitchen": "kitchen-small.urdf"}
        assert Path(spec.urdf).name == expected_urdf[environment_name]
        assert _configured_surfaces(spec) == _EXPECTED_SURFACES[environment_name]
        assert _configured_fixtures(spec) == _EXPECTED_FIXTURES[environment_name]
        assert _configured_rooms(spec) == _EXPECTED_ROOMS[environment_name]
        assert _configured_objects(spec) == _EXPECTED_OBJECTS[environment_name]

        for placement in (*spec.surface_objects, *spec.contained_objects):
            assert (placement.mesh is None) != (placement.scale is None)


@pytest.mark.parametrize("environment_name", ("apartment", "kitchen"))
def test_build_environment_creates_valid_semdt_relations(environment_name):
    spec = nlp_demo.ENVIRONMENTS[environment_name]
    world = nlp_demo._build_environment(spec)

    assert world.validate()

    for surface_spec in spec.surfaces:
        surface = _surface_on(world, surface_spec.name)
        assert type(surface) is surface_spec.annotation_type
        assert surface.supporting_surface is not None
        assert surface.supporting_surface.combined_mesh is not None

    for fixture_spec in spec.fixtures:
        fixture = _annotation_on(world, fixture_spec.name, fixture_spec.annotation_type)
        assert type(fixture) is fixture_spec.annotation_type

    for room_spec in spec.rooms:
        rooms = [
            room
            for room in world.semantic_annotations
            if type(room) is room_spec.annotation_type
            and str(room.name) == room_spec.name
        ]
        assert len(rooms) == 1
        assert rooms[0].floor is not None

    for placement in spec.surface_objects:
        body = world.get_body_by_name(placement.name)
        surface = _surface_on(world, placement.surface)
        object_annotation = _annotation_on(
            world, placement.name, placement.annotation_type
        )

        assert type(object_annotation) is placement.annotation_type
        assert body.parent_kinematic_structure_entity is surface.root
        assert is_supported_by(body, surface.root)
        assert object_annotation in surface.objects

    for placement in spec.contained_objects:
        body = world.get_body_by_name(placement.name)
        container_body = world.get_body_by_name(placement.container)
        container = _annotation_on(world, placement.container, placement.container_type)
        object_annotation = _annotation_on(
            world, placement.name, placement.annotation_type
        )

        assert type(object_annotation) is placement.annotation_type
        assert body.parent_kinematic_structure_entity is container_body
        if isinstance(container, HasStorageSpace):
            assert object_annotation in container.objects
        else:
            assert placement.container == "sink"


def test_build_world_visualize_false_keeps_four_value_api_headless(monkeypatch):
    built_world = object()
    robot = object()
    context = object()

    def fake_build_world_model(robot_name, environment):
        assert robot_name == "hsrb"
        assert environment == "kitchen"
        return built_world, robot, context

    def fail_if_visualized(_world):
        pytest.fail("visualization must not start when visualize=False")

    monkeypatch.setattr(nlp_demo, "build_world_model", fake_build_world_model)
    monkeypatch.setattr(nlp_demo, "_start_visualization", fail_if_visualized)

    result = nlp_demo.build_world(environment="kitchen", visualize=False)

    assert result == (built_world, robot, context, None)
