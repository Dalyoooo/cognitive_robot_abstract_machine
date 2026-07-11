"""Declarative world and placement data for the NLP Binder demo."""

import os
from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Bottle,
    Bowl,
    Cereal,
    CheezeIt,
    CoffeeTable,
    CounterTop,
    Dishwasher,
    Drawer,
    Fork,
    Fridge,
    GelatinBox,
    Kettle,
    Kitchen,
    Knife,
    LivingRoom,
    Milk,
    Mug,
    MustardBottle,
    Oven,
    Plate,
    Pringles,
    SaltContainer,
    SideTable,
    Sink,
    SoapBottle,
    Spoon,
    Table,
    TomatoSoup,
    TunaCan,
    WineBottle,
)


RESOURCES = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
OBJECTS_DIR = os.path.join(RESOURCES, "objects")


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    annotation_type: type


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    annotation_type: type


@dataclass(frozen=True)
class RoomSpec:
    annotation_type: type
    name: str
    center: tuple[float, float]
    size: tuple[float, float]


@dataclass(frozen=True)
class SurfacePlacement:
    annotation_type: type
    name: str
    surface: str
    offset: tuple[float, float]
    mesh: str | None = None
    scale: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ContainedPlacement:
    annotation_type: type
    name: str
    container: str
    container_type: type
    offset: tuple[float, float, float]
    mesh: str | None = None
    scale: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class EnvironmentSpec:
    urdf: str
    robot_start: tuple[float, float, float]
    surfaces: tuple[SurfaceSpec, ...]
    fixtures: tuple[FixtureSpec, ...]
    rooms: tuple[RoomSpec, ...]
    surface_objects: tuple[SurfacePlacement, ...]
    contained_objects: tuple[ContainedPlacement, ...]


KITCHEN_SURFACES = (
    SurfaceSpec("kitchen_island", CounterTop),
    SurfaceSpec("sink_area", CounterTop),
    SurfaceSpec("fridge_area", CounterTop),
    SurfaceSpec("oven_area_area", CounterTop),
    SurfaceSpec("kitchen_island_surface", CounterTop),
    SurfaceSpec("sink_area_surface", CounterTop),
    SurfaceSpec("table_area_main", Table),
)

KITCHEN_FIXTURES = (
    FixtureSpec("sink_area_sink", Sink),
    FixtureSpec("oven_area_oven_main", Oven),
    FixtureSpec("iai_fridge_main", Fridge),
    FixtureSpec("sink_area_dish_washer_main", Dishwasher),
)

KITCHEN_ROOMS = (RoomSpec(Kitchen, "kitchen", (-1.0, 0.56), (5.6, 4.9)),)

# Held-out evaluation inventory. IDs, types, and source relations are a contract.
KITCHEN_SURFACE_OBJECTS = (
    SurfacePlacement(
        WineBottle,
        "wine_bottle",
        "table_area_main",
        (-0.15, 0.15),
        scale=(0.07, 0.07, 0.25),
    ),
    SurfacePlacement(
        SoapBottle,
        "soap_bottle",
        "sink_area_surface",
        (0.15, 0.0),
        scale=(0.06, 0.08, 0.15),
    ),
    SurfacePlacement(
        Kettle,
        "kettle",
        "table_area_main",
        (0.20, 0.15),
        scale=(0.12, 0.12, 0.18),
    ),
    SurfacePlacement(
        CheezeIt,
        "cheezeit",
        "kitchen_island_surface",
        (0.20, 0.10),
        scale=(0.06, 0.06, 0.12),
    ),
)

KITCHEN_CONTAINED_OBJECTS = (
    ContainedPlacement(
        TunaCan,
        "tunacan",
        "sink_area_left_upper_drawer_main",
        Drawer,
        (0.0, 0.0, 0.0),
        scale=(0.06, 0.06, 0.08),
    ),
    ContainedPlacement(
        SaltContainer,
        "saltcontainer",
        "sink_area_left_middle_drawer_main",
        Drawer,
        (0.0, 0.0, 0.0),
        scale=(0.05, 0.05, 0.12),
    ),
    ContainedPlacement(
        MustardBottle,
        "mustard_bottle",
        "iai_fridge_main",
        Fridge,
        (0.10, -0.05, 0.15),
        scale=(0.06, 0.06, 0.18),
    ),
    ContainedPlacement(
        Pringles,
        "pringles",
        "iai_fridge_main",
        Fridge,
        (-0.10, 0.05, 0.0),
        scale=(0.07, 0.07, 0.20),
    ),
    ContainedPlacement(
        GelatinBox,
        "gelatinbox",
        "iai_fridge_main",
        Fridge,
        (0.10, 0.05, 0.0),
        scale=(0.06, 0.06, 0.08),
    ),
    ContainedPlacement(
        TomatoSoup,
        "tomatosoup",
        "iai_fridge_main",
        Fridge,
        (0.0, 0.0, -0.10),
        scale=(0.06, 0.06, 0.10),
    ),
)


APARTMENT_SURFACES = (
    SurfaceSpec("island_countertop", CounterTop),
    SurfaceSpec("countertop", CounterTop),
    SurfaceSpec("table_area_main", Table),
    SurfaceSpec("coffee_table", CoffeeTable),
    SurfaceSpec("bedside_table", SideTable),
)

APARTMENT_FIXTURES = (
    FixtureSpec("sink", Sink),
    FixtureSpec("oven", Oven),
    FixtureSpec("cabinet7", Dishwasher),
)

APARTMENT_ROOMS = (
    RoomSpec(Kitchen, "kitchen", (3.0, 2.5), (6.0, 4.5)),
    RoomSpec(LivingRoom, "living_room", (17.0, 2.5), (3.5, 4.5)),
)

APARTMENT_SURFACE_OBJECTS = (
    SurfacePlacement(
        Bowl, "bowl.stl", "island_countertop", (-0.20, -0.10), mesh="bowl.stl"
    ),
    SurfacePlacement(
        Cereal,
        "breakfast_cereal.stl",
        "island_countertop",
        (0.20, -0.10),
        mesh="breakfast_cereal.stl",
    ),
    SurfacePlacement(
        Milk, "milk.stl", "island_countertop", (-0.20, 0.10), mesh="milk.stl"
    ),
    SurfacePlacement(
        Bottle,
        "Static_CokeBottle.stl",
        "countertop",
        (0.0, 0.0),
        mesh="Static_CokeBottle.stl",
    ),
    SurfacePlacement(
        Mug,
        "jeroen_cup.stl",
        "table_area_main",
        (0.0, 0.0),
        mesh="jeroen_cup.stl",
    ),
    SurfacePlacement(
        Plate,
        "plate",
        "table_area_main",
        (-0.15, -0.10),
        scale=(0.18, 0.18, 0.02),
    ),
    SurfacePlacement(
        Plate,
        "plate_counter",
        "countertop",
        (0.15, -0.10),
        scale=(0.18, 0.18, 0.02),
    ),
    SurfacePlacement(
        Apple,
        "apple",
        "table_area_main",
        (-0.15, 0.10),
        scale=(0.08, 0.08, 0.08),
    ),
    SurfacePlacement(
        Apple,
        "apple_island",
        "island_countertop",
        (0.15, 0.10),
        scale=(0.08, 0.08, 0.08),
    ),
)

APARTMENT_CONTAINED_OBJECTS = (
    ContainedPlacement(
        Spoon,
        "spoon.stl",
        "cabinet10_drawer_top",
        Drawer,
        (-0.05, -0.10, 0.0),
        mesh="spoon.stl",
    ),
    ContainedPlacement(
        Fork,
        "fork",
        "cabinet10_drawer_top",
        Drawer,
        (-0.05, 0.0, 0.0),
        scale=(0.18, 0.02, 0.02),
    ),
    ContainedPlacement(
        Knife,
        "knife",
        "cabinet10_drawer_top",
        Drawer,
        (-0.05, 0.10, 0.0),
        scale=(0.18, 0.015, 0.02),
    ),
    # Sink has no HasStorageSpace mixin; the builder keeps a structural attachment.
    ContainedPlacement(
        Mug,
        "mug_sink",
        "sink",
        Sink,
        (0.0, 0.0, 0.0),
        scale=(0.08, 0.08, 0.10),
    ),
)


ENVIRONMENTS = {
    "apartment": EnvironmentSpec(
        urdf=os.path.join(RESOURCES, "worlds", "apartment.urdf"),
        robot_start=(1.5, 2.5, 0.0),
        surfaces=APARTMENT_SURFACES,
        fixtures=APARTMENT_FIXTURES,
        rooms=APARTMENT_ROOMS,
        surface_objects=APARTMENT_SURFACE_OBJECTS,
        contained_objects=APARTMENT_CONTAINED_OBJECTS,
    ),
    "kitchen": EnvironmentSpec(
        urdf=os.path.join(RESOURCES, "worlds", "kitchen.urdf"),
        robot_start=(0.3, 0.8, 0.0),
        surfaces=KITCHEN_SURFACES,
        fixtures=KITCHEN_FIXTURES,
        rooms=KITCHEN_ROOMS,
        surface_objects=KITCHEN_SURFACE_OBJECTS,
        contained_objects=KITCHEN_CONTAINED_OBJECTS,
    ),
}


OBJECT_COLORS = {
    "Bowl": (0.20, 0.40, 0.80),
    "Mug": (0.80, 0.20, 0.20),
    "Cereal": (0.80, 0.70, 0.20),
    "Milk": (0.92, 0.92, 0.92),
    "Spoon": (0.75, 0.75, 0.78),
    "Bottle": (0.70, 0.10, 0.10),
    "Plate": (0.92, 0.92, 0.92),
    "Apple": (0.80, 0.15, 0.15),
    "Fork": (0.75, 0.75, 0.78),
    "Knife": (0.75, 0.75, 0.78),
    "MustardBottle": (0.85, 0.72, 0.10),
    "WineBottle": (0.45, 0.10, 0.12),
    "SoapBottle": (0.20, 0.70, 0.30),
    "Kettle": (0.20, 0.20, 0.22),
    "TunaCan": (0.70, 0.50, 0.30),
    "CheezeIt": (0.90, 0.60, 0.05),
    "Pringles": (0.90, 0.10, 0.10),
    "GelatinBox": (0.60, 0.30, 0.70),
    "TomatoSoup": (0.80, 0.10, 0.10),
    "SaltContainer": (0.85, 0.85, 0.85),
}
