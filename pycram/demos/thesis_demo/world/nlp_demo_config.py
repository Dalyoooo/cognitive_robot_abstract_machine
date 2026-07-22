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
    TrashCan,
    TunaCan,
    WineBottle,
)

RESOURCES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "resources")
OBJECTS_DIR = os.path.join(RESOURCES, "objects")


@dataclass(eq=False)
class KitchenIsland(CounterTop): ...


@dataclass(eq=False)
class SinkCounter(CounterTop): ...


@dataclass(eq=False)
class OvenCounter(CounterTop): ...


FURNITURE_ANNOTATION_TYPES = (
    CoffeeTable,
    CounterTop,
    Dishwasher,
    Fridge,
    KitchenIsland,
    OvenCounter,
    Oven,
    SideTable,
    Sink,
    SinkCounter,
    Table,
    TrashCan,
)

SURFACE_ANNOTATION_TYPES = (
    CoffeeTable,
    CounterTop,
    KitchenIsland,
    OvenCounter,
    SideTable,
    SinkCounter,
    Table,
)


@dataclass(frozen=True)
class RoomSpec:
    annotation_type: type
    name: str
    center: tuple[float, float]
    size: tuple[float, float]


@dataclass(frozen=True)
class FurnitureAnnotation:
    annotation_type: type
    body: str


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
    rooms: tuple[RoomSpec, ...]
    furniture: tuple[FurnitureAnnotation, ...]
    surface_objects: tuple[SurfacePlacement, ...]
    contained_objects: tuple[ContainedPlacement, ...]


KITCHEN_ROOMS = (RoomSpec(Kitchen, "kitchen", (-1.0, 0.56), (5.6, 4.9)),)

KITCHEN_FURNITURE = (
    FurnitureAnnotation(KitchenIsland, "kitchen_island_surface"),
    FurnitureAnnotation(SinkCounter, "sink_area_surface"),
    FurnitureAnnotation(OvenCounter, "oven_area_area"),
    FurnitureAnnotation(Table, "table_area"),
    FurnitureAnnotation(Sink, "sink_area_sink"),
    FurnitureAnnotation(Oven, "oven_area_oven_main"),
    FurnitureAnnotation(Fridge, "iai_fridge_main"),
    FurnitureAnnotation(TrashCan, "sink_area_trash_drawer_main"),
)

# Held-out evaluation inventory. Names, types, and source relations are a contract.
KITCHEN_SURFACE_OBJECTS = (
    SurfacePlacement(
        WineBottle,
        "wine_bottle",
        "table_area",
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
        "table_area",
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
    SurfacePlacement(
        GelatinBox,
        "gelatinbox_counter",
        "kitchen_island_surface",
        (-0.20, -0.10),
        scale=(0.06, 0.06, 0.08),
    ),
    SurfacePlacement(
        TunaCan,
        "tunacan_counter",
        "table_area",
        (0.0, -0.15),
        scale=(0.06, 0.06, 0.08),
    ),
    SurfacePlacement(
        Pringles,
        "pringles",
        "oven_area_area",
        (0.0, 0.0),
        scale=(0.07, 0.07, 0.20),
    ),
    SurfacePlacement(
        GelatinBox,
        "gelatinbox",
        "sink_area_surface",
        (-0.15, 0.0),
        scale=(0.06, 0.06, 0.08),
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
        TomatoSoup,
        "tomatosoup",
        "iai_fridge_main",
        Fridge,
        (0.0, 0.0, -0.10),
        scale=(0.06, 0.06, 0.10),
    ),
    ContainedPlacement(
        Spoon,
        "spoon",
        "kitchen_island_left_upper_drawer_main",
        Drawer,
        (-0.05, -0.08, 0.0),
        scale=(0.18, 0.03, 0.02),
    ),
    ContainedPlacement(
        Fork,
        "fork",
        "kitchen_island_left_upper_drawer_main",
        Drawer,
        (-0.05, 0.0, 0.0),
        scale=(0.18, 0.02, 0.02),
    ),
    ContainedPlacement(
        Knife,
        "knife",
        "kitchen_island_left_upper_drawer_main",
        Drawer,
        (-0.05, 0.08, 0.0),
        scale=(0.18, 0.015, 0.02),
    ),
)


APARTMENT_ROOMS = (
    RoomSpec(Kitchen, "kitchen", (3.0, 2.5), (6.0, 4.5)),
    RoomSpec(LivingRoom, "living_room", (17.0, 2.5), (3.5, 4.5)),
)

APARTMENT_FURNITURE = (
    FurnitureAnnotation(CounterTop, "island_countertop"),
    FurnitureAnnotation(CounterTop, "countertop"),
    FurnitureAnnotation(Table, "table_area_main"),
    FurnitureAnnotation(CoffeeTable, "coffee_table"),
    FurnitureAnnotation(SideTable, "bedside_table"),
    FurnitureAnnotation(Sink, "sink"),
    FurnitureAnnotation(Oven, "oven"),
    FurnitureAnnotation(Dishwasher, "cabinet7"),
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
        "mug_sink",
        "countertop",
        (-0.15, 0.10),
        scale=(0.08, 0.08, 0.10),
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
)


ENVIRONMENTS = {
    "apartment": EnvironmentSpec(
        urdf=os.path.join(RESOURCES, "worlds", "apartment.urdf"),
        robot_start=(1.5, 2.5, 0.0),
        rooms=APARTMENT_ROOMS,
        furniture=APARTMENT_FURNITURE,
        surface_objects=APARTMENT_SURFACE_OBJECTS,
        contained_objects=APARTMENT_CONTAINED_OBJECTS,
    ),
    "kitchen": EnvironmentSpec(
        urdf=os.path.join(RESOURCES, "worlds", "kitchen-small.urdf"),
        robot_start=(0.3, 0.8, 0.0),
        rooms=KITCHEN_ROOMS,
        furniture=KITCHEN_FURNITURE,
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
