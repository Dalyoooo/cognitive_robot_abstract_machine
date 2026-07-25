from dataclasses import dataclass
from xml.etree.ElementTree import Element

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CheezeIt,
    CounterTop,
    Drawer,
    Fork,
    Fridge,
    GelatinBox,
    Kettle,
    Kitchen,
    Knife,
    MustardBottle,
    Oven,
    Pringles,
    SaltContainer,
    Sink,
    SoapBottle,
    Spoon,
    Table,
    TomatoSoup,
    TrashCan,
    TunaCan,
    WineBottle,
)

from thesis_demo.world.environment import (
    ContainedPlacement,
    EnvironmentSpec,
    FurnitureAnnotation,
    RESOURCES,
    RoomSpec,
    SurfacePlacement,
)


@dataclass(eq=False)
class KitchenIsland(CounterTop):
    pass


@dataclass(eq=False)
class SinkCounter(CounterTop):
    pass


def correct_oven_drawer_limit(urdf_root: Element):
    left_drawer_joint = urdf_root.find(
        "./joint[@name='oven_area_area_left_drawer_main_joint']"
    )
    if left_drawer_joint is None:
        return
    for duplicate_limit in left_drawer_joint.findall("limit")[1:]:
        left_drawer_joint.remove(duplicate_limit)


KITCHEN = EnvironmentSpec(
    urdf=f"{RESOURCES}/worlds/kitchen-small.urdf",
    robot_start=(0.3, 0.8, 0.0),
    rooms=(RoomSpec(Kitchen, "kitchen", (-1.0, 0.56), (5.6, 4.9)),),
    furniture=(
        FurnitureAnnotation(KitchenIsland, "kitchen_island_surface"),
        FurnitureAnnotation(SinkCounter, "sink_area_surface"),
        FurnitureAnnotation(Table, "table_area"),
        FurnitureAnnotation(Sink, "sink_area_sink"),
        FurnitureAnnotation(Oven, "oven_area_oven_main"),
        FurnitureAnnotation(Fridge, "iai_fridge_main"),
        FurnitureAnnotation(TrashCan, "sink_area_trash_drawer_main"),
    ),
    surface_objects=(
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
            "table_area",
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
    ),
    contained_objects=(
        ContainedPlacement(
            TunaCan,
            "tunacan",
            "sink_area_left_upper_drawer_main",
            Drawer,
            (0.0, 0.0, 0.0),
            scale=(0.06, 0.06, 0.035),
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
    ),
    urdf_customizer=correct_oven_drawer_limit,
)
