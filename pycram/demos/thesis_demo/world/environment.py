import os
from collections.abc import Callable
from dataclasses import dataclass
from xml.etree.ElementTree import Element

RESOURCES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "resources")
OBJECTS_DIR = os.path.join(RESOURCES, "objects")

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
    urdf_customizer: Callable[[Element], None] | None = None
