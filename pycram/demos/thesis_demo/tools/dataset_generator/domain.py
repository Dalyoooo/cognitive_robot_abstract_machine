import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal


Role = Literal["object", "surface", "container", "furniture", "room", "place"]
Slot = Literal["action", "object", "source", "destination"]
AREAS = (
    "kitchen",
    "dining",
    "hallway",
    "living_room",
    "pantry",
    "storage",
    "utility",
    "work_area",
)

POSITIONS = ("left", "right", "upper", "lower", "middle", "front", "back")

RECOGNIZED_POSITIONS = frozenset(POSITIONS) | {"top", "bottom", "center"}

IGNORED_REFERENCE_WORDS = frozenset({"main"})


def unique(values):
    return tuple(dict.fromkeys(values))


def normalise_words(value):
    return re.findall(r"[a-z0-9]+", value.casefold())


@dataclass(frozen=True)
class WorldContext:

    objects: tuple[str, ...]
    object_locations: dict[str, tuple[str, ...]]
    surfaces: tuple[str, ...]
    containers: tuple[str, ...]
    openables: tuple[str, ...]
    furniture: tuple[str, ...]
    rooms: tuple[str, ...]
    types: dict[str, str]

    def is_openable(self, name):
        return name in self.openables

    def to_dict(self):
        return asdict(self)

    @property
    def places(self):
        return unique(self.surfaces + self.containers + self.furniture + self.rooms)

    def role_of(self, name):
        for role, names in (
            ("object", self.objects),
            ("surface", self.surfaces),
            ("container", self.containers),
            ("furniture", self.furniture),
            ("room", self.rooms),
        ):
            if name in names:
                return role
        return None

    def names_for_role(self, role):
        if role == "object":
            return self.objects
        if role == "surface":
            return self.surfaces
        if role == "container":
            return self.containers
        if role == "furniture":
            return self.furniture
        if role == "room":
            return self.rooms
        return self.places

    def validate(self):
        category_sets = [
            set(self.surfaces),
            set(self.containers),
            set(self.furniture),
            set(self.rooms),
        ]
        for index, left in enumerate(category_sets):
            for right in category_sets[index + 1 :]:
                overlap = left & right
                if overlap:
                    raise ValueError(f"place categories overlap: {sorted(overlap)}")

        all_names = set(self.objects) | set(self.places)
        missing_types = all_names - set(self.types)
        if missing_types:
            raise ValueError(f"context names without types: {sorted(missing_types)}")

        unknown_objects = set(self.object_locations) - set(self.objects)
        if unknown_objects:
            raise ValueError(
                f"object_locations contains unknown objects: {sorted(unknown_objects)}"
            )

        unknown_openables = set(self.openables) - set(self.containers)
        if unknown_openables:
            raise ValueError(
                f"openables contains unknown containers: {sorted(unknown_openables)}"
            )
        for object_name, locations in self.object_locations.items():
            unknown_places = set(locations) - set(self.places)
            if unknown_places:
                raise ValueError(
                    f"{object_name!r} has unknown locations: {sorted(unknown_places)}"
                )


@dataclass(frozen=True)
class Intent:

    action: Literal[
        "transport",
        "pickup",
        "pickup_place",
        "navigate",
        "open",
        "close",
    ]
    object: str | None = None
    source: str | None = None
    destination: str | None = None
    relation: str | None = None
    source_explicit: bool = False

    def with_slot(self, slot, value):
        if slot == "object":
            return replace(self, object=value)
        if slot == "source":
            return replace(self, source=value, source_explicit=True)
        return replace(self, destination=value)


@dataclass(frozen=True)
class Mention:

    slot: Slot
    text: str | None
    role: Role


@dataclass(frozen=True)
class RenderedInstruction:
    text: str
    mentions: tuple[Mention, ...]
    template_id: str
    clauses: tuple["RenderedInstruction", ...] = ()


@dataclass(frozen=True)
class Resolution:

    status: Literal["resolved", "ambiguous", "missing"]
    intent: Intent
    slot: Slot | None = None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanStep:

    action: str
    object: str | None = None
    location: str | None = None
    relation: str | None = None
    source: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Scenario:

    id: str
    family: str
    world_id: str
    context: WorldContext
    intent: Intent
    references: dict[Slot, str | None]
    extra_intents: tuple[Intent, ...] = ()
    extra_references: tuple[dict[Slot, str | None], ...] = ()

    def intents(self):
        return (self.intent, *self.extra_intents)

    def reference_sets(self):
        return (self.references, *self.extra_references)


@dataclass
class RawExample:

    scenario_id: str
    family: str
    world_id: str
    context: WorldContext
    messages: list[dict[str, str | None]]
    plan: tuple[PlanStep, ...] = ()
    clarification: str | None = None
    template_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
