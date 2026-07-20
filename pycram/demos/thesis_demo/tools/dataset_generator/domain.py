from dataclasses import dataclass, field, replace
from typing import Any, Literal

CONTEXT_KEYS = (
    "objects",
    "object_locations",
    "surfaces",
    "containers",
    "openables",
    "furniture",
    "rooms",
    "types",
)

Role = Literal["object", "surface", "container", "furniture", "room", "place"]
Slot = Literal["action", "object", "source", "destination"]
DIRECTIONAL_RELATIONS = ("left_of", "right_of", "in_front_of", "behind")


def unique(values):
    """
    Remove duplicate values while preserving their input order.

    Args:
        values: Iterable of hashable values.

    Returns:
        A tuple containing each value once.
    """
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class WorldContext:
    """The compact world representation shown to the model."""

    objects: tuple[str, ...]
    object_locations: dict[str, tuple[str, ...]]
    surfaces: tuple[str, ...]
    containers: tuple[str, ...]
    openables: tuple[str, ...]
    furniture: tuple[str, ...]
    rooms: tuple[str, ...]
    types: dict[str, str]

    def to_dict(self):
        """
        Convert the context to the runtime dictionary format.

        Returns:
            JSON-compatible context data in runtime key order.
        """
        return {
            "objects": list(self.objects),
            "object_locations": {
                name: list(locations)
                for name, locations in self.object_locations.items()
            },
            "surfaces": list(self.surfaces),
            "containers": list(self.containers),
            "openables": list(self.openables),
            "furniture": list(self.furniture),
            "rooms": list(self.rooms),
            "types": dict(self.types),
        }

    @property
    def places(self):
        """
        Return all place-like context entities without duplicates.

        Returns:
            An ordered tuple of surfaces, containers, furniture, and rooms.
        """
        return unique(self.surfaces + self.containers + self.furniture + self.rooms)

    def role_of(self, name):
        """
        Return the planner role of a context entity.

        Args:
            name: Canonical context entity name.

        Returns:
            The entity role, or None if the name is unknown.
        """
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
        """
        Return context entity names for a planner role.

        Args:
            role: Planner role to select.

        Returns:
            Entity names for the role, or all places for a general place role.
        """
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
        """
        Validate that the context can ground generated plans.

        Raises:
            ValueError: If categories overlap or references use unknown names.
        """
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
    """A canonical task before it is rendered as language."""

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
        """
        Return an intent with one grounding slot replaced.

        Args:
            slot: Intent slot to replace.
            value: Canonical value assigned to the slot.

        Returns:
            A new intent containing the updated slot value.
        """
        if slot == "object":
            return replace(self, object=value)
        if slot == "source":
            return replace(self, source=value, source_explicit=True)
        return replace(self, destination=value)


@dataclass(frozen=True)
class Mention:
    """One entity reference used in a rendered instruction."""

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
    """Result of grounding generated mentions in a world context."""

    status: Literal["resolved", "ambiguous", "missing"]
    intent: Intent
    slot: Slot | None = None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanStep:
    """One exact five-field planner action."""

    action: str
    object: str | None = None
    location: str | None = None
    relation: str | None = None
    source: str | None = None

    def to_dict(self):
        """
        Convert the plan step to the exact runtime schema.

        Returns:
            A dictionary containing all five planner action fields.
        """
        return {
            "action": self.action,
            "object": self.object,
            "location": self.location,
            "relation": self.relation,
            "source": self.source,
        }


@dataclass(frozen=True)
class Scenario:
    """A sampled task, its context, and its chosen referring style."""

    id: str
    family: str
    world_id: str
    context: WorldContext
    intent: Intent
    references: dict[Slot, str | None]
    extra_intents: tuple[Intent, ...] = ()
    extra_references: tuple[dict[Slot, str | None], ...] = ()

    def intents(self):
        """
        Return every intent in conversation order.

        Returns:
            The primary intent followed by any additional intents.
        """
        return (self.intent, *self.extra_intents)

    def reference_sets(self):
        """
        Return every reference mapping in conversation order.

        Returns:
            The primary references followed by additional reference mappings.
        """
        return (self.references, *self.extra_references)


@dataclass
class RawExample:
    """A validated example before conversion to fine-tuning JSONL."""

    scenario_id: str
    family: str
    world_id: str
    context: WorldContext
    messages: list[dict[str, str | None]]
    plan: tuple[PlanStep, ...] = ()
    clarification: str | None = None
    template_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
