"""The plan vocabulary: step schema, action names, relations, context keys.

This module is the single owner of the plan contract (I2) and carries no
imports beyond the standard library, so the bridge, the tooling, and the
executor can all share it without dragging in semDT or pycram (I1).
"""

from dataclasses import dataclass

# The context contract: produced by world_context, consumed by prompt.
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

VALID_PYCRAM_ACTIONS = {
    "NavigateAction",
    "PickUpAction",
    "PlaceAction",
    "TransportAction",
    "OpenAction",
    "CloseAction",
    "ParkArmsAction",
}

DIRECTIONAL_RELATIONS = ("left_of", "right_of", "in_front_of", "behind")
INSIDE_RELATIONS = ("inside",)
PLAN_STEP_FIELDS = {"action", "object", "location", "relation", "source"}

NEEDS_OBJECT = {
    "PickUpAction",
    "PlaceAction",
    "TransportAction",
    "OpenAction",
    "CloseAction",
}

NEEDS_LOCATION = {
    "NavigateAction",
    "PlaceAction",
    "TransportAction",
}

RELATIONS_BY_ACTION = {
    "PlaceAction": {"on", "inside", *DIRECTIONAL_RELATIONS},
    "TransportAction": {"on", "inside", *DIRECTIONAL_RELATIONS},
}
SOURCE_ACTIONS = {"PickUpAction", "TransportAction"}
DESTINATION_ACTIONS = {"PlaceAction", "TransportAction"}

__all__ = [
    "CONTEXT_KEYS",
    "VALID_PYCRAM_ACTIONS",
    "DIRECTIONAL_RELATIONS",
    "INSIDE_RELATIONS",
    "PLAN_STEP_FIELDS",
    "NEEDS_OBJECT",
    "NEEDS_LOCATION",
    "RELATIONS_BY_ACTION",
    "SOURCE_ACTIONS",
    "DESTINATION_ACTIONS",
    "format_choices",
    "PlanStep",
    "parse_plan",
    "parse_clarification",
]


def format_choices(values):
    """Return the given names sorted and joined by ", "."""
    return ", ".join(sorted(values))


@dataclass
class PlanStep:
    action: str
    object: str | None = None
    location: str | None = None
    relation: str | None = None
    source: str | None = None

    def validate(self):
        """Validate this step against its action-specific schema."""
        if self.action not in VALID_PYCRAM_ACTIONS:
            raise ValueError(
                f"Unknown action {self.action!r}. "
                f"Valid actions: {format_choices(VALID_PYCRAM_ACTIONS)}"
            )

        for field in ("object", "location", "source"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be a non-empty string or null")

        if self.action in NEEDS_OBJECT:
            if self.object is None:
                raise ValueError(
                    f"{self.action} requires an 'object' (the thing to act on)"
                )
        elif self.object is not None:
            raise ValueError(f"{self.action} requires 'object' to be null")

        if self.action in NEEDS_LOCATION:
            if self.location is None:
                raise ValueError(
                    f"{self.action} requires a 'location' (where to go/place)"
                )
        elif self.location is not None:
            raise ValueError(f"{self.action} requires 'location' to be null")

        allowed_relations = RELATIONS_BY_ACTION.get(self.action)
        if allowed_relations is None:
            if self.relation is not None:
                raise ValueError(f"{self.action} requires 'relation' to be null")
        elif self.relation not in allowed_relations:
            raise ValueError(
                f"{self.action} relation must be one of: "
                f"{format_choices(allowed_relations)}"
            )

        if self.action not in SOURCE_ACTIONS and self.source is not None:
            raise ValueError(f"{self.action} requires 'source' to be null")

    def as_dict(self):
        """Return this step as a plain five-field plan dict."""
        return {
            "action": self.action,
            "object": self.object,
            "location": self.location,
            "relation": self.relation,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data):
        """Build a validated step from an exact five-field object."""
        if not isinstance(data, dict):
            raise ValueError("Each plan step must be a JSON object")
        fields = set(data)
        if fields != PLAN_STEP_FIELDS:
            missing = sorted(PLAN_STEP_FIELDS - fields)
            extra = sorted(fields - PLAN_STEP_FIELDS)
            raise ValueError(
                f"Plan step fields must be exactly {sorted(PLAN_STEP_FIELDS)}; "
                f"missing={missing}, extra={extra}"
            )
        plan_step = cls(
            action=data["action"],
            object=data["object"],
            location=data["location"],
            relation=data["relation"],
            source=data["source"],
        )
        plan_step.validate()
        return plan_step


def parse_plan(raw):
    """Parse one exact, non-empty plan response into PlanStep objects."""
    if not isinstance(raw, dict) or set(raw) != {"plan"}:
        raise ValueError("Response must contain exactly one top-level 'plan' field")
    if not isinstance(raw["plan"], list):
        raise ValueError("'plan' must be a JSON array")
    steps = [PlanStep.from_dict(step_data) for step_data in raw["plan"]]
    if not steps:
        raise ValueError("A valid plan must have at least one step")
    return steps


def parse_clarification(raw):
    """Parse one exact clarification response into its question string."""
    if not isinstance(raw, dict) or set(raw) != {"clarification"}:
        raise ValueError(
            "Response must contain exactly one top-level 'clarification' field"
        )
    question = raw["clarification"]
    if not isinstance(question, str):
        raise ValueError("Clarification must be a question string")
    if len(question.split()) < 3:
        raise ValueError("Clarification question is too short (need at least 3 words)")
    return question
