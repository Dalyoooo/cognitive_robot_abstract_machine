from dataclasses import dataclass, field
from types import MappingProxyType

CONTEXT_KEYS = (
    "instances",
    "objects",
    "object_locations",
    "surfaces",
    "containers",
    "openables",
    "furniture",
    "rooms",
    "types",
)

DIRECTIONAL_RELATIONS = ("left_of", "right_of", "in_front_of", "behind")
DESCRIPTION_QUALIFIERS = ("at", "contains")
INSIDE_RELATIONS = ("inside",)
PLAN_STEP_FIELDS = {"action", "object", "location", "relation", "source"}


@dataclass(frozen=True)
class ActionSpec:
    required_fields: set = field(default_factory=set)
    optional_fields: set = field(default_factory=set)
    relations: set = field(default_factory=set)


ACTION_SPECS = MappingProxyType(
    {
        "TransportAction": ActionSpec(
            required_fields={"object", "location", "relation"},
            optional_fields={"source"},
            relations={"on", "inside", *DIRECTIONAL_RELATIONS},
        ),
        "PickUpAction": ActionSpec(
            required_fields={"object"},
            optional_fields={"source"},
        ),
        "PlaceAction": ActionSpec(
            required_fields={"object", "location", "relation"},
            relations={"on", "inside", *DIRECTIONAL_RELATIONS},
        ),
        "NavigateAction": ActionSpec(
            required_fields={"location"},
        ),
        "OpenAction": ActionSpec(
            required_fields={"object"},
        ),
        "CloseAction": ActionSpec(
            required_fields={"object"},
        ),
        "ParkArmsAction": ActionSpec(),
    }
)

VALID_PYCRAM_ACTIONS = set(ACTION_SPECS)

SOURCE_ACTIONS = {
    name for name, spec in ACTION_SPECS.items() if "source" in spec.optional_fields
}

DESTINATION_ACTIONS = {
    name
    for name, spec in ACTION_SPECS.items()
    if {"object", "location"} <= spec.required_fields
}

# Working a container leaves the arm stretched over it; taking hold of anything
# needs it back in its parked pose first.
CONTAINER_ACTIONS = {"OpenAction", "CloseAction"}
GRASPING_ACTIONS = {"PickUpAction", "PlaceAction", "TransportAction"}


def format_choices(values):
    return ", ".join(sorted(values))


def _validate_description(field_name, description):
    if not isinstance(description, dict):
        raise ValueError(f"{field_name} must be an object describing an entity")

    entity_type = description.get("type")
    if not isinstance(entity_type, str) or not entity_type:
        raise ValueError(f"{field_name} needs a non-empty 'type'")

    unknown = sorted(set(description) - {"type", *DESCRIPTION_QUALIFIERS})
    if unknown:
        raise ValueError(
            f"{field_name} has unknown qualifier(s) {unknown}; "
            f"valid: {format_choices(DESCRIPTION_QUALIFIERS)}"
        )
    for qualifier in DESCRIPTION_QUALIFIERS:
        value = description.get(qualifier)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{field_name}.{qualifier} must be a non-empty string")


def render(description):
    if description is None:
        return "nothing"
    qualifiers = ", ".join(
        f"{key}={value}" for key, value in sorted(description.items()) if key != "type"
    )
    return f"{description['type']}({qualifiers})" if qualifiers else description["type"]


def description_key(description):
    if description is None:
        return None
    return tuple(sorted(description.items()))


@dataclass
class PlanStep:
    action: str
    object: dict | None = None
    location: dict | None = None
    relation: str | None = None
    source: dict | None = None

    def validate(self):
        self._validate_action()
        self._validate_names()
        spec = ACTION_SPECS[self.action]
        self._validate_required_field(
            spec, "object", self.object, "an 'object' (the thing to act on)"
        )
        self._validate_required_field(
            spec, "location", self.location, "a 'location' (where to go/place)"
        )
        self._validate_relation(spec)
        self._validate_source(spec)

    def as_dict(self):
        return {
            "action": self.action,
            "object": self.object,
            "location": self.location,
            "relation": self.relation,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data):
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

    def _validate_action(self):
        if not isinstance(self.action, str) or not self.action:
            raise ValueError("action must be a non-empty string")
        if self.action not in VALID_PYCRAM_ACTIONS:
            raise ValueError(
                f"Unknown action {self.action!r}. "
                f"Valid actions: {format_choices(VALID_PYCRAM_ACTIONS)}"
            )

    def _validate_names(self):
        if self.relation is not None and (
            not isinstance(self.relation, str) or not self.relation
        ):
            raise ValueError("relation must be a non-empty string or null")
        if self.object is not None:
            _validate_description("object", self.object)
        if self.location is not None:
            _validate_description("location", self.location)
        if self.source is not None:
            _validate_description("source", self.source)

    def _validate_required_field(self, spec, field_name, value, hint):
        if field_name in spec.required_fields and value is None:
            raise ValueError(f"{self.action} requires {hint}")
        if field_name not in spec.required_fields and value is not None:
            raise ValueError(f"{self.action} requires {field_name!r} to be null")

    def _validate_relation(self, spec):
        if not spec.relations and self.relation is not None:
            raise ValueError(f"{self.action} requires 'relation' to be null")
        if spec.relations and self.relation not in spec.relations:
            raise ValueError(
                f"{self.action} relation must be one of: "
                f"{format_choices(spec.relations)}"
            )

    def _validate_source(self, spec):
        if "source" not in spec.optional_fields and self.source is not None:
            raise ValueError(f"{self.action} requires 'source' to be null")


def parse_plan(response):
    if not isinstance(response, dict) or set(response) != {"plan"}:
        raise ValueError("Response must contain exactly one top-level 'plan' field")
    if not isinstance(response["plan"], list):
        raise ValueError("'plan' must be a JSON array")
    steps = [PlanStep.from_dict(step_data) for step_data in response["plan"]]
    if not steps:
        raise ValueError("A valid plan must have at least one step")
    return steps


def parse_clarification(response):
    if not isinstance(response, dict) or set(response) != {"clarification"}:
        raise ValueError(
            "Response must contain exactly one top-level 'clarification' field"
        )
    question = response["clarification"]
    if not isinstance(question, str):
        raise ValueError("Clarification must be a question string")
    if len(question.split()) < 3:
        raise ValueError("Clarification question is too short (need at least 3 words)")
    return question
