from dataclasses import dataclass, field

from thesis_demo.validation.schema import (
    DESTINATION_ACTIONS,
    DIRECTIONAL_RELATIONS,
    INSIDE_RELATIONS,
    SOURCE_ACTIONS,
    format_choices,
    parse_clarification,
    parse_plan,
)


@dataclass(frozen=True, eq=False)
class ContextNames:
    objects: set
    surfaces: set
    containers: set
    openables: set
    places: set
    sources: set

    def __iter__(self):
        return iter(self._values())

    def __eq__(self, other):
        if isinstance(other, ContextNames):
            return self._values() == other._values()
        if isinstance(other, tuple):
            return self._values() == other
        return NotImplemented

    def _values(self):
        return (
            self.objects,
            self.surfaces,
            self.containers,
            self.openables,
            self.places,
            self.sources,
        )


def verify(response, context=None):
    try:
        if isinstance(response, dict) and set(response) == {"clarification"}:
            parse_clarification(response)
            return True, "ok"
        steps = parse_plan(response)
    except (KeyError, TypeError, ValueError) as error:
        return False, str(error)

    if context:
        is_valid, message = check_names(steps, context)
        if not is_valid:
            return False, message
    return check_sequence(steps, context)


def check_names(steps, context):
    names = context_names(context)
    for step in steps:
        message = _name_error(step, names, context)
        if message is not None:
            return False, message
    return True, "ok"


def check_sequence(steps, context=None):
    plan = {"plan": [step.as_dict() for step in steps]}
    validator = _SequenceValidator(
        context=context or {},
        required_containers=required_open_containers(plan, context),
    )
    return validator.validate(steps)


def context_names(context):
    context = context or {}
    objects = set(context.get("objects", []))
    containers = set(context.get("containers", []))
    surfaces = set(context.get("surfaces", []))
    openables = openable_names(context)
    places = (
        surfaces
        | containers
        | openables
        | set(context.get("furniture", []))
        | set(context.get("rooms", []))
    )
    return ContextNames(
        objects=objects,
        surfaces=surfaces,
        containers=containers,
        openables=openables,
        places=places,
        sources=surfaces | containers,
    )


def openable_names(context):
    return set((context or {}).get("openables", []))


def allowed_objects_for(action, names):
    if action in ("OpenAction", "CloseAction"):
        return names.openables
    return names.objects


def allowed_locations_for(action, relation, names):
    if relation in DIRECTIONAL_RELATIONS:
        return names.objects
    if relation in INSIDE_RELATIONS:
        return names.containers
    if relation == "on":
        return names.surfaces
    if action == "NavigateAction":
        return names.places
    return set()


def required_open_containers(plan, context):
    openables = openable_names(context)
    return {
        container
        for step in plan.get("plan", [])
        if isinstance(step, dict)
        for container in openables
        if accesses_container(step, container)
    }


def accesses_container(step, container):
    return step.get("source") == container or is_inside_destination(step, container)


def is_inside_destination(step, container):
    return (
        step.get("location") == container and step.get("relation") in INSIDE_RELATIONS
    )


def _name_error(step, names, context):
    allowed_objects = allowed_objects_for(step.action, names)
    if step.object and step.object not in allowed_objects:
        return (
            f"unknown object {step.object!r}, "
            f"choose from: {format_choices(allowed_objects)}"
        )

    allowed_locations = allowed_locations_for(step.action, step.relation, names)
    if step.location and step.location not in allowed_locations:
        return (
            f"unknown {_location_kind(step.relation)} {step.location!r}, "
            f"choose from: {format_choices(allowed_locations)}"
        )

    if step.source and step.source not in names.sources:
        return (
            f"unknown source {step.source!r}, "
            f"choose from: {format_choices(names.sources)}"
        )

    if step.action not in SOURCE_ACTIONS or step.source:
        return None

    locations = context.get("object_locations", {}).get(step.object, [])
    if len(locations) > 1:
        return (
            f"ambiguous source for {step.object!r}, "
            f"choose from: {format_choices(locations)}"
        )

    container_locations = [
        location for location in locations if location in names.openables
    ]
    if not container_locations:
        return None
    return (
        f"missing source for {step.object!r} in container "
        f"{container_locations[0]!r}; name the source and open it before access"
    )


def _location_kind(relation):
    if relation in DIRECTIONAL_RELATIONS:
        return "reference object"
    if relation in INSIDE_RELATIONS:
        return "container"
    if relation == "on":
        return "surface"
    return "location"


@dataclass
class _SequenceValidator:
    context: dict
    required_containers: set
    current_location: str | None = None
    opened_containers: set = field(default_factory=set)
    held_objects: set = field(default_factory=set)
    held_sources: dict = field(default_factory=dict)

    def validate(self, steps):
        for step in steps:
            message = self._apply(step)
            if message is not None:
                return False, message
        return self._validate_completion()

    def _apply(self, step):
        if step.action == "NavigateAction":
            self.current_location = step.location
            return None

        message = self._action_error(step)
        if message is not None:
            return message

        self._remember_held_source(step)
        return self._openable_access_error(step)

    def _action_error(self, step):
        if step.action == "OpenAction":
            return self._open_error(step)
        if step.action == "CloseAction":
            return self._close_error(step)
        if step.action == "PickUpAction":
            return self._pick_up_error(step)
        if step.action == "PlaceAction":
            return self._place_error(step)
        if step.action == "TransportAction":
            return self._transport_error(step)
        return None

    def _open_error(self, step):
        if self.current_location != step.object:
            return (
                f"{step.action}({step.object!r}) "
                "without navigation to the same container"
            )
        if step.object in self.opened_containers:
            return f"unnecessary OpenAction for already-opened {step.object!r}"
        self.opened_containers.add(step.object)
        return None

    def _close_error(self, step):
        held_here = {
            object_name
            for object_name, source in self.held_sources.items()
            if object_name in self.held_objects and source == step.object
        }
        if held_here:
            return (
                f"CloseAction({step.object!r}) while holding "
                f"{format_choices(held_here)} from that container"
            )
        if self.current_location != step.object:
            return (
                f"{step.action}({step.object!r}) "
                "without navigation to the same container"
            )
        if (
            step.object in self.required_containers
            and step.object not in self.opened_containers
        ):
            return f"CloseAction({step.object!r}) without a preceding OpenAction"
        self.opened_containers.discard(step.object)
        return None

    def _pick_up_error(self, step):
        locations = set(self.context.get("object_locations", {}).get(step.object, []))
        expected_locations = {step.source} if step.source else locations
        if expected_locations and self.current_location not in expected_locations:
            return (
                f"PickUpAction({step.object!r}) without navigation "
                f"to {format_choices(expected_locations)}"
            )
        if self.held_objects:
            return (
                f"PickUpAction({step.object!r}) while the gripper already holds "
                f"{format_choices(self.held_objects)}"
            )
        self.held_objects.add(step.object)
        return None

    def _place_error(self, step):
        expected_locations = {step.location}
        if step.relation in DIRECTIONAL_RELATIONS:
            expected_locations = set(
                self.context.get("object_locations", {}).get(step.location, [])
            )
        if expected_locations and self.current_location not in expected_locations:
            return (
                f"PlaceAction({step.object!r}) without navigation to "
                f"{format_choices(expected_locations)}"
            )
        if step.object not in self.held_objects:
            return f"PlaceAction({step.object!r}) without a preceding PickUpAction"
        self.held_objects.discard(step.object)
        return None

    def _transport_error(self, step):
        if not self.held_objects:
            return None
        return (
            f"TransportAction({step.object!r}) while the gripper already holds "
            f"{format_choices(self.held_objects)}"
        )

    def _remember_held_source(self, step):
        if step.action == "PlaceAction":
            self.held_sources.pop(step.object, None)
            return
        if step.action != "PickUpAction":
            return

        locations = self.context.get("object_locations", {}).get(step.object, [])
        source = step.source
        if source is None and len(locations) == 1:
            source = locations[0]
        self.held_sources[step.object] = source

    def _openable_access_error(self, step):
        if (
            step.action in DESTINATION_ACTIONS
            and step.relation in INSIDE_RELATIONS
            and step.location in self.required_containers
            and step.location not in self.opened_containers
        ):
            return (
                f"{step.action}(relation={step.relation!r}, "
                f"location={step.location!r}) without preceding OpenAction"
            )
        if (
            step.action in SOURCE_ACTIONS
            and step.source
            and step.source in self.required_containers
            and step.source not in self.opened_containers
        ):
            return f"{step.action}(source={step.source!r}) without preceding OpenAction"
        return None

    def _validate_completion(self):
        allowed_open_containers = {
            source
            for object_name, source in self.held_sources.items()
            if object_name in self.held_objects and source in self.required_containers
        }
        containers_to_close = self.required_containers - allowed_open_containers
        for container in sorted(containers_to_close):
            if container in self.opened_containers:
                return False, f"container {container!r} must be closed after access"
        return True, "ok"
