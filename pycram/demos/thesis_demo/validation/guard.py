from dataclasses import dataclass, field

from thesis_demo.validation.schema import (
    CONTAINER_ACTIONS,
    GRASPING_ACTIONS,
    description_key,
    render,
    DESTINATION_ACTIONS,
    DIRECTIONAL_RELATIONS,
    INSIDE_RELATIONS,
    SOURCE_ACTIONS,
    format_choices,
    parse_clarification,
    parse_plan,
)


@dataclass(frozen=True)
class ContextNames:
    objects: set
    surfaces: set
    containers: set
    openables: set
    places: set
    sources: set


@dataclass(frozen=True)
class GuardReport:
    is_valid: bool
    message: str
    names_valid: bool = None
    sequence_valid: bool = None


def inspect(response, context=None):
    try:
        if isinstance(response, dict) and set(response) == {"clarification"}:
            parse_clarification(response)
            return GuardReport(True, "ok")
        steps = parse_plan(response)
    except (KeyError, TypeError, ValueError) as error:
        return GuardReport(False, str(error))

    # Both checks always run so each can be reported on its own, even though
    # only the first failure is worth telling the planner about.
    names_valid, names_message = (True, "ok")
    if context:
        names_valid, names_message = check_names(steps, context)
    sequence_valid, sequence_message = check_sequence(steps, context)

    message = names_message if not names_valid else sequence_message
    return GuardReport(
        is_valid=names_valid and sequence_valid,
        message=message,
        names_valid=names_valid,
        sequence_valid=sequence_valid,
    )


def verify(response, context=None):
    report = inspect(response, context)
    return report.is_valid, report.message


def check_names(steps, context):
    names = context_names(context)
    for step in steps:
        message = _resolution_error(step, names, context)
        if message is not None:
            return False, message
    return True, "ok"


def check_sequence(steps, context=None):
    validator = _SequenceValidator(
        context=context or {},
        required_containers=required_open_containers(steps, context),
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


def allowed_objects_for(action_name, names):
    if action_name in ("OpenAction", "CloseAction"):
        return names.openables
    return names.objects


def allowed_locations_for(action_name, relation, names):
    if relation in DIRECTIONAL_RELATIONS:
        return names.objects
    if relation in INSIDE_RELATIONS:
        return names.containers
    if relation == "on":
        return names.surfaces
    if action_name == "NavigateAction":
        return names.places
    return set()


def required_open_containers(steps, context):
    openables = openable_names(context)
    reached = set()
    for step in steps:
        for description in accessed_containers(step):
            if description["type"] in openables:
                reached.add(description_key(description))
    return reached


def accessed_containers(step):
    if step.source is not None:
        yield step.source
    if step.location is not None and step.relation in INSIDE_RELATIONS:
        yield step.location


def entity_type(description):
    return description["type"] if description else None


def _resolution_error(step, names, context):
    description_checks = (
        (
            "object",
            step.object,
            allowed_objects_for(step.action, names),
            step.action not in ("OpenAction", "CloseAction"),
        ),
        (
            _location_kind(step.relation),
            step.location,
            allowed_locations_for(step.action, step.relation, names),
            step.relation in DIRECTIONAL_RELATIONS,
        ),
        ("source", step.source, names.sources, False),
    )
    for (
        description_role,
        description,
        allowed_types,
        must_be_identified,
    ) in description_checks:
        if description is None:
            continue
        message = _describes_entities(
            description_role,
            description,
            allowed_types,
            context,
            must_be_identified,
        )
        if message is not None:
            return message
    return _unnamed_source_error(step, names, context)


def _unnamed_source_error(step, names, context):
    if step.action not in SOURCE_ACTIONS or step.source is not None:
        return None

    places = context.get("object_locations", {}).get(entity_type(step.object), [])
    if len(places) > 1:
        return (
            f"ambiguous source for {render(step.object)}, "
            f"choose from: {format_choices(places)}"
        )
    shut_away = [place for place in places if place in names.openables]
    if not shut_away:
        return None
    return (
        f"missing source for {render(step.object)} in container "
        f"{shut_away[0]!r}; name the source and open it before access"
    )


def _describes_entities(
    description_role,
    description,
    allowed_types,
    context,
    must_be_identified,
):
    entity_type = description["type"]
    if entity_type not in allowed_types:
        return (
            f"unknown {description_role} type {entity_type!r}, "
            f"choose from: {format_choices(allowed_types)}"
        )

    instances = context.get("instances", {}).get(entity_type)
    if not instances:
        return None

    qualifiers = {key: value for key, value in description.items() if key != "type"}
    matching_instances = [
        instance
        for instance in instances
        if all(instance.get(key) == value for key, value in qualifiers.items())
    ]
    if not matching_instances:
        return (
            f"no {entity_type} matches {qualifiers!r}; " f"choose from: {instances!r}"
        )
    if must_be_identified and len(matching_instances) > 1:
        return (
            f"ambiguous {description_role}: "
            f"{len(matching_instances)} instances of {entity_type} match "
            f"{qualifiers!r}; ask which one is meant"
        )
    return None


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
    current_location: dict | None = None
    opened_containers: set = field(default_factory=set)
    held_objects: set = field(default_factory=set)
    held_sources: dict = field(default_factory=dict)
    arms_are_extended: bool = False

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

        if step.action == "ParkArmsAction":
            self.arms_are_extended = False
            return None

        message = self._action_error(step)
        if message is not None:
            return message

        self._remember_held_source(step)
        return self._openable_access_error(step)

    def _action_error(self, step):
        if step.action in GRASPING_ACTIONS and self.arms_are_extended:
            return (
                f"{step.action}({render(step.object)}) with the arms still "
                "reaching into a container; park the arms after opening or "
                "closing one"
            )
        if step.action in CONTAINER_ACTIONS:
            self.arms_are_extended = True
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

    def _at(self, description):
        return description_key(self.current_location) == description_key(description)

    def _open_error(self, step):
        if not self._at(step.object):
            return (
                f"{step.action}({render(step.object)}) "
                "without navigation to the same container"
            )
        if description_key(step.object) in self.opened_containers:
            return f"unnecessary OpenAction for already-opened {render(step.object)}"
        self.opened_containers.add(description_key(step.object))
        return None

    def _close_error(self, step):
        container = description_key(step.object)
        held_here = {
            name
            for name, source in self.held_sources.items()
            if name in self.held_objects and source == container
        }
        if held_here:
            return (
                f"CloseAction({render(step.object)}) while still holding "
                "something taken from it"
            )
        if not self._at(step.object):
            return (
                f"{step.action}({render(step.object)}) "
                "without navigation to the same container"
            )
        if (
            container in self.required_containers
            and container not in self.opened_containers
        ):
            return f"CloseAction({render(step.object)}) without a preceding OpenAction"
        self.opened_containers.discard(container)
        return None

    def _pick_up_error(self, step):
        expected_locations = (
            {entity_type(step.source)}
            if step.source is not None
            else set(
                self.context.get("object_locations", {}).get(
                    entity_type(step.object),
                    [],
                )
            )
        )
        if (
            expected_locations
            and entity_type(self.current_location) not in expected_locations
        ):
            return (
                f"PickUpAction({render(step.object)}) without navigation "
                f"to {format_choices(expected_locations)}"
            )
        if self.held_objects:
            return (
                f"PickUpAction({render(step.object)}) "
                "while the gripper already holds something"
            )
        self.held_objects.add(description_key(step.object))
        return None

    def _place_error(self, step):
        expected_locations = {entity_type(step.location)}
        if step.relation in DIRECTIONAL_RELATIONS:
            expected_locations = set(
                self.context.get("object_locations", {}).get(
                    entity_type(step.location), []
                )
            )
        if (
            expected_locations
            and entity_type(self.current_location) not in expected_locations
        ):
            return (
                f"PlaceAction({render(step.object)}) without navigation to "
                f"{format_choices(expected_locations)}"
            )
        if description_key(step.object) not in self.held_objects:
            return (
                f"PlaceAction({render(step.object)}) without a preceding PickUpAction"
            )
        self.held_objects.discard(description_key(step.object))
        return None

    def _transport_error(self, step):
        if not self.held_objects:
            return None
        return (
            f"TransportAction({render(step.object)}) "
            "while the gripper already holds something"
        )

    def _remember_held_source(self, step):
        if step.action == "PlaceAction":
            self.held_sources.pop(description_key(step.object), None)
            return
        if step.action != "PickUpAction":
            return

        source = description_key(step.source)
        if source is None:
            places = self.context.get("object_locations", {}).get(
                entity_type(step.object), []
            )
            if len(places) == 1:
                source = places[0]
        self.held_sources[description_key(step.object)] = source

    def _openable_access_error(self, step):
        if (
            step.action in DESTINATION_ACTIONS
            and step.relation in INSIDE_RELATIONS
            and description_key(step.location) in self.required_containers
            and description_key(step.location) not in self.opened_containers
        ):
            return (
                f"{step.action}(relation={step.relation!r}, "
                f"location={render(step.location)}) without preceding OpenAction"
            )
        if (
            step.action in SOURCE_ACTIONS
            and step.source is not None
            and description_key(step.source) in self.required_containers
            and description_key(step.source) not in self.opened_containers
        ):
            return (
                f"{step.action}(source={render(step.source)}) "
                "without preceding OpenAction"
            )
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
                return (
                    False,
                    f"container {render(dict(container))} must be closed after access",
                )
        return True, "ok"
