from collections import namedtuple

from .schema import (
    DESTINATION_ACTIONS,
    DIRECTIONAL_RELATIONS,
    INSIDE_RELATIONS,
    SOURCE_ACTIONS,
    format_choices,
    parse_clarification,
    parse_plan,
)

ContextNames = namedtuple(
    "ContextNames",
    ("objects", "surfaces", "containers", "openables", "places", "sources"),
)


def openable_names(ctx):
    """Return entities that the current world exposes as openable."""
    return set((ctx or {}).get("openables", []))


def context_names(ctx):
    """Return planner name sets grouped by their world-context capabilities."""
    ctx = ctx or {}
    objects = set(ctx.get("objects", []))
    containers = set(ctx.get("containers", []))
    surfaces = set(ctx.get("surfaces", []))
    openables = openable_names(ctx)
    places = (
        surfaces
        | containers
        | openables
        | set(ctx.get("furniture", []))
        | set(ctx.get("rooms", []))
    )
    return ContextNames(
        objects,
        surfaces,
        containers,
        openables,
        places,
        surfaces | containers,
    )


def allowed_objects_for(action, names):
    """Return valid object IDs for a planner action."""
    if action in ("OpenAction", "CloseAction"):
        return names.openables
    return names.objects


def allowed_locations_for(action, relation, names):
    """Return valid location IDs for a planner action and relation."""
    if relation in DIRECTIONAL_RELATIONS:
        return names.objects
    if relation in INSIDE_RELATIONS:
        return names.containers
    if relation == "on":
        return names.surfaces
    if action == "NavigateAction":
        return names.places
    return set()


def is_inside_destination(step, container):
    """Return whether a step targets the container with an inside relation."""
    return (
        step.get("location") == container and step.get("relation") in INSIDE_RELATIONS
    )


def accesses_container(step, container):
    """Return whether the container is a source or inside destination."""
    return step.get("source") == container or is_inside_destination(step, container)


def required_open_containers(raw_plan, ctx):
    """Return openable containers accessed by a raw plan."""
    ctx = ctx or {}
    openables = openable_names(ctx)
    required = set()
    for step in raw_plan.get("plan", []):
        if not isinstance(step, dict):
            continue
        for container in openables:
            if accesses_container(step, container):
                required.add(container)
    return required


def check_sequence(steps, ctx=None):
    """Simulate plan execution and validate state transitions.

    Tracks the current location, opened containers, and held objects/sources
    while walking the steps. Returns a ``(valid, message)`` tuple.
    """
    ctx = ctx or {}
    raw_plan = {"plan": [step.as_dict() for step in steps]}
    required = required_open_containers(raw_plan, ctx)
    current_location = None
    opened_containers = set()
    held_objects = set()
    held_sources = {}

    for step in steps:
        if step.action == "NavigateAction":
            current_location = step.location
            continue

        if step.action == "OpenAction":
            if current_location != step.object:
                return False, (
                    f"{step.action}({step.object!r}) "
                    "without navigation to the same container"
                )
            if step.object in opened_containers:
                return False, (
                    f"unnecessary OpenAction for already-opened {step.object!r}"
                )
            opened_containers.add(step.object)

        elif step.action == "CloseAction":
            held_here = {
                object_name
                for object_name, source in held_sources.items()
                if object_name in held_objects and source == step.object
            }
            if held_here:
                return False, (
                    f"CloseAction({step.object!r}) while holding "
                    f"{format_choices(held_here)} from that container"
                )
            if current_location != step.object:
                return False, (
                    f"{step.action}({step.object!r}) "
                    "without navigation to the same container"
                )
            if step.object in required and step.object not in opened_containers:
                return False, (
                    f"CloseAction({step.object!r}) without a preceding OpenAction"
                )
            opened_containers.discard(step.object)

        elif step.action == "PickUpAction":
            possible = set(ctx.get("object_locations", {}).get(step.object, []))
            expected_locations = {step.source} if step.source else possible
            if expected_locations and current_location not in expected_locations:
                return False, (
                    f"PickUpAction({step.object!r}) without navigation "
                    f"to {format_choices(expected_locations)}"
                )
            if held_objects:
                return False, (
                    f"PickUpAction({step.object!r}) while the gripper already holds "
                    f"{format_choices(held_objects)}"
                )
            held_objects.add(step.object)

        elif step.action == "PlaceAction":
            expected_locations = {step.location}
            if step.relation in DIRECTIONAL_RELATIONS:
                expected_locations = set(
                    ctx.get("object_locations", {}).get(step.location, [])
                )
            if expected_locations and current_location not in expected_locations:
                return False, (
                    f"PlaceAction({step.object!r}) without navigation to "
                    f"{format_choices(expected_locations)}"
                )
            if step.object not in held_objects:
                return False, (
                    f"PlaceAction({step.object!r}) without a preceding PickUpAction"
                )
            held_objects.discard(step.object)

        elif step.action == "TransportAction" and held_objects:
            return False, (
                f"TransportAction({step.object!r}) while the gripper already holds "
                f"{format_choices(held_objects)}"
            )

        # Remember where a held object came from until it is placed.
        if step.action == "PickUpAction":
            locations = ctx.get("object_locations", {}).get(step.object, [])
            source = step.source
            if not source and len(locations) == 1:
                source = locations[0]
            held_sources[step.object] = source
        elif step.action == "PlaceAction":
            held_sources.pop(step.object, None)

        # Accessing an openable container requires a preceding OpenAction.
        if (
            step.action in DESTINATION_ACTIONS
            and step.relation in INSIDE_RELATIONS
            and step.location in required
            and step.location not in opened_containers
        ):
            return False, (
                f"{step.action}(relation={step.relation!r}, "
                f"location={step.location!r}) without preceding OpenAction"
            )
        if (
            step.action in SOURCE_ACTIONS
            and step.source
            and step.source in required
            and step.source not in opened_containers
        ):
            return False, (
                f"{step.action}(source={step.source!r}) without preceding OpenAction"
            )

    # A source may stay open only while its object is still held.
    allowed_open = {
        source
        for object_name, source in held_sources.items()
        if object_name in held_objects and source in required
    }
    for container in sorted(required - allowed_open):
        if container in opened_containers:
            return False, f"container {container!r} must be closed after access"
    return True, "ok"


def check_names(steps, ctx):
    """Validate every plan-step name against the world context.

    Returns a ``(valid, message)`` tuple.
    """
    names = context_names(ctx)

    for step in steps:
        allowed_objects = allowed_objects_for(step.action, names)
        if step.object and step.object not in allowed_objects:
            return False, (
                f"unknown object {step.object!r}, "
                f"choose from: {format_choices(allowed_objects)}"
            )

        allowed = allowed_locations_for(step.action, step.relation, names)
        if step.location and step.location not in allowed:
            if step.relation in DIRECTIONAL_RELATIONS:
                location_kind = "reference object"
            elif step.relation in INSIDE_RELATIONS:
                location_kind = "container"
            elif step.relation == "on":
                location_kind = "surface"
            else:
                location_kind = "location"
            return False, (
                f"unknown {location_kind} {step.location!r}, "
                f"choose from: {format_choices(allowed)}"
            )

        if step.source and step.source not in names.sources:
            return False, (
                f"unknown source {step.source!r}, "
                f"choose from: {format_choices(names.sources)}"
            )

        if step.action in SOURCE_ACTIONS and not step.source:
            locations = ctx.get("object_locations", {}).get(step.object, [])
            if len(locations) > 1:
                return False, (
                    f"ambiguous source for {step.object!r}, "
                    f"choose from: {format_choices(locations)}"
                )
            container_locations = [
                location for location in locations if location in names.openables
            ]
            if container_locations:
                return False, (
                    f"missing source for {step.object!r} in container "
                    f"{container_locations[0]!r}; "
                    "name the source and open it before access"
                )

    return True, "ok"


def verify(raw, ctx=None):
    try:
        if isinstance(raw, dict) and set(raw) == {"clarification"}:
            parse_clarification(raw)
            return True, "ok"
        steps = parse_plan(raw)
    except (KeyError, TypeError, ValueError) as error:
        return False, str(error)

    if ctx:
        is_valid, message = check_names(steps, ctx)
        if not is_valid:
            return False, message

    return check_sequence(steps, ctx)
