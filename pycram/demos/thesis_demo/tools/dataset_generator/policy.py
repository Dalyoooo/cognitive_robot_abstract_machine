from thesis_demo.validation.schema import DIRECTIONAL_RELATIONS


def storage_containers(context, catalog):
    return tuple(
        name
        for name in context.containers
        if catalog.can_store_objects(context.types[name])
    )


def resolve_source(intent, context):
    locations = context.object_locations.get(intent.object, ())
    if intent.source is not None:
        return intent.source, len(locations)
    if not locations:
        raise ValueError(f"source is missing for {intent.object!r}")
    if len(locations) > 1:
        raise ValueError(f"source is ambiguous for {intent.object!r}: {locations}")
    return locations[0], len(locations)


def include_source(intent, source, location_count, context):
    return (
        intent.source_explicit
        or context.is_openable(source)
        or location_count > 1
    )


def check_destination(intent, context, catalog):
    destination = intent.destination
    if intent.relation == "on":
        if destination not in context.surfaces:
            raise ValueError("relation 'on' requires a surface destination")
        return destination
    if intent.relation == "inside":
        if destination not in storage_containers(context, catalog):
            raise ValueError("relation 'inside' requires a storage destination")
        return destination
    if intent.relation in DIRECTIONAL_RELATIONS:
        if destination not in context.objects:
            raise ValueError("directional relation requires another object")
        if destination == intent.object:
            raise ValueError("an object cannot be placed relative to itself")
        return destination
    raise ValueError(
        "relation must be on, inside, left_of, right_of, in_front_of, or behind"
    )
