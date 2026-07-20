from dataclasses import replace

from .domain import DIRECTIONAL_RELATIONS, Intent, PlanStep, Scenario
from .resolver import matching_names, natural_reference
from .world import storage_containers

TRANSPORT_FAMILIES = {
    "transport_surface_surface": ("surface", "surface"),
    "transport_surface_container": ("surface", "container"),
    "transport_container_surface": ("container", "surface"),
    "transport_container_container": ("container", "container"),
}

DIRECTIONAL_FAMILIES = {
    f"transport_{relation}": relation for relation in DIRECTIONAL_RELATIONS
}

MULTI_FAMILIES = (
    "multi_transport_same_destination",
    "multi_transport_distinct_surfaces",
    "multi_transport_mixed_destination",
    "multi_transport_container_swap",
)

FAMILIES = (
    *TRANSPORT_FAMILIES,
    *DIRECTIONAL_FAMILIES,
    "pickup_surface",
    "pickup_container",
    "pickup_place_surface",
    "pickup_place_container",
    "pickup_place_directional",
    "navigate",
    "open",
    "close",
    *MULTI_FAMILIES,
    "clarify_missing_object",
    "clarify_open_container",
    "clarify_object",
    "clarify_destination",
    "clarify_source",
)


def _replace_location(context, object_name, locations):
    updated = dict(context.object_locations)
    updated[object_name] = locations
    return replace(context, object_locations=updated)


def _references(object_text=None, source_text=None, destination_text=None):
    """Build the three reference slots used by every scenario."""
    return {
        "object": object_text,
        "source": source_text,
        "destination": destination_text,
    }


def _is_instance_name(label, names):
    """Return whether a catalog label is also an exact world instance name."""
    normalised_label = label.strip().casefold()
    return any(normalised_label == name.strip().casefold() for name in names)


def _choices_except(names, *excluded):
    """Return choices in their original order, without excluded names."""
    excluded_names = set(excluded)
    return tuple(name for name in names if name not in excluded_names)


class ScenarioSampler:
    """Sample canonical task families without embedding language templates."""

    def __init__(self, catalog):
        self.catalog = catalog

    def _storage_places(self, context):
        """Return containers that can hold generated objects."""
        return storage_containers(context, self.catalog)

    def _object_places(self, context):
        """Return every surface or storage container that may hold an object."""
        return context.surfaces + self._storage_places(context)

    @staticmethod
    def _require_openables(context):
        """Return openable containers or explain why sampling cannot continue."""
        if not context.openables:
            raise ValueError("world has no openable container")
        return context.openables

    def sample(self, family, context, rng, serial, *, world_id="composed"):
        """
        Sample one canonical task from a scenario family.

        Args:
            family: Scenario family to sample.
            context: World context that supplies task entities.
            rng: Random generator used for deterministic choices.
            serial: Serial number used to construct the scenario ID.
            world_id: Identifier assigned to the scenario world.

        Returns:
            A scenario containing any context adjustments required by the family.

        Raises:
            ValueError: If the family is unknown or the context cannot support it.
        """
        if family not in FAMILIES:
            raise ValueError(f"unknown scenario family: {family}")
        if not context.objects:
            raise ValueError("scenario generation requires objects")

        if not context.surfaces and not self._storage_places(context):
            raise ValueError("scenario generation requires object storage places")

        extra_intents = ()
        extra_references = ()
        if family in TRANSPORT_FAMILIES:
            context, intent, references = self._transport(
                context,
                rng,
                *TRANSPORT_FAMILIES[family],
            )
        elif family in DIRECTIONAL_FAMILIES:
            context, intent, references = self._directional_transport(
                context,
                rng,
                DIRECTIONAL_FAMILIES[family],
            )
        elif family in {"pickup_surface", "pickup_container"}:
            source_role = "surface" if family.endswith("surface") else "container"
            context, intent, references = self._pickup(context, rng, source_role)
        elif family.startswith("pickup_place_"):
            context, intent, references = self._pickup_place(family, context, rng)
        elif family in {"navigate", "open", "close"}:
            intent, references = self._simple_action(family, context, rng, serial)
        elif family in MULTI_FAMILIES:
            context, intents, reference_sets = self._multi_transport(
                context, rng, family
            )
            intent, *remaining_intents = intents
            references, *remaining_references = reference_sets
            extra_intents = tuple(remaining_intents)
            extra_references = tuple(remaining_references)
        else:
            context, intent, references = self._clarification(family, context, rng)

        scenario_id = f"scenario-{serial:06d}"
        return Scenario(
            id=scenario_id,
            family=family,
            world_id=world_id,
            context=context,
            intent=intent,
            references=references,
            extra_intents=extra_intents,
            extra_references=extra_references,
        )

    def _simple_action(self, family, context, rng, serial):
        """Sample an action that does not move a household object."""
        if family == "navigate":
            destination = rng.choice(context.places)
            reference = natural_reference(
                destination, context.places, context, self.catalog
            )
            return Intent("navigate", destination=destination), _references(
                destination_text=reference
            )

        openable = self._require_openables(context)
        type_names = sorted({context.types[name] for name in openable})
        selected_type = type_names[serial % len(type_names)]
        matching_containers = tuple(
            name for name in openable if context.types[name] == selected_type
        )
        container = rng.choice(matching_containers)
        reference = natural_reference(
            container, context.containers, context, self.catalog
        )
        return Intent(family, object=container), _references(object_text=reference)

    def _clarification(self, family, context, rng):
        """Sample one of the single-task clarification families."""
        if family == "clarify_missing_object":
            return self._missing_object(context, rng)
        if family == "clarify_open_container":
            container, label = self._ambiguous_container(context, rng)
            return (
                context,
                Intent("open", object=container),
                _references(object_text=label),
            )
        if family == "clarify_object":
            return self._ambiguous_object_task(context, rng)
        if family == "clarify_source":
            return self._ambiguous_source_task(context, rng)
        return self._ambiguous_destination_task(context, rng)

    def _transport(self, context, rng, source_role, target_role):
        storage = self._storage_places(context)
        sources = context.surfaces if source_role == "surface" else storage
        targets = context.surfaces if target_role == "surface" else storage
        if not sources or not targets:
            raise ValueError("world has no transport endpoints")
        source = rng.choice(sources)
        destinations = _choices_except(targets, source)
        if not destinations:
            raise ValueError("transport needs different endpoints")
        destination = rng.choice(destinations)
        object_name = rng.choice(context.objects)
        context = _replace_location(context, object_name, (source,))
        explicit_source = rng.random() < 0.5

        intent = Intent(
            "transport",
            object=object_name,
            source=source,
            destination=destination,
            relation="on" if target_role == "surface" else "inside",
            source_explicit=explicit_source,
        )
        references = self._transport_references(context, intent)
        return context, intent, references

    def _directional_transport(self, context, rng, relation):
        object_name, reference_object = rng.sample(list(context.objects), 2)
        object_places = self._object_places(context)
        reference_surface = rng.choice(context.surfaces)
        source_choices = _choices_except(object_places, reference_surface)
        source = rng.choice(source_choices or object_places)
        context = _replace_location(context, object_name, (source,))
        context = _replace_location(context, reference_object, (reference_surface,))
        explicit_source = rng.random() < 0.5

        intent = Intent(
            "transport",
            object=object_name,
            source=source,
            destination=reference_object,
            relation=relation,
            source_explicit=explicit_source,
        )
        references = self._transport_references(context, intent)
        return context, intent, references

    def _pickup(self, context, rng, source_role):
        object_name = rng.choice(context.objects)
        sources = (
            context.surfaces
            if source_role == "surface"
            else self._storage_places(context)
        )
        if not sources:
            raise ValueError("world has no pickup source")
        source = rng.choice(sources)
        context = _replace_location(context, object_name, (source,))
        explicit_source = rng.random() < 0.5
        object_reference = natural_reference(
            object_name, context.objects, context, self.catalog
        )
        source_reference = None
        if explicit_source:
            source_reference = natural_reference(
                source, context.places, context, self.catalog
            )
        references = _references(
            object_text=object_reference,
            source_text=source_reference,
        )
        intent = Intent(
            "pickup",
            object=object_name,
            source=source,
            source_explicit=explicit_source,
        )
        return context, intent, references

    def _pickup_place(self, family, context, rng):
        object_name = rng.choice(context.objects)
        storage = self._storage_places(context)
        source = rng.choice(context.surfaces)
        context = _replace_location(context, object_name, (source,))

        if family == "pickup_place_surface":
            destination = rng.choice(_choices_except(context.surfaces, source))
            relation = "on"
        elif family == "pickup_place_container":
            if not storage:
                raise ValueError("world has no storage destination")
            destination = rng.choice(storage)
            relation = "inside"
        else:
            reference_object = rng.choice(_choices_except(context.objects, object_name))
            reference_surface = rng.choice(
                _choices_except(context.surfaces, source) or context.surfaces
            )
            context = _replace_location(context, reference_object, (reference_surface,))
            destination = reference_object
            relation = rng.choice(DIRECTIONAL_RELATIONS)

        intent = Intent(
            "pickup_place",
            object=object_name,
            source=source,
            destination=destination,
            relation=relation,
            source_explicit=True,
        )
        references = self._transport_references(context, intent)
        return context, intent, references

    def _multi_transport(self, context, rng, family):
        first_object, second_object = rng.sample(list(context.objects), 2)
        storage = self._storage_places(context)
        places = context.surfaces + storage
        if family == "multi_transport_same_destination":
            first_destination = second_destination = rng.choice(context.surfaces)
        elif family == "multi_transport_distinct_surfaces":
            first_destination, second_destination = rng.sample(
                list(context.surfaces), 2
            )
        else:
            first_destination = rng.choice(context.surfaces)
            if not storage:
                raise ValueError("world has no storage destination")
            second_destination = rng.choice(storage)

        if family == "multi_transport_container_swap":
            first_source = second_destination
            second_source = rng.choice(
                _choices_except(context.surfaces, first_destination) or context.surfaces
            )
        else:
            first_source = rng.choice(
                _choices_except(places, first_destination) or places
            )
            second_source = rng.choice(
                _choices_except(places, second_destination) or places
            )
        context = _replace_location(context, first_object, (first_source,))
        context = _replace_location(context, second_object, (second_source,))
        first_source_explicit = family == "multi_transport_container_swap"

        second_relation = "on"
        if family in {
            "multi_transport_mixed_destination",
            "multi_transport_container_swap",
        }:
            second_relation = "inside"

        first = Intent(
            "transport",
            object=first_object,
            source=first_source,
            destination=first_destination,
            relation="on",
            source_explicit=first_source_explicit,
        )
        second = Intent(
            "transport",
            object=second_object,
            source=second_source,
            destination=second_destination,
            relation=second_relation,
        )
        first_references = self._transport_references(context, first)
        second_references = self._transport_references(context, second)
        return context, (first, second), (first_references, second_references)

    def _missing_object(self, context, rng):
        object_name = rng.choice(context.objects)
        object_places = self._object_places(context)
        destination = rng.choice(context.surfaces)
        source_choices = _choices_except(object_places, destination)
        source = rng.choice(source_choices or object_places)
        context = _replace_location(context, object_name, (source,))
        references = _references(
            destination_text=natural_reference(
                destination, context.surfaces, context, self.catalog
            )
        )
        intent = Intent(
            "transport",
            object=object_name,
            source=source,
            destination=destination,
            relation="on",
        )
        return context, intent, references

    def _ambiguous_object_task(self, context, rng):
        object_name, label = self._ambiguous_object(context, rng)
        object_places = self._object_places(context)
        source = rng.choice(object_places)
        destination = rng.choice(_choices_except(object_places, source))
        context = _replace_location(context, object_name, (source,))
        intent = Intent(
            "transport",
            object=object_name,
            source=source,
            destination=destination,
            relation="on" if destination in context.surfaces else "inside",
        )
        references = self._transport_references(context, intent)
        references["object"] = label
        return context, intent, references

    def _ambiguous_source_task(self, context, rng):
        object_name = rng.choice(context.objects)
        object_places = self._object_places(context)
        source, second = rng.sample(list(object_places), 2)
        context = _replace_location(context, object_name, (source, second))
        destination = rng.choice(
            _choices_except(context.surfaces, source, second) or context.surfaces
        )
        intent = Intent(
            "transport",
            object=object_name,
            source=source,
            destination=destination,
            relation="on",
        )
        references = self._transport_references(context, intent)
        return context, intent, references

    def _ambiguous_destination_task(self, context, rng):
        object_name = rng.choice(context.objects)
        object_places = self._object_places(context)
        destination, label = self._ambiguous_destination(
            context,
            rng,
            object_places,
        )
        source = rng.choice(_choices_except(object_places, destination))
        context = _replace_location(context, object_name, (source,))
        intent = Intent(
            "transport",
            object=object_name,
            source=source,
            destination=destination,
            relation="on" if destination in context.surfaces else "inside",
        )
        references = self._transport_references(context, intent)
        references["destination"] = label
        return context, intent, references

    def _transport_references(self, context, intent):
        """Create spoken references for a fully specified transport intent."""
        object_name = intent.object
        source = intent.source
        destination = intent.destination
        object_reference = natural_reference(
            object_name, context.objects, context, self.catalog
        )
        source_reference = None
        if intent.source_explicit:
            source_reference = natural_reference(
                source, context.places, context, self.catalog
            )
        destination_names = (
            context.objects
            if destination in context.objects
            else context.names_for_role(context.role_of(destination))
        )
        destination_reference = natural_reference(
            destination, destination_names, context, self.catalog
        )
        return _references(
            object_text=object_reference,
            source_text=source_reference,
            destination_text=destination_reference,
        )

    def _ambiguous_destination(self, context, rng, destination_names):
        candidates = []
        for name in destination_names:
            role_names = context.names_for_role(context.role_of(name))
            for label in self.catalog.reference_labels(context.types[name]):
                if len(matching_names(label, role_names, context, self.catalog)) > 1:
                    candidates.append((name, label))
        if not candidates:
            raise ValueError("world has no ambiguous destination label")
        return rng.choice(candidates)

    def _ambiguous_container(self, context, rng):
        openable = self._require_openables(context)
        candidates = []
        for name in openable:
            for label in self.catalog.reference_labels(context.types[name]):
                matches = matching_names(label, openable, context, self.catalog)
                exact_instance = _is_instance_name(label, openable)
                if len(matches) > 1 and not exact_instance:
                    candidates.append((name, label))
        if not candidates:
            raise ValueError("world has no ambiguous openable-container label")
        return rng.choice(candidates)

    def _ambiguous_object(self, context, rng):
        candidates = []
        for name in context.objects:
            for label in self.catalog.labels_for(
                context.types[name], include_ancestors=True
            ):
                matches = matching_names(label, context.objects, context, self.catalog)
                exact_instance = _is_instance_name(label, context.objects)
                if len(matches) > 1 and not exact_instance:
                    candidates.append((name, label))
        if not candidates:
            raise ValueError("world has no ambiguous object label")
        return rng.choice(candidates)


# ----- Plan compilation -----


def _navigation_steps(location):
    """Park the arms before navigating to a location."""
    return (
        PlanStep("ParkArmsAction"),
        PlanStep("NavigateAction", location=location),
    )


def _open(container):
    return PlanStep("OpenAction", object=container)


def _close(container):
    return PlanStep("CloseAction", object=container)


def _require_name(name, allowed, field):
    if name is None:
        raise ValueError(f"{field} is required")
    if name not in allowed:
        raise ValueError(f"unknown {field} {name!r}; choose from {list(allowed)}")
    return name


def _is_openable(name, context):
    return name in context.openables


def _source_for(intent, context, catalog):
    """Return the selected source and number of possible object locations."""
    object_name = _require_name(intent.object, context.objects, "object")
    locations = context.object_locations.get(object_name, ())
    source = intent.source
    if source is None:
        if not locations:
            raise ValueError(f"source is missing for {object_name!r}")
        if len(locations) > 1:
            raise ValueError(f"source is ambiguous for {object_name!r}: {locations}")
        source = locations[0]

    _require_name(
        source,
        context.surfaces + storage_containers(context, catalog),
        "source",
    )
    if locations and source not in locations:
        raise ValueError(f"{object_name!r} is not located at source {source!r}")
    return source, len(locations)


def _destination_for(intent, context, catalog):
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
        destination = _require_name(destination, context.objects, "reference object")
        if destination == intent.object:
            raise ValueError("an object cannot be placed relative to itself")
        return destination
    raise ValueError(
        "relation must be on, inside, left_of, right_of, in_front_of, or behind"
    )


def _include_source(intent, source, location_count, context):
    """Return whether the source is serialized: explicit in the instruction,
    an openable container whose access must be shown, or one of several
    known object locations."""
    return intent.source_explicit or _is_openable(source, context) or location_count > 1


def _compile_transport(intent, context, catalog):
    object_name = _require_name(intent.object, context.objects, "object")
    destination = _destination_for(intent, context, catalog)
    source, location_count = _source_for(intent, context, catalog)
    if source == destination:
        raise ValueError("transport source and destination must be different")

    source_openable = _is_openable(source, context)
    destination_openable = _is_openable(destination, context)
    steps = []
    if source_openable:
        steps.extend((*_navigation_steps(source), _open(source)))
    if destination_openable:
        steps.extend((*_navigation_steps(destination), _open(destination)))

    step_source = None
    if _include_source(intent, source, location_count, context):
        step_source = source
    steps.append(
        PlanStep(
            "TransportAction",
            object=object_name,
            location=destination,
            relation=intent.relation,
            source=step_source,
        )
    )

    # TransportAction ends at the destination. Close it first, then return to
    # any source that was opened solely for access.
    if destination_openable:
        steps.append(_close(destination))
    if source_openable:
        steps.extend((*_navigation_steps(source), _close(source)))
    return tuple(steps)


def _compile_pickup(intent, context, catalog):
    object_name = _require_name(intent.object, context.objects, "object")
    source, location_count = _source_for(intent, context, catalog)
    source_openable = _is_openable(source, context)
    steps = list(_navigation_steps(source))
    if source_openable:
        steps.append(_open(source))

    step_source = None
    if _include_source(intent, source, location_count, context):
        step_source = source
    steps.append(
        PlanStep(
            "PickUpAction",
            object=object_name,
            source=step_source,
        )
    )
    return tuple(steps)


def _placement_navigation(intent, context, catalog):
    destination = _destination_for(intent, context, catalog)
    if intent.relation not in DIRECTIONAL_RELATIONS:
        return destination

    locations = context.object_locations.get(destination, ())
    if len(locations) != 1:
        raise ValueError(f"reference object {destination!r} needs one known location")
    return _require_name(
        locations[0],
        context.surfaces,
        "reference-object surface",
    )


def _compile_pickup_place(intent, context, catalog):
    object_name = _require_name(intent.object, context.objects, "object")
    source, location_count = _source_for(intent, context, catalog)
    if _is_openable(source, context):
        raise ValueError("pickup-and-place examples require a surface source")

    destination = _destination_for(intent, context, catalog)
    destination_openable = _is_openable(destination, context)
    placement_navigation = _placement_navigation(intent, context, catalog)
    steps = []

    # Open a destination before pickup so a one-arm robot can still access it.
    if destination_openable:
        steps.extend((*_navigation_steps(destination), _open(destination)))

    steps.extend(_navigation_steps(source))
    step_source = None
    if _include_source(intent, source, location_count, context):
        step_source = source
    steps.append(
        PlanStep(
            "PickUpAction",
            object=object_name,
            source=step_source,
        )
    )
    if placement_navigation != source:
        steps.extend(_navigation_steps(placement_navigation))
    steps.append(
        PlanStep(
            "PlaceAction",
            object=object_name,
            location=destination,
            relation=intent.relation,
        )
    )
    if destination_openable:
        steps.append(_close(destination))
    return tuple(steps)


def _container_target(intent, context):
    target = intent.destination
    if target is None:
        target = intent.object
    target = _require_name(target, context.containers, "container")
    if not _is_openable(target, context):
        raise ValueError(f"container {target!r} is not openable")
    return target


def compile_plan(intent, context, catalog):
    """
    Compile one fully grounded intent into its canonical plan.

    Args:
        intent: Grounded canonical intent to compile.
        context: World context containing all referenced entities.
        catalog: Semantic catalog providing storage capabilities.

    Returns:
        An ordered tuple of canonical planner steps.

    Raises:
        ValueError: If the intent or context cannot produce a valid plan.
    """
    context.validate()

    if intent.action == "transport":
        return _compile_transport(intent, context, catalog)
    if intent.action == "pickup":
        return _compile_pickup(intent, context, catalog)
    if intent.action == "pickup_place":
        return _compile_pickup_place(intent, context, catalog)
    if intent.action == "navigate":
        destination = _require_name(intent.destination, context.places, "destination")
        return _navigation_steps(destination)
    if intent.action in {"open", "close"}:
        target = _container_target(intent, context)
        if intent.action == "open":
            action = _open(target)
        else:
            action = _close(target)
        return (*_navigation_steps(target), action)
    raise ValueError(f"unsupported action: {intent.action!r}")
