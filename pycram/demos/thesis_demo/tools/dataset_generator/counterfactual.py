from collections import Counter
from dataclasses import replace


def _object_mapping(context):
    used_names = set(context.places)
    mapping = {}
    next_number = 1

    for object_name in context.objects:
        new_name = f"entity_{next_number:02d}"
        while new_name in used_names:
            next_number += 1
            new_name = f"entity_{next_number:02d}"
        mapping[object_name] = new_name
        used_names.add(new_name)
        next_number += 1

    return mapping


def _rename_context_objects(context, mapping):
    renamed_types = {}
    for name, type_name in context.types.items():
        renamed_name = mapping.get(name, name)
        renamed_types[renamed_name] = type_name

    renamed_locations = {}
    for name, locations in context.object_locations.items():
        renamed_locations[mapping[name]] = locations

    return replace(
        context,
        objects=tuple(mapping[name] for name in context.objects),
        object_locations=renamed_locations,
        types=renamed_types,
    )


def _rename_intent_objects(intent, mapping):
    return replace(
        intent,
        object=mapping.get(intent.object, intent.object),
        destination=mapping.get(intent.destination, intent.destination),
    )


def make_counterfactual_scenario(scenario):
    if scenario.family.startswith("clarify_"):
        return None

    referenced_objects = set()
    for intent in scenario.intents():
        for name in (intent.object, intent.destination):
            if name in scenario.context.objects:
                referenced_objects.add(name)
    if not referenced_objects:
        return None

    type_counts = Counter(
        scenario.context.types[name] for name in scenario.context.objects
    )
    for name in referenced_objects:
        type_name = scenario.context.types[name]
        if type_counts[type_name] > 1:
            return None

    mapping = _object_mapping(scenario.context)
    renamed_intents = tuple(
        _rename_intent_objects(intent, mapping) for intent in scenario.intents()
    )
    first_intent, *extra_intents = renamed_intents
    return replace(
        scenario,
        id=f"{scenario.id}-counterfactual",
        world_id=f"{scenario.world_id}-counterfactual",
        context=_rename_context_objects(scenario.context, mapping),
        intent=first_intent,
        extra_intents=tuple(extra_intents),
    )
