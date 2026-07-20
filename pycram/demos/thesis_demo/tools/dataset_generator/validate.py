import json
import re

from .domain import DIRECTIONAL_RELATIONS
from .world import storage_containers

STEP_KEYS = {"action", "object", "location", "relation", "source"}
ACTIONS = {
    "ParkArmsAction",
    "NavigateAction",
    "PickUpAction",
    "PlaceAction",
    "TransportAction",
    "OpenAction",
    "CloseAction",
}


def _required(value, field):
    if value is None:
        raise ValueError(f"intent has no {field}")
    return value


def _source(intent, context):
    object_name = _required(intent.object, "object")
    locations = context.object_locations.get(object_name, ())
    if intent.source is not None:
        return intent.source
    if len(locations) != 1:
        raise ValueError(f"intent needs one source for {object_name!r}")
    return locations[0]


def _openable(name, context):
    return name in context.openables


def _goal_step(steps, action):
    matches = [step for step in steps if step.action == action]
    if len(matches) != 1:
        raise ValueError(f"plan must contain exactly one {action}")
    return matches[0]


def _expected_source(intent, context):
    source = _source(intent, context)
    locations = context.object_locations.get(intent.object, ())
    include_source = (
        intent.source_explicit or len(locations) > 1 or _openable(source, context)
    )
    if include_source:
        return source
    return None


def _validate_transport_goal(intent, steps, context):
    step = _goal_step(steps, "TransportAction")
    actual = (step.object, step.location, step.relation, step.source)
    expected = (
        intent.object,
        intent.destination,
        intent.relation,
        _expected_source(intent, context),
    )
    if actual != expected:
        raise ValueError(f"TransportAction {actual!r} does not match the intent")


def _validate_pickup_goal(intent, steps, context):
    step = _goal_step(steps, "PickUpAction")
    if step.object != intent.object:
        raise ValueError("PickUpAction uses the wrong object")
    if step.source != _expected_source(intent, context):
        raise ValueError("PickUpAction uses the wrong source")


def _validate_pickup_place_goal(intent, steps, context):
    pickup = _goal_step(steps, "PickUpAction")
    place = _goal_step(steps, "PlaceAction")
    if pickup.object != intent.object or place.object != intent.object:
        raise ValueError("pickup/place uses the wrong object")
    if pickup.source != _expected_source(intent, context):
        raise ValueError("PickUpAction uses the wrong source")
    if (place.location, place.relation) != (intent.destination, intent.relation):
        raise ValueError("PlaceAction does not match the intent")


def _validate_goal(intent, steps, context):
    if intent.action == "transport":
        _validate_transport_goal(intent, steps, context)
        return

    if intent.action == "pickup":
        _validate_pickup_goal(intent, steps, context)
        return

    if intent.action == "pickup_place":
        _validate_pickup_place_goal(intent, steps, context)
        return

    if intent.action == "navigate":
        step = _goal_step(steps, "NavigateAction")
        if step.location != intent.destination:
            raise ValueError("NavigateAction uses the wrong destination")
        return

    if intent.action in {"open", "close"}:
        if intent.action == "open":
            action = "OpenAction"
        else:
            action = "CloseAction"
        target = intent.destination or intent.object
        if _goal_step(steps, action).object != target:
            raise ValueError(f"{action} uses the wrong container")
        return

    raise ValueError(f"unsupported intent action: {intent.action!r}")


def _validate_relations(intent, context, catalog):
    if intent.action not in {"transport", "pickup_place"}:
        return
    destination = _required(intent.destination, "destination")
    if intent.relation == "on" and destination not in context.surfaces:
        raise ValueError("relation 'on' requires a surface")
    if intent.relation == "inside" and destination not in storage_containers(
        context, catalog
    ):
        raise ValueError("relation 'inside' requires storage")
    if intent.relation in DIRECTIONAL_RELATIONS:
        if destination not in context.objects or destination == intent.object:
            raise ValueError("directional relation requires another object")


def _navigation_target(step, context):
    if step.location not in context.places:
        raise ValueError(f"unknown navigation target {step.location!r}")
    return step.location


def _validate_navigation_sequence(steps):
    """Require every arm-parking step and navigation step to form a pair."""
    for index, step in enumerate(steps):
        if step.action == "ParkArmsAction":
            if index + 1 >= len(steps) or steps[index + 1].action != "NavigateAction":
                raise ValueError("ParkArmsAction must be followed by NavigateAction")
        elif step.action == "NavigateAction":
            if index == 0 or steps[index - 1].action != "ParkArmsAction":
                raise ValueError("NavigateAction needs a preceding ParkArmsAction")


def _validate_open_step(step, current_location, opened, context):
    if current_location != step.object:
        raise ValueError("OpenAction needs matching navigation")
    if not _openable(step.object, context):
        raise ValueError(f"non-openable container {step.object!r}")
    opened.add(step.object)


def _validate_close_step(step, current_location, opened, intent):
    if current_location != step.object:
        raise ValueError("CloseAction needs matching navigation")
    if intent.action != "close" and step.object not in opened:
        raise ValueError("CloseAction needs a preceding OpenAction")
    opened.discard(step.object)


def _validate_pickup_step(step, held):
    if held:
        raise ValueError("cannot pick up a second object")
    held.add(step.object)


def _validate_place_step(step, held, opened, context):
    if step.object not in held:
        raise ValueError("PlaceAction needs a preceding PickUpAction")
    if (
        step.relation == "inside"
        and _openable(step.location, context)
        and step.location not in opened
    ):
        raise ValueError("PlaceAction accesses an unopened container")
    held.remove(step.object)


def _validate_transport_step(step, opened, context):
    for container in (step.source, step.location):
        if _openable(container, context) and container not in opened:
            raise ValueError(f"TransportAction accesses unopened {container!r}")


def _expected_open_containers(intent, context):
    if intent.action == "open":
        return {intent.destination or intent.object}
    if intent.action == "pickup":
        source = _source(intent, context)
        if _openable(source, context):
            return {source}
    return set()


def _validate_access(intent, steps, context):
    """Follow the plan and check navigation, open containers, and held objects."""
    current_location = None
    opened = set()
    held = set()

    for step in steps:
        if step.action == "NavigateAction":
            current_location = _navigation_target(step, context)
        elif step.action == "OpenAction":
            _validate_open_step(step, current_location, opened, context)
        elif step.action == "CloseAction":
            _validate_close_step(step, current_location, opened, intent)
        elif step.action == "PickUpAction":
            _validate_pickup_step(step, held)
        elif step.action == "PlaceAction":
            _validate_place_step(step, held, opened, context)
        elif step.action == "TransportAction":
            _validate_transport_step(step, opened, context)

    expected_open = _expected_open_containers(intent, context)
    if opened != expected_open:
        raise ValueError(f"unexpected open containers: {sorted(opened)}")


def validate_plan(intent, steps, context, catalog):
    """
    Check the intended goal and semDT container-access invariants.

    Args:
        intent: Grounded intent that the plan must satisfy.
        steps: Canonical plan steps to validate.
        context: World context containing referenced entities.
        catalog: Semantic catalog providing storage capabilities.

    Raises:
        ValueError: If the context, goal, sequence, or access pattern is invalid.
    """
    context.validate()
    _validate_relations(intent, context, catalog)
    _validate_goal(intent, steps, context)
    _validate_navigation_sequence(steps)
    _validate_access(intent, steps, context)


def _validate_payload(payload):
    if set(payload) == {"clarification"}:
        if (
            not isinstance(payload["clarification"], str)
            or not payload["clarification"].strip()
        ):
            raise ValueError("clarification must be non-empty text")
        return
    if set(payload) != {"plan"} or not isinstance(payload["plan"], list):
        raise ValueError("assistant JSON must contain one plan or clarification")
    for step in payload["plan"]:
        if not isinstance(step, dict) or set(step) != STEP_KEYS:
            raise ValueError("every plan step needs the exact five-field schema")
        if step["action"] not in ACTIONS:
            raise ValueError(f"unknown action {step['action']!r}")


def _instruction_text(content):
    match = re.search(r"<user_instruction>(.*?)</user_instruction>", content, re.DOTALL)
    if match is None:
        raise ValueError("user message has no <user_instruction>")
    return match.group(1)


def _reject_raw_ids(text, context):
    for name in context.objects + context.places:
        if "_" not in name and not any(character.isdigit() for character in name):
            continue
        pattern = rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])"
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError(f"user-facing text contains raw instance ID {name!r}")


def validate_messages(example):
    """
    Validate role alternation, JSON shape, and user-facing text.

    Args:
        example: Raw example containing the conversation and world context.

    Raises:
        ValueError: If message order, payload shape, or visible text is invalid.
    """
    if not example.messages or example.messages[0].get("role") != "system":
        raise ValueError("conversation must start with a system message")

    expected = "user"
    assistant_count = 0
    for message in example.messages[1:]:
        role = message.get("role")
        if role != expected:
            raise ValueError(f"expected {expected!r} message, got {role!r}")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        if role == "user":
            _reject_raw_ids(_instruction_text(content), example.context)
            expected = "assistant"
        else:
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as error:
                raise ValueError("assistant content is not JSON") from error
            _validate_payload(payload)
            if "clarification" in payload:
                _reject_raw_ids(payload["clarification"], example.context)
            assistant_count += 1
            expected = "user"

    if expected != "user" or assistant_count == 0:
        raise ValueError("conversation must end with an assistant response")


def validate_unique_examples(examples):
    """
    Reject duplicate conversations and scenario IDs.

    Args:
        examples: Raw examples to compare.

    Raises:
        ValueError: If a scenario ID or serialized conversation is duplicated.
    """
    scenario_ids = set()
    conversations = set()
    for example in examples:
        if example.scenario_id in scenario_ids:
            raise ValueError(f"duplicate scenario ID: {example.scenario_id}")
        scenario_ids.add(example.scenario_id)
        key = json.dumps(example.messages, sort_keys=True)
        if key in conversations:
            raise ValueError(f"duplicate conversation for {example.scenario_id}")
        conversations.add(key)
