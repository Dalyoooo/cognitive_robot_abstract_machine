import json
import re

from .domain import DIRECTIONAL_RELATIONS
from .policy import check_destination, include_source, resolve_source

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


def _goal_step(steps, action):
    matches = [step for step in steps if step.action == action]
    if len(matches) != 1:
        raise ValueError(f"plan must contain exactly one {action}")
    return matches[0]


def _expected_source(intent, context):
    _required(intent.object, "object")
    source, location_count = resolve_source(intent, context)
    if include_source(intent, source, location_count, context):
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
    _required(intent.destination, "destination")
    check_destination(intent, context, catalog)


def _navigation_target(step, context):
    if step.location not in context.places:
        raise ValueError(f"unknown navigation target {step.location!r}")
    return step.location


def _validate_navigation_sequence(steps):
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
    if not context.is_openable(step.object):
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
        and context.is_openable(step.location)
        and step.location not in opened
    ):
        raise ValueError("PlaceAction accesses an unopened container")
    held.remove(step.object)


def _validate_transport_step(step, opened, context):
    for container in (step.source, step.location):
        if context.is_openable(container) and container not in opened:
            raise ValueError(f"TransportAction accesses unopened {container!r}")


def _expected_open_containers(intent, context):
    if intent.action == "open":
        return {intent.destination or intent.object}
    if intent.action == "pickup":
        source, _location_count = resolve_source(intent, context)
        if context.is_openable(source):
            return {source}
    return set()


def _validate_access(intent, steps, context):
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
