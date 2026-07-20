from .domain import Mention, RenderedInstruction

TRANSPORT_TEMPLATES = (
    "Move the {object} {preposition} the {destination}.",
    "Put the {object} {preposition} the {destination}.",
    "Please place the {object} {preposition} the {destination}.",
    "Bring the {object} {preposition} the {destination}.",
    "Could you set the {object} {preposition} the {destination}?",
    "Set the {object} {preposition} the {destination}.",
    "Place the {object} {preposition} the {destination} for me.",
    "Would you move the {object} {preposition} the {destination}?",
)
TRANSPORT_FROM_TEMPLATES = (
    "Move the {object} from the {source} {preposition} the {destination}.",
    "Carry the {object} from the {source} {preposition} the {destination}.",
    "Please move the {object} from the {source} {preposition} the {destination}.",
    "Take the {object} from the {source} and set it {preposition} the {destination}.",
    "Please bring the {object} from the {source} and put it {preposition} the {destination}.",
    "Transfer the {object} from the {source} {preposition} the {destination}.",
    "Please take the {object} from the {source} and place the held item {preposition} the {destination}.",
    "Would you move the {object} from the {source} {preposition} the {destination}?",
)
PICKUP_TEMPLATES = (
    "Pick up the {object}.",
    "Please grab the {object}.",
    "Take the {object} and hold it.",
    "Grab the {object} for me.",
    "Could you pick up the {object}?",
    "Lift the {object}.",
    "Please take hold of the {object}.",
    "Would you grab the {object} for me?",
)
PICKUP_FROM_TEMPLATES = (
    "Pick up the {object} from the {source}.",
    "Take the {object} from the {source} and hold it.",
    "Grab the {object} from the {source}.",
    "Please get the {object} from the {source}.",
    "Could you pick up the {object} from the {source}?",
    "Lift the {object} from the {source}.",
    "Please collect the {object} from the {source}.",
    "Would you grab the {object} from the {source}?",
)
PICKUP_PLACE_TEMPLATES = (
    "Pick up the {object} from the {source}, then place the held item {preposition} the {destination}.",
    "Take the {object} from the {source} and put the held item {preposition} the {destination}.",
    "Grab the {object} from the {source}, then set it {preposition} the {destination}.",
    "Please pick up the {object} from the {source} and place it {preposition} the {destination}.",
    "Could you take the {object} from the {source} and put it {preposition} the {destination}?",
    "Lift the {object} from the {source}, then place the held item {preposition} the {destination}.",
    "Collect the {object} from the {source} and set the held item {preposition} the {destination}.",
    "Would you pick up the {object} from the {source} and place the held item {preposition} the {destination}?",
)
NAVIGATE_TEMPLATES = (
    "Go to the {destination}.",
    "Navigate to the {destination}.",
    "Please drive to the {destination}.",
    "Move over to the {destination}.",
    "Could you go to the {destination}?",
    "Head to the {destination}.",
    "Please make your way to the {destination}.",
    "Would you navigate to the {destination}?",
)
OPEN_TEMPLATES = (
    "Open the {object}.",
    "Please open the {object}.",
    "Could you open the {object}?",
    "Open up the {object} for me.",
    "Please open the {object} for me.",
    "Open the {object}, please.",
    "Would you open the {object}?",
    "I need you to open the {object}.",
)
CLOSE_TEMPLATES = (
    "Close the {object}.",
    "Please shut the {object}.",
    "Could you close the {object}?",
    "Shut the {object} for me.",
    "Please close the {object} for me.",
    "Close the {object}, please.",
    "Would you shut the {object}?",
    "I need you to close the {object}.",
)
MISSING_OBJECT_TEMPLATES = (
    "Put this {preposition} the {destination}.",
    "Move it {preposition} the {destination}.",
    "Please place this {preposition} the {destination}.",
    "Could you put it {preposition} the {destination}?",
    "Take this and set it {preposition} the {destination}.",
    "Would you move this {preposition} the {destination}?",
    "Please put that {preposition} the {destination}.",
    "Could you place the item {preposition} the {destination}?",
)
_RELATION_PHRASES = {
    "inside": "into",
    "on": "onto",
    "left_of": "to the left of",
    "right_of": "to the right of",
    "in_front_of": "in front of",
    "behind": "behind",
}


def _shown(value, fallback):
    return value if value is not None else fallback


def _fill_template(templates, rng, **values):
    """Choose and fill one template, returning its stable list position."""
    template = rng.choice(templates)
    return template.format(**values), templates.index(template)


def _join_clauses(clauses):
    """Join direct multi-task commands into one coordinated instruction."""
    parts = []
    for index, clause in enumerate(clauses):
        text = clause.text.rstrip(".?!")
        if index:
            for prefix in ("Please ", "Could you ", "Would you ", "I need you to "):
                if text.startswith(prefix):
                    text = text.removeprefix(prefix)
                    break
            text = text[:1].lower() + text[1:]
        parts.append(text)
    return ", and then ".join(parts) + "."


def _render_single(intent, references, context, rng, family):
    object_reference = references.get("object")
    source_reference = references.get("source")
    destination_reference = references.get("destination")

    if family == "clarify_missing_object":
        text, template_index = _fill_template(
            MISSING_OBJECT_TEMPLATES,
            rng,
            destination=_shown(destination_reference, "somewhere"),
            preposition=_RELATION_PHRASES[intent.relation],
        )
        template_id = f"transport:missing:{template_index}"
    elif intent.action == "transport":
        templates = (
            TRANSPORT_FROM_TEMPLATES if source_reference else TRANSPORT_TEMPLATES
        )
        text, template_index = _fill_template(
            templates,
            rng,
            object=_shown(object_reference, "object"),
            source=_shown(source_reference, "somewhere"),
            destination=_shown(destination_reference, "somewhere"),
            preposition=_RELATION_PHRASES[intent.relation],
        )
        template_id = f"transport:{template_index}:{bool(source_reference)}"
    elif intent.action == "pickup":
        templates = PICKUP_FROM_TEMPLATES if source_reference else PICKUP_TEMPLATES
        text, template_index = _fill_template(
            templates,
            rng,
            object=_shown(object_reference, "object"),
            source=_shown(source_reference, "somewhere"),
        )
        template_id = f"pickup:{template_index}:{bool(source_reference)}"
    elif intent.action == "pickup_place":
        text, template_index = _fill_template(
            PICKUP_PLACE_TEMPLATES,
            rng,
            object=_shown(object_reference, "object"),
            source=_shown(source_reference, "somewhere"),
            destination=_shown(destination_reference, "somewhere"),
            preposition=_RELATION_PHRASES[intent.relation],
        )
        template_id = f"pickup_place:{template_index}"
    elif intent.action == "navigate":
        text, template_index = _fill_template(
            NAVIGATE_TEMPLATES,
            rng,
            destination=_shown(destination_reference, "somewhere"),
        )
        template_id = f"navigate:{template_index}"
    elif intent.action in {"open", "close"}:
        templates = OPEN_TEMPLATES if intent.action == "open" else CLOSE_TEMPLATES
        text, template_index = _fill_template(
            templates,
            rng,
            object=_shown(object_reference, "container"),
        )
        template_id = f"{intent.action}:{template_index}"
    else:
        raise ValueError(f"unsupported action: {intent.action!r}")

    mentions = []
    if intent.action in {"transport", "pickup", "pickup_place", "open", "close"}:
        object_role = "container" if intent.action in {"open", "close"} else "object"
        mentions.append(Mention("object", object_reference, object_role))
    if source_reference is not None or intent.source_explicit:
        mentions.append(Mention("source", source_reference, "place"))
    if intent.action in {"transport", "pickup_place", "navigate"}:
        destination_role = context.role_of(intent.destination or "") or "place"
        mentions.append(Mention("destination", destination_reference, destination_role))
    return RenderedInstruction(text, tuple(mentions), template_id)


def render_instruction(scenario, rng):
    """
    Render a single task or direct multi-task instruction.

    Args:
        scenario: Scenario whose intents and references should be rendered.
        rng: Random generator used to select language templates.

    Returns:
        The rendered instruction with its grounding mentions.

    Raises:
        ValueError: If the scenario contains an unsupported action.
    """
    clauses = []
    for intent, references in zip(scenario.intents(), scenario.reference_sets()):
        clause = _render_single(
            intent,
            references,
            scenario.context,
            rng,
            scenario.family,
        )
        clauses.append(clause)
    clauses = tuple(clauses)
    if len(clauses) == 1:
        return clauses[0]

    mentions = [mention for clause in clauses for mention in clause.mentions]
    text = _join_clauses(clauses)
    template_id = "multi:" + ":".join(clause.template_id for clause in clauses)
    return RenderedInstruction(text, tuple(mentions), template_id, clauses)
