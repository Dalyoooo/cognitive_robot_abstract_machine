import re
from dataclasses import replace

from .domain import (
    IGNORED_REFERENCE_WORDS,
    RECOGNIZED_POSITIONS,
    Resolution,
    unique,
)

_REQUIRED_SLOTS = {
    "transport": ("object", "destination"),
    "pickup": ("object",),
    "pickup_place": ("object", "destination"),
    "navigate": ("destination",),
    "open": ("destination",),
    "close": ("destination",),
}


def _same_text(left, right):
    return left.strip().casefold() == right.strip().casefold()


def _normalise_reference(text):
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(text))
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _description_parts(name, type_name, catalog):
    label = catalog.natural_label(type_name)
    label_words = _normalise_reference(label).split()
    words = _normalise_reference(name).split()

    if "".join(words) == "".join(label_words):
        return label, (), (), None

    trailing_number = None
    match = re.search(r"(?:_|-)(\d+)$", name)
    if match:
        trailing_number = match.group(1)

    unused = list(words)
    for word in label_words:
        if word in unused:
            unused.remove(word)

    positions = tuple(word for word in unused if word in RECOGNIZED_POSITIONS)
    qualifiers = tuple(
        word
        for word in unused
        if word not in RECOGNIZED_POSITIONS
        and word not in IGNORED_REFERENCE_WORDS
        and word != trailing_number
    )
    return label, positions, qualifiers, trailing_number


def reference_phrases(name, context, catalog):
    type_name = context.types.get(name)
    if type_name is None:
        return ()

    labels = catalog.reference_labels(type_name)
    label, positions, qualifiers, number = _description_parts(name, type_name, catalog)
    label_words = tuple(_normalise_reference(label).split())
    core = " ".join((*positions, *label_words))

    descriptions = []
    if positions:
        descriptions.append(core)
    if qualifiers:
        qualifier = " ".join(qualifiers)
        descriptions.append(f"{label} in the {qualifier}")
        descriptions.append(f"{core} in the {qualifier}")
    if number:
        numbered = f"{core} number {number}"
        descriptions.append(numbered)
        if qualifiers:
            descriptions.append(f"{numbered} in the {' '.join(qualifiers)}")

    descriptions.append(_normalise_reference(name))
    return unique(
        phrase.strip()
        for phrase in (*labels, *descriptions)
        if phrase and phrase.strip()
    )


def matching_names(text, names, context, catalog):
    wanted = _normalise_reference(text)
    matches = []
    for name in names:
        type_name = context.types.get(name)
        if type_name is None:
            continue
        phrases = reference_phrases(name, context, catalog)
        label_matches = catalog.matches(type_name, text)
        phrase_matches = False
        if not label_matches:
            for phrase in phrases:
                if _normalise_reference(phrase) == wanted:
                    phrase_matches = True
                    break
        if label_matches or phrase_matches:
            matches.append(name)
    return unique(matches)


def natural_reference(name, names, context, catalog):
    names = tuple(names)
    if name not in names:
        raise ValueError(f"{name!r} is not among the allowed reference names")
    for phrase in reference_phrases(name, context, catalog):
        if matching_names(phrase, names, context, catalog) == (name,):
            return phrase
    raise ValueError(f"no unique natural reference for {name!r}")


def candidates_for_mention(mention, context, catalog, *, allowed_names=None):
    if mention.text is None or not mention.text.strip():
        return ()

    role_names = context.names_for_role(mention.role)
    if allowed_names is not None:
        allowed = set(allowed_names)
        role_names = tuple(name for name in role_names if name in allowed)

    exact = tuple(name for name in role_names if _same_text(name, mention.text))
    if exact:
        return exact

    return matching_names(mention.text, role_names, context, catalog)


def _mentions_by_slot(mentions):
    by_slot = {}
    duplicate_slots = set()
    for mention in mentions:
        if mention.slot in by_slot:
            duplicate_slots.add(mention.slot)
        by_slot[mention.slot] = mention
    if duplicate_slots:
        raise ValueError(f"multiple mentions for slots: {sorted(duplicate_slots)}")
    return by_slot


def _resolve_mention(intent, mention, context, catalog, *, allowed_names=None):
    candidates = candidates_for_mention(
        mention,
        context,
        catalog,
        allowed_names=allowed_names,
    )
    if not candidates:
        return Resolution("missing", intent, slot=mention.slot)
    if len(candidates) > 1:
        return Resolution("ambiguous", intent, slot=mention.slot, candidates=candidates)
    return Resolution("resolved", intent.with_slot(mention.slot, candidates[0]))


def _infer_source(intent, context):
    if (
        intent.action not in {"pickup", "pickup_place", "transport"}
        or intent.object is None
    ):
        return Resolution("resolved", intent)

    locations = context.object_locations.get(intent.object, ())
    if intent.source_explicit and intent.source is not None:
        if intent.source not in locations:
            return Resolution("missing", intent, slot="source")
        return Resolution("resolved", intent)
    if not locations:
        return Resolution("missing", intent, slot="source")
    if len(locations) > 1:
        return Resolution("ambiguous", intent, slot="source", candidates=locations)

    return Resolution(
        "resolved",
        replace(intent, source=locations[0], source_explicit=False),
    )


def _missing_slot_resolution(intent):
    for slot in _REQUIRED_SLOTS[intent.action]:
        if slot == "destination" and intent.action in {"open", "close"}:
            if intent.destination is None and intent.object is None:
                return Resolution("missing", intent, slot=slot)
            continue
        if getattr(intent, slot) is None:
            return Resolution("missing", intent, slot=slot)
    return None


def resolve_mentions(intent, mentions, context, catalog):
    current = intent
    by_slot = _mentions_by_slot(mentions)

    action_mention = by_slot.get("action")
    if action_mention is not None and (
        action_mention.text is None or not action_mention.text.strip()
    ):
        return Resolution("missing", current, slot="action")

    for slot in ("object", "destination"):
        mention = by_slot.get(slot)
        if mention is None:
            continue
        result = _resolve_mention(current, mention, context, catalog)
        if result.status != "resolved":
            return result
        current = result.intent

    source_mention = by_slot.get("source")
    if source_mention is not None:
        known_locations = ()
        if current.object is not None:
            known_locations = context.object_locations.get(current.object, ())
        result = _resolve_mention(
            current,
            source_mention,
            context,
            catalog,
            allowed_names=known_locations,
        )
        if result.status != "resolved":
            return result
        current = result.intent

    missing_resolution = _missing_slot_resolution(current)
    if missing_resolution is not None:
        return missing_resolution

    if source_mention is None:
        return _infer_source(current, context)
    return Resolution("resolved", current)


def _join_candidates(candidates):
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return f"{candidates[0]} or {candidates[1]}"
    return f"{', '.join(candidates[:-1])}, or {candidates[-1]}"


def _format_candidates(resolution, context, catalog):
    if resolution.slot == "action":
        return _join_candidates(resolution.candidates)
    spoken = tuple(
        natural_reference(name, resolution.candidates, context, catalog)
        for name in resolution.candidates
    )
    return _join_candidates(spoken)


def clarification_question(resolution, context, catalog):
    if resolution.status == "resolved" or resolution.slot is None:
        raise ValueError("a resolved command does not need clarification")

    if resolution.status == "ambiguous":
        label = {
            "action": "action",
            "object": "object",
            "source": "source",
            "destination": "destination",
        }[resolution.slot]
        if resolution.slot == "object" and resolution.intent.action in {
            "open",
            "close",
        }:
            label = "container"
        candidates = _format_candidates(resolution, context, catalog)
        return f"Which {label} do you mean: {candidates}?"

    return {
        "action": "What should the robot do?",
        "object": "Which object should the robot use?",
        "source": "Where is the object currently located?",
        "destination": "Where should the robot put or go to?",
    }[resolution.slot]
