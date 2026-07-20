import re
from dataclasses import replace

from .domain import Resolution, unique

_POSITIONS = {
    "back",
    "bottom",
    "center",
    "front",
    "left",
    "lower",
    "middle",
    "right",
    "top",
    "upper",
}
_IGNORED_WORDS = {"main"}
_REQUIRED_SLOTS = {
    "transport": ("object", "destination"),
    "pickup": ("object",),
    "pickup_place": ("object", "destination"),
    "navigate": ("destination",),
    "open": ("destination",),
    "close": ("destination",),
}


def _same_text(left, right):
    """Compare an instance reference without changing the instance ID."""
    return left.strip().casefold() == right.strip().casefold()


def _normalise_reference(text):
    """Normalise a spoken phrase without exposing an instance identifier."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(text))
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _description_parts(name, type_name, catalog):
    """Split an instance ID into a natural type, position, and qualifier."""
    label = catalog.natural_label(type_name)
    label_words = _normalise_reference(label).split()
    words = _normalise_reference(name).split()

    # Compressed object IDs such as ``liquidcap`` already have a better semDT
    # label (``liquid cap``), so they need no instance-derived qualifier.
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

    positions = tuple(word for word in unused if word in _POSITIONS)
    qualifiers = tuple(
        word
        for word in unused
        if word not in _POSITIONS
        and word not in _IGNORED_WORDS
        and word != trailing_number
    )
    return label, positions, qualifiers, trailing_number


def reference_phrases(name, context, catalog):
    """
    Return semDT-grounded spoken phrases for one world instance.

    Short phrases come first. The final phrase contains every useful instance
    descriptor, but uses ordinary words rather than the raw identifier.

    Args:
        name: Canonical instance name to describe.
        context: World context containing the instance and its type.
        catalog: Semantic catalog providing natural labels and aliases.

    Returns:
        An ordered tuple of natural reference phrases, or an empty tuple when
        the context does not provide the instance type.
    """
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

    # This readable fallback is unique for the composed worlds even when a
    # short type or position phrase is shared by several instances.
    descriptions.append(_normalise_reference(name))
    return unique(
        phrase.strip()
        for phrase in (*labels, *descriptions)
        if phrase and phrase.strip()
    )


def matching_names(text, names, context, catalog):
    """
    Return instance IDs described by one natural reference.

    Args:
        text: Natural reference to match.
        names: Candidate canonical instance names.
        context: World context containing candidate types.
        catalog: Semantic catalog used to compare labels.

    Returns:
        An ordered tuple of matching canonical instance names.
    """
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
    """
    Return the shortest spoken phrase that uniquely identifies an instance.

    Raises when no natural phrase is unique instead of leaking a raw instance
    ID into training text.

    Args:
        name: Canonical instance name to describe.
        names: Candidate names from which the instance must be distinguished.
        context: World context containing candidate types.
        catalog: Semantic catalog providing labels and aliases.

    Returns:
        The shortest natural phrase that uniquely matches the instance.

    Raises:
        ValueError: If the name is disallowed or no phrase is unique.
    """
    names = tuple(names)
    if name not in names:
        raise ValueError(f"{name!r} is not among the allowed reference names")
    for phrase in reference_phrases(name, context, catalog):
        if matching_names(phrase, names, context, catalog) == (name,):
            return phrase
    raise ValueError(f"no unique natural reference for {name!r}")


def candidates_for_mention(mention, context, catalog, *, allowed_names=None):
    """
    Return context instance IDs matched by one mention.

    An exact instance ID always wins. Only when there is no exact match do we
    use semDT labels and spoken instance descriptions. ``allowed_names`` is useful for source mentions, which must
    name one of the object's known locations rather than an arbitrary place in
    the world.

    Args:
        mention: Rendered entity mention to ground.
        context: World context containing role-specific candidates.
        catalog: Semantic catalog used to resolve natural labels.
        allowed_names: Optional subset of canonical names that may match.

    Returns:
        An ordered tuple of canonical names matched by the mention.
    """
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
    """Index mentions and report duplicate slots before grounding them."""
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
    """Resolve one mention and update its matching intent slot."""
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
    """Infer an unnamed source, refusing to guess between known locations."""
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

    # The source is known to the plan builder, but it was not explicitly said by the
    # user.  This distinction controls whether the plan needs a source field.
    return Resolution(
        "resolved",
        replace(intent, source=locations[0], source_explicit=False),
    )


def _missing_slot_resolution(intent):
    """Return a missing-slot resolution for the first empty required slot."""
    for slot in _REQUIRED_SLOTS[intent.action]:
        # Open/close scenarios may store their container in ``object`` for
        # convenience. The plan builder accepts both forms.
        if slot == "destination" and intent.action in {"open", "close"}:
            if intent.destination is None and intent.object is None:
                return Resolution("missing", intent, slot=slot)
            continue
        if getattr(intent, slot) is None:
            return Resolution("missing", intent, slot=slot)
    return None


def resolve_mentions(intent, mentions, context, catalog):
    """
    Ground all recorded mentions and return the first unresolved slot.

    Object and destination references are handled before the source so the
    source can be restricted to the selected object's ``object_locations``.
    Generated examples should contain at most one mention per slot.

    Args:
        intent: Canonical intent whose slots should be grounded.
        mentions: Rendered mentions associated with the intent.
        context: World context used to resolve canonical names.
        catalog: Semantic catalog used to match natural labels.

    Returns:
        The resolved, ambiguous, or missing grounding result.

    Raises:
        ValueError: If more than one mention targets the same slot.
    """
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
    """
    Build a deterministic question that names every ambiguity candidate.

    Args:
        resolution: Ambiguous or missing grounding result to clarify.
        context: World context containing the candidate entities.
        catalog: Semantic catalog used to create natural references.

    Returns:
        A single natural-language clarification question.

    Raises:
        ValueError: If the resolution is already complete.
    """
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
