"""Validation and arithmetic for a scene interpretation.

An interpretation names which utterance is the instruction, which others change
what the robot should do, and which are ignored. :func:`parse_interpretation` accepts
one only if its shape is exact, every index exists, every object type appears in
the world context, and every utterance carries one role, the instruction aside:
an utterance that both orders and corrects itself is instruction and context at
once.
:func:`aggregate` sums the changes in how many of something is needed, and
:func:`fuse` writes the instruction handed to the planner.

Shape alone does not make a change believable, so
:func:`check_deltas_are_supported` reads each claimed change back against the
words it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum

from typing_extensions import Protocol

# %% the utterance contract the dialogue stage relies on


class SpokenUtterance(Protocol):
    """What this module needs to know about one heard utterance."""

    text: str
    """The recognised words."""

    speaker_id: int | None
    """Which voice said it, or None when voices were not told apart."""


# %% vocabulary of the interpretation payload


class InterpretationKey(StrEnum):
    """Top-level keys of an interpretation."""

    INSTRUCTION = "instruction"
    QUANTITY = "quantity"
    CONTEXT = "context"
    IGNORE = "ignore"


class ContextItemKey(StrEnum):
    """Keys of one entry under :attr:`InterpretationKey.CONTEXT`."""

    UTTERANCE = "utterance"
    EFFECT = "effect"
    OBJECT = "object"
    DELTA = "delta"


class QuantityKey(StrEnum):
    """Keys of one entry under :attr:`InterpretationKey.QUANTITY`."""

    OBJECT = "object"
    COUNT = "count"


class UtteranceRole(StrEnum):
    """The part one utterance plays in the scene."""

    INSTRUCTION = "instruction"
    CONTEXT = "context"
    IGNORED = "ignored"


def _key_names(keys):
    """Render a key set the way it appears in the model's answer.

    Formatting the members directly spells them as ``<ClassName.MEMBER: 'x'>``,
    and anything in angle brackets disappears when the reason is shown as HTML.
    """
    return sorted(str(key) for key in keys)


INTERPRETATION_KEYS = frozenset(InterpretationKey)
CONTEXT_ITEM_KEYS = frozenset(ContextItemKey)
QUANTITY_KEYS = frozenset(QuantityKey)

MAX_ABS_DELTA = 20
"""Largest change in a count a single utterance may claim."""

MAX_COUNT = 100
"""Largest number of one object type an instruction may ask for."""


# %% parsed interpretation


@dataclass(frozen=True)
class QuantityRequest:
    """How many of one object type the instruction itself asks for."""

    object_type: str
    """Type name as spelled in the world context."""

    count: int
    """Number the instruction asks for."""


@dataclass(frozen=True)
class ContextItem:
    """A background utterance kept because it changes the instruction's outcome."""

    utterance: int
    """Index of the utterance in the scene."""

    effect: str
    """How the task changes, as one clause the planner can act on."""

    object_type: str | None = None
    """Type whose required number changes, or None for any other change."""

    delta: int | None = None
    """Change in that number, negative for fewer, or None."""

    @property
    def is_quantitative(self) -> bool:
        """Whether this item changes *how many* of something is needed."""
        return self.object_type is not None and self.delta is not None


@dataclass(frozen=True)
class Interpretation:
    """How a recorded scene splits into instruction, relevant context and noise.

    Every utterance index appears once across :attr:`instruction`,
    :attr:`context` and :attr:`ignore`, except the instruction's own index,
    which may also appear in :attr:`context` when that utterance corrects
    itself.
    """

    instruction: int | None
    """Index of the utterance that instructs the robot, or None when absent."""

    context: tuple[ContextItem, ...]
    """Background utterances that change the instruction's outcome."""

    ignore: tuple[int, ...]
    """Indices of utterances that do not affect the task."""

    quantity: tuple[QuantityRequest, ...] = ()
    """Numbers the instruction itself asks for, empty when it names none."""

    def role_of(self, utterance: int) -> UtteranceRole:
        """Return the part the utterance at this index plays."""
        if self.instruction == utterance:
            return UtteranceRole.INSTRUCTION
        if any(item.utterance == utterance for item in self.context):
            return UtteranceRole.CONTEXT
        return UtteranceRole.IGNORED

    def effect_of(self, utterance: int) -> str:
        """Return how the utterance at this index changes the task, if at all."""
        for item in self.context:
            if item.utterance == utterance:
                return item.effect
        return ""


@dataclass(frozen=True)
class Aggregate:
    """The summed effect of every quantitative constraint, per object type."""

    totals: dict[str, int] = field(default_factory=dict)
    """Net change in how many of each object type is needed."""

    final: dict[str, int] = field(default_factory=dict)
    """Resulting absolute count, for object types the instruction gave a number for."""

    assumed_distinct_speakers: bool = False
    """Whether claims were counted as separate people for want of voice labels.

    .. note:: The totals are only as reliable as that assumption.
    """

    duplicate_claims: tuple[int, ...] = ()
    """Indices whose claim repeated one already counted for the same voice."""


# %% field-level checks


def _is_index(value, utterance_count: int) -> bool:
    # bool is an int subclass; a JSON true/false is never a valid index.
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < utterance_count
    )


def _require_index(value, utterance_count: int, field_name: str) -> int:
    if not _is_index(value, utterance_count):
        raise ValueError(
            f"{field_name} must be an utterance index in 0..{utterance_count - 1}, "
            f"got {value!r}"
        )
    return value


def _require_object_type(value, known_object_types, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty object type")
    name = value.strip()
    if known_object_types is not None and name not in known_object_types:
        choices = ", ".join(sorted(known_object_types)) or "none"
        raise ValueError(f"unknown {field_name} type {name!r}; choose from: {choices}")
    return name


def _require_delta(value) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"delta must be a whole number, got {value!r}")
    if value == 0:
        raise ValueError("delta must not be 0; omit the constraint instead")
    if abs(value) > MAX_ABS_DELTA:
        raise ValueError(f"delta {value} is out of range (max {MAX_ABS_DELTA})")
    return value


def _require_count(value) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"count must be a whole number >= 0, got {value!r}")
    if value > MAX_COUNT:
        raise ValueError(f"count {value} is out of range (max {MAX_COUNT})")
    return value


# %% payload parsing


def _parse_quantity(value, known_object_types) -> list[QuantityRequest]:
    if not isinstance(value, list):
        raise ValueError(f"{InterpretationKey.QUANTITY!s} must be a JSON array")
    requests: list[QuantityRequest] = []
    for entry in value:
        if not isinstance(entry, dict) or frozenset(entry) != QUANTITY_KEYS:
            raise ValueError(
                f"each {InterpretationKey.QUANTITY!s} item needs exactly the keys "
                f"{_key_names(QUANTITY_KEYS)}"
            )
        name = _require_object_type(
            entry[QuantityKey.OBJECT], known_object_types, "quantity.object"
        )
        if any(request.object_type == name for request in requests):
            raise ValueError(f"quantity lists {name!r} more than once")
        requests.append(
            QuantityRequest(
                object_type=name, count=_require_count(entry[QuantityKey.COUNT])
            )
        )
    return requests


def _parse_context(value, utterance_count: int, known_object_types) -> list[ContextItem]:
    if not isinstance(value, list):
        raise ValueError(f"{InterpretationKey.CONTEXT!s} must be a JSON array")
    items: list[ContextItem] = []
    for entry in value:
        if not isinstance(entry, dict) or frozenset(entry) != CONTEXT_ITEM_KEYS:
            raise ValueError(
                f"each {InterpretationKey.CONTEXT!s} item needs exactly the keys "
                f"{_key_names(CONTEXT_ITEM_KEYS)}"
            )
        index = _require_index(
            entry[ContextItemKey.UTTERANCE], utterance_count, "context.utterance"
        )
        effect = entry[ContextItemKey.EFFECT]
        if not isinstance(effect, str) or not effect.strip():
            raise ValueError("context.effect must be a non-empty string")

        object_type = entry[ContextItemKey.OBJECT]
        delta = entry[ContextItemKey.DELTA]
        if (object_type is None) != (delta is None):
            raise ValueError(
                "context items need 'object' and 'delta' either both set (a "
                "change in how many) or both null (any other change)"
            )
        if object_type is not None:
            object_type = _require_object_type(
                object_type, known_object_types, "context.object"
            )
            delta = _require_delta(delta)

        items.append(
            ContextItem(
                utterance=index,
                effect=effect.strip(),
                object_type=object_type,
                delta=delta,
            )
        )
    return items


def _parse_ignore(value, utterance_count: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{InterpretationKey.IGNORE!s} must be a JSON array")
    return [
        _require_index(index, utterance_count, InterpretationKey.IGNORE)
        for index in value
    ]


def _check_partition(instruction, context_items, ignore, utterance_count: int) -> None:
    """Check that every utterance index carries a role, and only one.

    The instruction is the exception: a speaker who corrects themselves without
    pausing leaves the order and the correction in one utterance, so that index
    may also carry a context item. It can never be ignored.
    """
    if instruction is not None and instruction in ignore:
        raise ValueError(
            f"utterance {instruction} is the instruction, so it cannot also be ignored"
        )

    indices = [item.utterance for item in context_items] + list(ignore)
    seen, duplicates = set(), set()
    for index in indices:
        if index in seen:
            duplicates.add(index)
        seen.add(index)
    if duplicates:
        raise ValueError(
            f"utterance(s) {sorted(duplicates)} assigned to more than one role"
        )
    if instruction is not None:
        seen.add(instruction)
    missing = sorted(set(range(utterance_count)) - seen)
    if missing:
        raise ValueError(
            f"utterance(s) {missing} left unassigned; every index must be the "
            "instruction, context or ignored"
        )


def known_object_types(context) -> set[str] | None:
    """Return the object types the scene may talk about, or None if unrestricted."""
    if not context:
        return None
    names = set(context.get("objects", []))
    return names or None


# %% the words this world is spoken about in

SPOKEN_NAME_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
"""Where a type name written as ``CoffeeTable`` splits into the words it is said as."""

MASS_NOUNS = frozenset({"milk", "cereal"})
"""Objects a kitchen scene never counts, so their plural is a word nobody says."""

IRREGULAR_PLURALS = {"knife": "knives"}
"""Plurals that adding an ``s`` would spell wrong."""

VOCABULARY_SOURCES = ("surfaces", "containers", "furniture")
"""Context keys holding places rather than objects; nobody asks for two tables."""


def spoken_name(type_name: str) -> str:
    """Return one world type name as the words it is spoken as."""
    return SPOKEN_NAME_BOUNDARY.sub(" ", type_name).lower()


def spoken_plural(word: str) -> str | None:
    """Return the plural of one spoken object name, or None when it has none."""
    if word in MASS_NOUNS:
        return None
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    return f"{word}es" if word.endswith(("s", "x", "ch", "sh")) else f"{word}s"


def spoken_vocabulary(context) -> tuple[str, ...]:
    """Return the words a scene in this world may name, as people say them.

    Speech recognition is offered these as a hint. On a quiet recording Whisper
    hears "marks" where the scene says "mugs", and naming the things that
    actually exist settles it. Objects come in the plural too, because a scene
    asks for four mugs far more often than for one.
    """
    if not context:
        return ()
    words = []
    for type_name in context.get("objects", ()):
        word = spoken_name(type_name)
        words.append(word)
        plural = spoken_plural(word)
        if plural is not None:
            words.append(plural)
    for key in VOCABULARY_SOURCES:
        words.extend(spoken_name(name) for name in context.get(key, ()))
    # The same word twice is no better a hint than once, and the context lists a
    # table as both a surface and a piece of furniture.
    return tuple(dict.fromkeys(words))


def parse_interpretation(
    data, utterance_count: int, context=None
) -> Interpretation:
    """Validate a raw interpretation and return it as an :class:`Interpretation`.

    :raises ValueError: on any deviation, worded so it can be handed back to the
        model as a correction.
    """
    if not isinstance(data, dict):
        raise ValueError("interpretation must be a JSON object")
    if frozenset(data) != INTERPRETATION_KEYS:
        # Spelled through _key_names for the same reason as the item messages:
        # a raw member reads as <InterpretationKey.X: 'x'> and everything in
        # angle brackets disappears once the reason is shown as HTML.
        missing = _key_names(INTERPRETATION_KEYS - frozenset(data))
        extra = sorted(frozenset(data) - INTERPRETATION_KEYS)
        raise ValueError(
            f"interpretation keys must be exactly {_key_names(INTERPRETATION_KEYS)}; "
            f"missing={missing}, extra={extra}"
        )

    known_types = known_object_types(context)
    instruction = data[InterpretationKey.INSTRUCTION]
    if instruction is not None:
        _require_index(instruction, utterance_count, InterpretationKey.INSTRUCTION)
    quantity = _parse_quantity(data[InterpretationKey.QUANTITY], known_types)
    context_items = _parse_context(
        data[InterpretationKey.CONTEXT], utterance_count, known_types
    )
    ignore = _parse_ignore(data[InterpretationKey.IGNORE], utterance_count)
    _check_partition(instruction, context_items, ignore, utterance_count)
    if instruction is None and (context_items or quantity):
        raise ValueError(
            "a scene without an instruction cannot carry 'context' or 'quantity'; "
            "either name the instruction utterance or ignore everything"
        )

    return Interpretation(
        instruction=instruction,
        context=tuple(context_items),
        ignore=tuple(ignore),
        quantity=tuple(quantity),
    )


# %% believability of a claimed change


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
"""Number words the speech front-end writes out instead of as digits."""

IMPLIED_AMOUNT = 1
"""How many an utterance means when it names no number at all.

"He already has a glass" counts one glass, not an unsaid number of them.
"""

_WORD = re.compile(r"[a-z0-9]+")


def spoken_amounts(text: str) -> set[int]:
    """Return every number the utterance names, as digits or as words.

    Only whole words count, so the "one" in "someone" is not a number. ``a`` and
    ``an`` are left out on purpose: they are too common to read as a count, and
    the utterances that use them are covered by :data:`IMPLIED_AMOUNT`.
    """
    amounts = set()
    for word in _WORD.findall(text.lower()):
        if word.isdigit():
            amounts.add(int(word))
        elif word in NUMBER_WORDS:
            amounts.add(NUMBER_WORDS[word])
    return amounts


def check_deltas_are_supported(
    interpretation: Interpretation, utterances: list[SpokenUtterance]
) -> None:
    """Check that no utterance is read as changing more than it names.

    An utterance can only ever account for as many objects as it mentions, and
    one that mentions no number means :data:`IMPLIED_AMOUNT`. The failure this
    catches is the model doing the arithmetic the prompt forbids: told to bring
    three spoons and hearing "I already have one spoon", it answers ``-2`` --
    the result of the subtraction -- where the utterance's own change is ``-1``.

    The bound is on the magnitude rather than the exact number, because an
    utterance may state a target instead of a change: "Make it two" against an
    instruction asking for one is a delta of ``+1``, and that has to stay
    allowed.

    :raises ValueError: worded so it can be handed back to the model as a
        correction.
    """
    for item in interpretation.context:
        if not item.is_quantitative:
            continue
        text = utterances[item.utterance].text
        bound = max(spoken_amounts(text), default=IMPLIED_AMOUNT)
        if abs(item.delta) > bound:
            # Naming the subtraction is what makes the correction land, but only
            # a claim for fewer can have come from one; saying it to a model
            # that claimed too many would point it the wrong way.
            hint = (
                "never subtract it from the number the instruction asks for"
                if item.delta < 0
                else "never scale it by the number the instruction asks for"
            )
            raise ValueError(
                f"context.delta {item.delta} for utterance {item.utterance} "
                f"changes more than the utterance names: {text.strip()!r} "
                f"mentions at most {bound}. State only this utterance's own "
                f"change; {hint}."
            )


def clamp_deltas_to_speech(
    interpretation: Interpretation, utterances: list[SpokenUtterance]
) -> tuple[Interpretation, tuple[int, ...]]:
    """Bring every claimed change inside what its utterance names.

    The same bound :func:`check_deltas_are_supported` refuses on is used here to
    repair instead, for the case where the model was asked again and answered
    the same way: a count that is one too many is worth acting on, where a
    scene thrown away leaves the robot with nothing at all.

    The sign is kept, because whether the utterance adds or removes was never
    in doubt -- only how many. :attr:`ContextItem.effect` is left exactly as the
    model wrote it: it is that utterance's own words about the task, and
    rewriting the prose to match a number the model did not choose would be
    inventing speech. Naming the correction belongs to whoever shows the result.

    :return: the interpretation to use and the utterance indices corrected,
        empty when every claim already fitted.
    """
    corrected: list[int] = []
    items = []
    for item in interpretation.context:
        if item.is_quantitative:
            bound = max(
                spoken_amounts(utterances[item.utterance].text),
                default=IMPLIED_AMOUNT,
            )
            if abs(item.delta) > bound:
                item = replace(item, delta=bound if item.delta > 0 else -bound)
                corrected.append(item.utterance)
        items.append(item)
    if not corrected:
        return interpretation, ()
    return replace(interpretation, context=tuple(items)), tuple(corrected)



# %% counting


def aggregate(
    interpretation: Interpretation, utterances: list[SpokenUtterance]
) -> Aggregate:
    """Sum the quantitative constraints, counting each voice's claim once.

    .. note:: Where a claim carries no voice label it cannot be matched against
        another, so it counts on its own and
        :attr:`Aggregate.assumed_distinct_speakers` is set.
    """
    claims = [item for item in interpretation.context if item.is_quantitative]
    requested = {
        request.object_type: request.count for request in interpretation.quantity
    }

    totals: dict[str, int] = {}
    seen_claims: set[tuple[int, str, int]] = set()
    duplicates: list[int] = []
    assumed_distinct = False
    for item in claims:
        speaker = utterances[item.utterance].speaker_id
        if speaker is None:
            assumed_distinct = True
        else:
            # The voice is part of the key, so one person repeating themselves
            # collapses while two people making the same claim each count.
            claim = (speaker, item.object_type, item.delta)
            if claim in seen_claims:
                duplicates.append(item.utterance)
                continue
            seen_claims.add(claim)
        totals[item.object_type] = totals.get(item.object_type, 0) + item.delta

    final = {
        object_type: max(0, requested[object_type] + total)
        for object_type, total in totals.items()
        if object_type in requested
    }

    return Aggregate(
        totals=totals,
        final=final,
        assumed_distinct_speakers=assumed_distinct,
        duplicate_claims=tuple(duplicates),
    )


# %% instruction wording


def pluralize(object_type: str, count: int) -> str:
    """Return the type name in the number matching the count."""
    if count == 1:
        return object_type
    lowered = object_type.lower()
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return f"{object_type}es"
    if lowered.endswith("fe"):
        return f"{object_type[:-2]}ves"
    if lowered.endswith("f"):
        return f"{object_type[:-1]}ves"
    return f"{object_type}s"


def _quantity_clauses(summary: Aggregate) -> list[str]:
    clauses = []
    for object_type in sorted(summary.totals):
        if object_type in summary.final:
            count = summary.final[object_type]
            clauses.append(f"Bring exactly {count} {pluralize(object_type, count)}.")
            continue
        amount = abs(summary.totals[object_type])
        direction = "fewer" if summary.totals[object_type] < 0 else "more"
        clauses.append(
            f"Bring {amount} {direction} {pluralize(object_type, amount)} "
            "than the instruction asks for."
        )
    return clauses


def fuse(
    interpretation: Interpretation, utterances: list[SpokenUtterance]
) -> str | None:
    """Compose the single instruction for the planner, or None without an instruction.

    Quantitative constraints become one counted clause per object type; every
    other constraint contributes its own clause.
    """
    if interpretation.instruction is None:
        return None
    base = utterances[interpretation.instruction].text.strip()
    summary = aggregate(interpretation, utterances)
    other_clauses = [
        item.effect.strip()
        for item in interpretation.context
        if not item.is_quantitative
    ]
    clauses = _quantity_clauses(summary) + other_clauses
    return " ".join([base, *clauses]) if clauses else base
