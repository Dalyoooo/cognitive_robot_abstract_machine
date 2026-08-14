"""Checks the model's answer and works out the numbers it implies.

An interpretation says which utterance is the command, which of the others
change what the robot should do, and which are ignored. ``parse_interpretation``
accepts it only if the shape is exact, every index exists, every object type
appears in the world context, and each utterance got exactly one role.

``aggregate`` then adds up the changes in how many of something is needed, and
``fuse`` writes the single instruction that goes to the planner. Counting
happens here in Python rather than in the prompt, so the planner is handed one
number instead of several remarks to add up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The answer must carry exactly these four keys, no more and no fewer.
INTERPRETATION_KEYS = {"command", "quantity", "context", "ignore"}
CONTEXT_ITEM_KEYS = {"utterance", "effect", "object", "delta"}
QUANTITY_KEYS = {"object", "count"}

# A household scene never legitimately shifts a count by more than this, so a
# wilder number is a model slip rather than something to act on.
MAX_ABS_DELTA = 20
MAX_COUNT = 100


@dataclass(frozen=True)
class QuantityRequest:
    """How many of one object type the command itself asks for."""

    object: str
    count: int


@dataclass(frozen=True)
class ContextItem:
    """A background utterance kept because it changes the command's outcome.

    ``object`` and ``delta`` are set only for utterances that change *how many*
    of something is needed; they are counted symbolically instead of being left
    to the planner's arithmetic. Constraints that change something else (a
    destination, a substitution) carry ``effect`` alone.
    """

    utterance: int
    effect: str
    object: str | None = None
    delta: int | None = None

    @property
    def is_quantitative(self):
        return self.object is not None and self.delta is not None


@dataclass(frozen=True)
class Interpretation:
    """How a recorded scene splits into command, relevant context and noise.

    Every utterance index of the scene appears exactly once across ``command``,
    ``context`` and ``ignore`` -- nothing heard is silently dropped or counted
    twice. ``command`` is None only when no utterance addresses the robot.
    """

    command: int | None
    context: tuple[ContextItem, ...]
    ignore: tuple[int, ...]
    quantity: tuple[QuantityRequest, ...] = ()


@dataclass(frozen=True)
class Aggregate:
    """The summed effect of every quantitative constraint, per object type.

    ``totals`` maps an object type to the net change in how many are needed.
    ``final`` maps it to the resulting absolute count, but only where the
    command named one. ``assumed_distinct_speakers`` is True when claims had to
    be counted as coming from different people because no speaker labels were
    available -- the count is then only as good as that assumption.
    """

    totals: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)
    assumed_distinct_speakers: bool = False
    duplicate_claims: tuple = ()


def _is_index(value, utterance_count):
    # bool is an int subclass; a JSON true/false is never a valid index.
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < utterance_count
    )


def _require_index(value, utterance_count, role):
    if not _is_index(value, utterance_count):
        raise ValueError(
            f"{role} must be an utterance index in 0..{utterance_count - 1}, "
            f"got {value!r}"
        )
    return value


def _require_object(value, known_objects, role):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} must be a non-empty object type")
    name = value.strip()
    if known_objects is not None and name not in known_objects:
        choices = ", ".join(sorted(known_objects)) or "none"
        raise ValueError(
            f"unknown {role} type {name!r}; choose from: {choices}"
        )
    return name


def _require_delta(value):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"delta must be a whole number, got {value!r}")
    if value == 0:
        raise ValueError("delta must not be 0; omit the constraint instead")
    if abs(value) > MAX_ABS_DELTA:
        raise ValueError(f"delta {value} is out of range (max {MAX_ABS_DELTA})")
    return value


def _require_count(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"count must be a whole number >= 0, got {value!r}")
    if value > MAX_COUNT:
        raise ValueError(f"count {value} is out of range (max {MAX_COUNT})")
    return value


def _parse_quantity(value, known_objects):
    if not isinstance(value, list):
        raise ValueError("'quantity' must be a JSON array")
    requests = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != QUANTITY_KEYS:
            raise ValueError(
                f"each 'quantity' item needs exactly the keys {sorted(QUANTITY_KEYS)}"
            )
        name = _require_object(entry["object"], known_objects, "quantity.object")
        if any(request.object == name for request in requests):
            raise ValueError(f"quantity lists {name!r} more than once")
        requests.append(
            QuantityRequest(object=name, count=_require_count(entry["count"]))
        )
    return requests


def _parse_context(value, utterance_count, known_objects):
    if not isinstance(value, list):
        raise ValueError("'context' must be a JSON array")
    items = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != CONTEXT_ITEM_KEYS:
            raise ValueError(
                "each 'context' item needs exactly the keys "
                f"{sorted(CONTEXT_ITEM_KEYS)}"
            )
        index = _require_index(entry["utterance"], utterance_count, "context.utterance")
        effect = entry["effect"]
        if not isinstance(effect, str) or not effect.strip():
            raise ValueError("context.effect must be a non-empty string")

        object_name, delta = entry["object"], entry["delta"]
        if (object_name is None) != (delta is None):
            raise ValueError(
                "context items need 'object' and 'delta' either both set (a "
                "change in how many) or both null (any other change)"
            )
        if object_name is not None:
            object_name = _require_object(
                object_name, known_objects, "context.object"
            )
            delta = _require_delta(delta)

        items.append(
            ContextItem(
                utterance=index,
                effect=effect.strip(),
                object=object_name,
                delta=delta,
            )
        )
    return items


def _parse_ignore(value, utterance_count):
    if not isinstance(value, list):
        raise ValueError("'ignore' must be a JSON array")
    return [_require_index(index, utterance_count, "ignore") for index in value]


def _check_partition(command, context_items, ignore, utterance_count):
    indices = []
    if command is not None:
        indices.append(command)
    indices.extend(item.utterance for item in context_items)
    indices.extend(ignore)

    seen, duplicates = set(), set()
    for index in indices:
        if index in seen:
            duplicates.add(index)
        seen.add(index)
    if duplicates:
        raise ValueError(
            f"utterance(s) {sorted(duplicates)} assigned to more than one role"
        )
    missing = sorted(set(range(utterance_count)) - seen)
    if missing:
        raise ValueError(
            f"utterance(s) {missing} left unassigned; every index must be the "
            "command, context or ignored"
        )


def known_object_names(context):
    """Object types the scene may talk about, taken from the world context."""
    if not context:
        return None
    names = set(context.get("objects", []))
    return names or None


def parse_interpretation(data, utterance_count, context=None):
    """Validate a raw interpretation object and return an :class:`Interpretation`.

    Raises ValueError on any deviation, so the interpreter can hand the message
    back to the model as a correction, exactly like the planner's guard. Object
    types are checked against the world context, so a count can never be
    attached to something the world does not hold.
    """
    if not isinstance(data, dict):
        raise ValueError("interpretation must be a JSON object")
    if set(data) != INTERPRETATION_KEYS:
        missing = sorted(INTERPRETATION_KEYS - set(data))
        extra = sorted(set(data) - INTERPRETATION_KEYS)
        raise ValueError(
            f"interpretation keys must be exactly {sorted(INTERPRETATION_KEYS)}; "
            f"missing={missing}, extra={extra}"
        )

    known_objects = known_object_names(context)
    command = data["command"]
    if command is not None:
        _require_index(command, utterance_count, "command")
    quantity = _parse_quantity(data["quantity"], known_objects)
    context_items = _parse_context(data["context"], utterance_count, known_objects)
    ignore = _parse_ignore(data["ignore"], utterance_count)
    _check_partition(command, context_items, ignore, utterance_count)
    if command is None and (context_items or quantity):
        # A constraint modifies a command, so without one there is nothing for
        # it to change; keeping it would record counts that never reach a plan.
        raise ValueError(
            "a scene without a command cannot carry 'context' or 'quantity'; "
            "either name the command utterance or ignore everything"
        )

    return Interpretation(
        command=command,
        context=tuple(context_items),
        ignore=tuple(ignore),
        quantity=tuple(quantity),
    )


def aggregate(interpretation, utterances):
    """Sum the quantitative constraints, counting each speaker's claim once.

    Two people each saying "I already have one" means two fewer; one person
    saying it twice means one fewer. Without speaker labels the two cannot be
    told apart, so identical claims are counted as distinct people -- the safer
    reading for a room of several speakers -- and the result records that the
    assumption was needed.
    """
    claims = [item for item in interpretation.context if item.is_quantitative]
    requested = {request.object: request.count for request in interpretation.quantity}

    totals, seen_claims, duplicates = {}, set(), []
    assumed_distinct = False
    for item in claims:
        speaker = utterances[item.utterance].speaker_id
        if speaker is None:
            # No label to compare, so this claim cannot be merged with another.
            assumed_distinct = True
        else:
            # The key holds the voice as well as the claim, so the same person
            # repeating themselves collapses while two people making the same
            # claim stay separate and each count.
            claim = (speaker, item.object, item.delta)
            if claim in seen_claims:
                duplicates.append(item.utterance)
                continue
            seen_claims.add(claim)
        totals[item.object] = totals.get(item.object, 0) + item.delta

    final = {}
    for object_name, total in totals.items():
        # An absolute count is only possible where the command named one; five
        # requested and three already taken care of leaves two, never below zero.
        if object_name in requested:
            final[object_name] = max(0, requested[object_name] + total)

    return Aggregate(
        totals=totals,
        final=final,
        assumed_distinct_speakers=assumed_distinct,
        duplicate_claims=tuple(duplicates),
    )


def plural(object_name, count):
    """Best-effort plural of a world-context type name, for readable clauses."""
    if count == 1:
        return object_name
    lowered = object_name.lower()
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return f"{object_name}es"
    if lowered.endswith("fe"):
        return f"{object_name[:-2]}ves"
    if lowered.endswith("f"):
        return f"{object_name[:-1]}ves"
    return f"{object_name}s"


def _quantity_clauses(summary):
    clauses = []
    for object_name in sorted(summary.totals):
        total = summary.totals[object_name]
        if object_name in summary.final:
            count = summary.final[object_name]
            clauses.append(
                f"Bring exactly {count} {plural(object_name, count)}."
            )
            continue
        amount = abs(total)
        direction = "fewer" if total < 0 else "more"
        clauses.append(
            f"Bring {amount} {direction} {plural(object_name, amount)} "
            "than the command asks for."
        )
    return clauses


def fuse(interpretation, utterances):
    """Compose the single instruction that goes to the planner.

    Quantitative constraints collapse into one counted clause per object type,
    so three people each saying "I already have one" yields one clear number
    instead of three repeated remarks. Other constraints keep their own clause.
    Returns None when there is no command.
    """
    if interpretation.command is None:
        return None
    # The command text carries the task; the clauses are appended to it, so the
    # planner receives one ordinary sentence and needs to know nothing about how
    # the scene was triaged.
    base = utterances[interpretation.command].text.strip()
    summary = aggregate(interpretation, utterances)
    other_clauses = [
        item.effect.strip()
        for item in interpretation.context
        if not item.is_quantitative
    ]
    clauses = _quantity_clauses(summary) + other_clauses
    return " ".join([base, *clauses]) if clauses else base
