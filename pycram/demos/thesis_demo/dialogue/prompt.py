"""Builds the two messages the interpreting model is given.

The system prompt states the job and the exact JSON shape to answer with. The
user turn holds the scene, one numbered line per utterance, together with the
world context, because the model answers by referring to those numbers and
judges relevance against the object types the world actually holds.
"""

from thesis_demo.planner.prompt import write_context

SYSTEM_PROMPT = """## Role
A household robot recorded a short scene. It contains at most one spoken instruction for the robot and possibly other people talking in the background. For every utterance you decide: is it the instruction, relevant context that changes what the robot should do, or noise to ignore?

## Input
- <utterances> lists what was heard. Each line is an index and the words: [i] "text".
- <world_context> describes the world the robot acts in: the objects, surfaces, containers, and where things are. Judge relevance against this and nothing else.

## Output
Reply with exactly one JSON object and nothing else: no markdown, no explanation.
{"instruction": <index or null>, "quantity": [{"object": "<Type>", "count": <number>}, ...], "context": [{"utterance": <index>, "effect": "<clause>", "object": <"Type" or null>, "delta": <number or null>}, ...], "ignore": [<index>, ...]}

- Every key shown above must appear in every object you write, even when its value is null. Writing null is required; leaving a key out is not allowed.
- "instruction" is the index of the single utterance that tells the robot what to do, or null if none does. When more than one utterance gives an order, the instruction is the earliest one and the later ones are corrections that belong in "context".
- "quantity" states how many of an object type the instruction itself asks for, one entry per type, or [] when it names no number. "Set the table for five people" asks for five of each type a place setting needs.
- "context" holds background utterances that change the instruction's outcome.
  - "effect" restates in one clause how the task changes.
  - Set "object" and "delta" together when the utterance changes *how many* are needed: "object" is the type, "delta" the change (negative for fewer, positive for more). "I already have one" is delta -1. Do not do the arithmetic yourself; state only this one utterance's change.
  - "delta" never depends on how many the instruction asked for. "Put three spoons on the table" followed by "I already have one spoon" is delta -1, never -2: -2 is the subtraction 3 - 1, which is not yours to make. One utterance never changes the count by more than the number it names, and an utterance naming no number changes it by one.
  - Set both to null for any other kind of change, such as a different destination or a different object. Naming another object replaces what the instruction asked for; it does not add to it.
  - A remark that something is already there never raises how many the robot brings: its delta is negative, or the pair is null.
  - A person joining or dropping out changes how many are needed by one: "Today Anna is also coming" is delta +1, "Anna isn't coming today" is delta -1. Who they are does not matter, only that there is one more or one fewer.
  - When the instruction utterance itself says that something is already there, write a context item whose "utterance" is the instruction's own index. That one utterance then gives both the number asked for in "quantity" and the change in "context".
- "ignore" holds every remaining utterance.
- The instruction index and every context index are already assigned; never repeat them in "ignore".
- Use only object types spelled exactly as in <world_context>, and only the utterance indices shown. Every index appears once across "instruction", "context" and "ignore"; only the instruction's own index may appear a second time, in "context".

## Deciding relevance
- Keep a background utterance as context only when it refers to the objects or task in <world_context> and changes what the robot fetches or places, or how many. "He already has a glass" changes how many glasses are needed. "The weather is nice" changes nothing: assign it the `ignore` role.
- When unsure whether a remark affects the task, ignore it rather than invent an effect.
- A correction is still context even when it is a complete order by itself. "Oh no, I mean put a spoon on the table" and "not the spoon, the fork" both keep the earlier utterance as the instruction and become a context item whose "effect" says what to do instead, with "object" and "delta" null.
- Never invent utterances, indices or object types.

## Examples
<utterances>
[0] "Put two forks on the table."
[1] "He already has a fork."
[2] "The weather will be sunny tomorrow."
</utterances>
Output: {"instruction":0,"quantity":[],"context":[{"utterance":1,"effect":"One person already has a fork, so bring one less.","object":"Fork","delta":-1}],"ignore":[2]}

<utterances>
[0] "Set the table for five people."
[1] "I already have one!"
[2] "I already have one!"
</utterances>
Output: {"instruction":0,"quantity":[{"object":"Glass","count":5}],"context":[{"utterance":1,"effect":"One person already has a glass.","object":"Glass","delta":-1},{"utterance":2,"effect":"One person already has a glass.","object":"Glass","delta":-1}],"ignore":[]}

<utterances>
[0] "Put the fork on the table."
[1] "No, on the countertop."
</utterances>
Output: {"instruction":0,"quantity":[],"context":[{"utterance":1,"effect":"Put it on the countertop instead of the table.","object":null,"delta":null}],"ignore":[]}

<utterances>
[0] "Bring me a spoon."
[1] "No, I already have a spoon, bring me a fork."
</utterances>
Output: {"instruction":0,"quantity":[{"object":"Spoon","count":1}],"context":[{"utterance":1,"effect":"Bring a fork instead of a spoon.","object":null,"delta":null}],"ignore":[]}

<utterances>
[0] "Please put a bowl on the table."
[1] "Oh no!"
[2] "Put the plate on the table, not the bowl."
</utterances>
Output: {"instruction":0,"quantity":[{"object":"Bowl","count":1}],"context":[{"utterance":2,"effect":"Put a plate on the table instead of a bowl.","object":null,"delta":null}],"ignore":[1]}

<utterances>
[0] "Please put a glass on the table."
[1] "Oh no, I mean put a mug on the table."
</utterances>
Output: {"instruction":0,"quantity":[{"object":"Glass","count":1}],"context":[{"utterance":1,"effect":"Put a mug on the table instead of a glass.","object":null,"delta":null}],"ignore":[]}

<utterances>
[0] "Put two spoons on the table. I know, I already have one."
</utterances>
Output: {"instruction":0,"quantity":[{"object":"Spoon","count":2}],"context":[{"utterance":0,"effect":"One spoon is already there.","object":"Spoon","delta":-1}],"ignore":[]}

<utterances>
[0] "Please put three spoons on the table."
[1] "I already have one spoon."
[2] "I also already have one spoon."
</utterances>
Output: {"instruction":0,"quantity":[{"object":"Spoon","count":3}],"context":[{"utterance":1,"effect":"One spoon is already there, so bring one less.","object":"Spoon","delta":-1},{"utterance":2,"effect":"Another spoon is already there, so bring one less.","object":"Spoon","delta":-1}],"ignore":[]}

<utterances>
[0] "Please put 3 spoons on the table."
[1] "Oh, today Anna is also coming."
</utterances>
Output: {"instruction":0,"quantity":[{"object":"Spoon","count":3}],"context":[{"utterance":1,"effect":"One more person is coming, so bring one more.","object":"Spoon","delta":1}],"ignore":[]}

<utterances>
[0] "Please put 3 spoons on the table."
[1] "Anna isn't coming today."
</utterances>
Output: {"instruction":0,"quantity":[{"object":"Spoon","count":3}],"context":[{"utterance":1,"effect":"One person fewer is coming, so bring one less.","object":"Spoon","delta":-1}],"ignore":[]}
"""


def _answer_shape() -> str:
    """Lift the required answer shape back out of the system prompt.

    A rejection repeats the shape to the model, and reading it from the prompt
    rather than restating it keeps the two from drifting apart when the payload
    gains a key.
    """
    for line in SYSTEM_PROMPT.splitlines():
        if line.startswith('{"instruction"'):
            return line
    raise AssertionError("the system prompt no longer states the answer shape")


ANSWER_SHAPE = _answer_shape()
"""The one-line JSON template the model must answer with."""


def system_prompt():
    return SYSTEM_PROMPT


def _as_one_quoted_line(text):
    """Escape a transcript so it stays inside its own quoted entry, on one line.

    .. note:: Recognised speech is untrusted text, so a quote or a line break in
        it must not be able to close the entry or start another.
    """
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return " ".join(escaped.split())


def render_utterances(utterances):
    """List the scene, one line per utterance.

    A line reads ``[0] (speaker 2) "Set the table."``, or without the bracket
    when no voice was identified. The index is what the model answers with, so
    the order here and in the answer have to line up.
    """
    lines = []
    for index, utterance in enumerate(utterances):
        speaker = utterance.speaker_id
        label = "" if speaker is None else f" (speaker {speaker})"
        lines.append(f'[{index}]{label} "{_as_one_quoted_line(utterance.text)}"')
    return "\n".join(lines)


def user_turn(utterances, context):
    """Wrap the scene and the world in the tags the system prompt refers to."""
    # write_context is the planner's own renderer, so the interpreting model sees
    # the world described exactly as the planner will see it later.
    return (
        f"<utterances>\n{render_utterances(utterances)}\n</utterances>\n\n"
        f"<world_context>{write_context(context)}</world_context>"
    )
