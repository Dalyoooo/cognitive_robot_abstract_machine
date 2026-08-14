"""Builds the two messages the interpreting model is given.

The system prompt states the job and the exact JSON shape to answer with. The
user turn holds the scene, one numbered line per utterance, together with the
world context, because the model answers by referring to those numbers and
judges relevance against the object types the world actually holds.
"""

from thesis_demo.planner.prompt import write_context

SYSTEM_PROMPT = """## Role
A household robot recorded a short scene. It contains at most one spoken command for the robot and possibly other people talking in the background. For every utterance you decide: is it the command, relevant context that changes what the robot should do, or noise to ignore?

## Input
- <utterances> lists what was heard. Each line is an index and the words: [i] "text".
- <world_context> describes the world the robot acts in: the objects, surfaces, containers, and where things are. Judge relevance against this and nothing else.

## Output
Reply with exactly one JSON object and nothing else: no markdown, no explanation.
{"command": <index or null>, "quantity": [{"object": "<Type>", "count": <number>}, ...], "context": [{"utterance": <index>, "effect": "<clause>", "object": <"Type" or null>, "delta": <number or null>}, ...], "ignore": [<index>, ...]}

- "command" is the index of the single utterance that tells the robot what to do, or null if none does.
- "quantity" states how many of an object type the command itself asks for, one entry per type, or [] when it names no number. "Set the table for five people" asks for five of each type a place setting needs.
- "context" holds background utterances that change the command's outcome.
  - "effect" restates in one clause how the task changes.
  - Set "object" and "delta" together when the utterance changes *how many* are needed: "object" is the type, "delta" the change (negative for fewer, positive for more). "I already have one" is delta -1. Do not do the arithmetic yourself; state only this one utterance's change.
  - Set both to null for any other kind of change, such as a different destination or a different object.
- "ignore" holds every remaining utterance.
- Use only object types spelled exactly as in <world_context>, and only the utterance indices shown. Every index must appear exactly once across "command", "context" and "ignore".

## Deciding relevance
- Keep a background utterance as context only when it refers to the objects or task in <world_context> and changes what the robot fetches or places, or how many. "He already has a glass" changes how many glasses are needed. "The weather is nice" changes nothing: ignore it.
- When unsure whether a remark affects the task, ignore it rather than invent an effect.
- Never invent utterances, indices or object types.

## Examples
<utterances>
[0] "Set the table."
[1] "He already has a glass."
[2] "The weather will be sunny tomorrow."
</utterances>
Output: {"command":0,"quantity":[],"context":[{"utterance":1,"effect":"One person already has a glass, so bring one less.","object":"Glass","delta":-1}],"ignore":[2]}

<utterances>
[0] "Set the table for five people."
[1] "I already have one!"
[2] "I already have one!"
</utterances>
Output: {"command":0,"quantity":[{"object":"Glass","count":5}],"context":[{"utterance":1,"effect":"One person already has a glass.","object":"Glass","delta":-1},{"utterance":2,"effect":"One person already has a glass.","object":"Glass","delta":-1}],"ignore":[]}

<utterances>
[0] "Put the cup on the table."
[1] "No, on the countertop."
</utterances>
Output: {"command":0,"quantity":[],"context":[{"utterance":1,"effect":"Put it on the countertop instead of the table.","object":null,"delta":null}],"ignore":[]}
"""


def system_prompt():
    return SYSTEM_PROMPT


def _as_one_quoted_line(text):
    """Keep a transcript inside its own quoted entry, on a single line.

    Recognised speech is untrusted text: an unescaped quote followed by a line
    break would otherwise render as a further ``[i] "..."`` entry and invent an
    utterance nobody said.
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
