"""Asks a model to triage a scene and checks the answer before using it.

One pass builds the two messages, generates an answer, parses it as JSON and runs
it through the guard in :mod:`thesis_demo.dialogue.schema`. A rejected answer is
returned to the model together with the reason, up to ``max_attempts`` times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from thesis_demo.dialogue.prompt import ANSWER_SHAPE, system_prompt, user_turn
from thesis_demo.dialogue.schema import (
    Aggregate,
    Interpretation,
    SpokenUtterance,
    aggregate,
    fuse,
    parse_interpretation,
)

DEFAULT_MAX_ATTEMPTS = 2
"""How often the model may answer again after a rejection."""

DEFAULT_TEMPERATURE = 0.1
"""Sampling temperature for the triage, low so the answer stays reproducible."""

GENERATE_UNTIL_END_OF_TEXT = -1
"""Token limit meaning "generate until the model stops"."""

TRIAGE_TOKEN_LIMIT = 512
"""Most tokens one triage answer may take.

The answer is a single small JSON object, a few hundred tokens at most. Letting
it run to the model's own stopping point sounds harmless but is not: the
context is the model's native one, tens of thousands of tokens wide, so an
answer that never stops keeps generating for hours with nothing to show. A cap
this far above a real answer only ever cuts off a runaway one, which was
unusable anyway, and turns a hang into a rejection the loop can retry.
"""


class Outcome(StrEnum):
    """What came of interpreting a scene."""

    OK = "ok"
    NO_INSTRUCTION = "no_instruction"
    ERROR = "error"


class MessageRole(StrEnum):
    """Speaker of one chat message sent to the model."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MessageKey(StrEnum):
    """Keys of one chat message."""

    ROLE = "role"
    CONTENT = "content"


# %% result


@dataclass(frozen=True)
class InterpretationResult:
    """The triage of one scene, whether or not it succeeded."""

    outcome: Outcome
    """Whether an instruction was found, none was, or no valid answer arrived."""

    instruction: str | None
    """The single sentence for the planner, absent without an instruction."""

    interpretation: Interpretation | None
    """The validated interpretation, absent on :attr:`Outcome.ERROR`."""

    instruction_text: str | None
    """Words of the utterance that instructs the robot."""

    constraints: tuple[str, ...]
    """Effect of every accepted background utterance, in the order heard."""

    ignored: tuple[str, ...]
    """Words of every utterance judged not to affect the task."""

    raw_response: str
    """Last answer the model gave, kept for inspection."""

    attempts: int
    """How many answers were needed."""

    rejection_reason: str | None = None
    """Why the last answer was refused, on :attr:`Outcome.ERROR`."""

    counts: Aggregate | None = None
    """How the counted constraints added up."""


# %% the loop


def _correction_message(reason: str) -> str:
    return (
        f"Your answer was rejected: {reason}\n"
        f"Reply again with exactly one JSON object of this shape:\n{ANSWER_SHAPE}\n"
        "Write every key shown, using null where a value is absent; never leave "
        "a key out. Use only the utterance indices shown and assign every index "
        "exactly once. No explanation, no markdown."
    )


def _chat_message(role: MessageRole, content: str) -> dict[str, str]:
    return {MessageKey.ROLE: role.value, MessageKey.CONTENT: content}


def _parse(raw_response: str, utterance_count: int, context):
    try:
        data = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        return None, "Output must be exactly one JSON object."
    try:
        return parse_interpretation(data, utterance_count, context), None
    except (ValueError, KeyError, TypeError) as error:
        return None, str(error)


def interpret(
    utterances: list[SpokenUtterance],
    context,
    generate,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> InterpretationResult:
    """Triage a recorded scene into instruction, relevant context and noise.

    :param generate: maps a list of chat messages to the model's raw text reply.
        Passing it in keeps this loop independent of any particular model.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    messages = [
        _chat_message(MessageRole.SYSTEM, system_prompt()),
        _chat_message(MessageRole.USER, user_turn(utterances, context)),
    ]

    attempt_reasons = []
    raw_response = ""
    interpretation = None
    attempts = 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        raw_response = generate(messages)
        interpretation, reason = _parse(raw_response, len(utterances), context)
        if interpretation is not None:
            break
        attempt_reasons.append(reason)
        if attempt < max_attempts:
            messages = messages + [
                _chat_message(MessageRole.ASSISTANT, raw_response),
                _chat_message(MessageRole.USER, _correction_message(reason)),
            ]

    if interpretation is None:
        return InterpretationResult(
            outcome=Outcome.ERROR,
            instruction=None,
            interpretation=None,
            instruction_text=None,
            constraints=(),
            ignored=(),
            raw_response=raw_response,
            attempts=attempts,
            rejection_reason=attempt_reasons[-1] if attempt_reasons else "unknown",
        )

    has_instruction = interpretation.instruction is not None
    return InterpretationResult(
        outcome=Outcome.OK if has_instruction else Outcome.NO_INSTRUCTION,
        instruction=fuse(interpretation, utterances),
        interpretation=interpretation,
        instruction_text=(
            utterances[interpretation.instruction].text
            if has_instruction
            else None
        ),
        constraints=tuple(item.effect for item in interpretation.context),
        ignored=tuple(utterances[index].text for index in interpretation.ignore),
        raw_response=raw_response,
        attempts=attempts,
        counts=aggregate(interpretation, utterances),
    )


# %% model backend


def planner_backend(
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 0,
    max_tokens: int = TRIAGE_TOKEN_LIMIT,
):
    """Build a ``generate`` callable backed by the planner's loaded GGUF model.

    :raises RuntimeError: when no model has been loaded yet.
    """
    # Imported here rather than at module level so this module stays importable
    # where llama.cpp is absent.
    from thesis_demo.planner import llm as planner_llm

    def generate(messages):
        if planner_llm._state.llm is None:
            raise RuntimeError(
                "Planner LLM not initialized. Call setup_planner() first."
            )
        completion = planner_llm._state.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
        )
        return completion["choices"][0]["message"]["content"]

    return generate
