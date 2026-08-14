"""Asks the model to triage a scene, and checks what comes back.

One pass is: build the two messages, generate an answer, parse it as JSON, run it
through the guard in :mod:`thesis_demo.dialogue.schema`. If the guard rejects it,
the reason is appended to the conversation and the model answers again, up to
``max_attempts`` times.

``generate`` is passed in rather than imported so the loop can run against a stub
in tests, and against the planner's already-loaded model in the demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from thesis_demo.dialogue.prompt import system_prompt, user_turn
from thesis_demo.dialogue.schema import (
    Aggregate,
    Interpretation,
    aggregate,
    fuse,
    parse_interpretation,
)


@dataclass(frozen=True)
class InterpretationResult:
    # "ok": a command was found. "no_command": the scene held no command for the
    # robot. "error": no valid inter
    # pretation after every attempt.
    outcome: str
    instruction: str | None
    interpretation: Interpretation | None
    command_text: str | None
    constraints: tuple[str, ...]
    ignored: tuple[str, ...]
    raw_response: str
    attempts: int
    rejection_reason: str | None = None
    # How the counted constraints added up, and whether counting had to assume
    # that identical claims came from different people.
    counts: Aggregate | None = None


def _correction_message(reason):
    return (
        f"Your answer was rejected: {reason}\n"
        "Reply again with exactly one JSON object of the required shape. "
        "Use only the utterance indices shown and assign every index exactly "
        "once. No explanation, no markdown."
    )


def _parse(raw_response, utterance_count, context):
    try:
        data = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        return None, "Output must be exactly one JSON object."
    try:
        return parse_interpretation(data, utterance_count, context), None
    except (ValueError, KeyError, TypeError) as error:
        return None, str(error)


def interpret(utterances, context, generate, max_attempts=2):
    """Triage a recorded scene into command, context and noise.

    ``generate`` is any callable that maps a list of chat messages to the
    model's raw text reply. Keeping it injectable lets the deterministic parts
    run under a mock here and against the real GGUF model in the planner
    environment, without this
     module depending on the model.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_turn(utterances, context)},
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
            # The rejected answer and the reason for it are added to the
            # conversation, so the next attempt sees what was wrong with the last.
            messages = messages + [
                {"role": "assistant", "content": raw_response},
                {"role": "user", "content": _correction_message(reason)},
            ]

    if interpretation is None:
        return InterpretationResult(
            outcome="error",
            instruction=None,
            interpretation=None,
            command_text=None,
            constraints=(),
            ignored=(),
            raw_response=raw_response,
            attempts=attempts,
            rejection_reason=attempt_reasons[-1] if attempt_reasons else "unknown",
        )

    command_text = (
        utterances[interpretation.command].text
        if interpretation.command is not None
        else None
    )
    return InterpretationResult(
        outcome="ok" if interpretation.command is not None else "no_command",
        instruction=fuse(interpretation, utterances),
        interpretation=interpretation,
        command_text=command_text,
        constraints=tuple(item.effect for item in interpretation.context),
        ignored=tuple(utterances[index].text for index in interpretation.ignore),
        raw_response=raw_response,
        attempts=attempts,
        counts=aggregate(interpretation, utterances),
    )


def planner_backend(temperature=0.1, seed=0, max_tokens=-1):
    """Build a ``generate`` callable backed by the planner's loaded GGUF model.

    Imported lazily so this module stays usable (and testable) in environments
    without llama.cpp. Reuses the single model instance setup_planner() loaded;
    it does not load a second one.
    """
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
