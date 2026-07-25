from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from thesis_demo.config import select_backend
from thesis_demo.planner.prompt import system_prompt, user_instruction, user_turn
from thesis_demo.validation.guard import verify
from thesis_demo.validation.schema import parse_clarification, parse_plan

MAX_NEW_TOKENS = 2048
N_CTX = 16384  # context window


@dataclass(frozen=True)
class InferenceConfiguration:
    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = MAX_NEW_TOKENS
    base_seed: int | None = None
    max_attempts: int = 1

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def for_case(self, case_identifier):
        base_seed = self.seed if self.base_seed is None else self.base_seed
        digest = sha256(f"{base_seed}:{case_identifier}".encode()).digest()
        case_seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
        return replace(self, seed=case_seed, base_seed=base_seed)


@dataclass(frozen=True)
class PlannerMetrics:
    json_valid: bool
    schema_valid: bool
    guard_valid: bool
    rejection_reason: str | None
    raw_response: str
    attempts: int = 1
    attempt_reasons: list = field(default_factory=list)


@dataclass(frozen=True)
class PlannerResult:
    outcome: str
    payload: object
    history: list
    metrics: PlannerMetrics


@dataclass
class _PlannerState:
    llm: object = field(default=None)
    error: str | None = field(default=None)
    inference: InferenceConfiguration = field(default_factory=InferenceConfiguration)


_state = _PlannerState()


def setup_planner(
    model,
    gguf_file=None,
    n_gpu_layers=None,
    n_ctx=N_CTX,
    inference=None,
):
    _state.llm = None
    _state.error = None
    _state.inference = inference or InferenceConfiguration()
    try:
        print(f"[planner] loading GGUF {model!r} ({gguf_file!r}) ...", flush=True)
        _load_gguf(model, gguf_file, n_gpu_layers, n_ctx)
        print(f"[planner] GGUF {model!r} ready", flush=True)
    except (ImportError, RuntimeError, OSError) as gguf_error:
        _state.error = f"GGUF load failed for {model!r}: {gguf_error!r}"
        print(f"[planner] {_state.error}", flush=True)
        raise


def load_planner():
    gguf_model = os.environ.get("PLANNER_GGUF_MODEL", "wijan/action-planner-gguf")
    gguf_file = os.environ.get("PLANNER_GGUF_FILE", "qwen2.5-3b-instruct.Q4_K_M.gguf")
    setup_planner(gguf_model, gguf_file)


def _resolve_gguf_path(model, gguf_file):
    if gguf_file and Path(gguf_file).exists():
        return gguf_file
    if gguf_file == model:
        raise RuntimeError(
            "PLANNER_GGUF_FILE must be a GGUF filename inside "
            f"{model!r}, not the repository ID."
        )
    try:
        return hf_hub_download(repo_id=model, filename=gguf_file, local_files_only=True)
    except OSError:
        return hf_hub_download(
            repo_id=model, filename=gguf_file, local_files_only=False
        )


def _load_gguf(model, gguf_file, n_gpu_layers, n_ctx):
    if n_gpu_layers is None:
        n_gpu_layers = -1 if select_backend() == "cuda" else 0
    model_path = _resolve_gguf_path(model, gguf_file)
    _state.llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        n_threads=2,
        n_threads_batch=2,  # leave cores free for JupyterLab / RViz
    )


def is_planner_ready():
    return _state.llm is not None


def get_error():
    return _state.error


def plan(
    transcript,
    conversation=None,
    context=None,
    inference=None,
):
    messages = (conversation or []) + [{"role": "user", "content": transcript}]
    events = _run_loop(messages, context or {}, inference)
    for event in events:
        if event["type"] == "done":
            return PlannerResult(
                outcome=event["outcome"],
                payload=event["payload"],
                history=event["history"],
                metrics=event["metrics"],
            )


def plan_stream(transcript, conversation=None, context=None, inference=None):
    messages = (conversation or []) + [{"role": "user", "content": transcript}]
    return _run_loop(messages, context or {}, inference)


def _parse_json_response(text):
    # Anything but exactly one JSON object is a validation failure.
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _generate_response(messages, inference=None):
    inference = inference or _state.inference
    stream = _state.llm.create_chat_completion(
        messages=messages,
        max_tokens=inference.max_tokens,
        seed=inference.seed,
        stream=True,
        temperature=inference.temperature,
    )
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            yield delta


def _prepare_messages(messages, context):
    system = {"role": "system", "content": system_prompt()}
    history_messages = list(messages)
    model_messages = [dict(message) for message in messages]
    latest_user_found = False
    for message in reversed(model_messages):
        if message["role"] == "user":
            if not latest_user_found:
                message["content"] = user_turn(message["content"], context)
                latest_user_found = True
            else:
                message["content"] = user_instruction(message["content"])
    return [system] + model_messages, history_messages


def _correction_message(rejection_reason):
    return (
        f"Your answer was rejected: {rejection_reason}\n"
        "Please answer again with exactly one JSON object and try to fix this problem. "
        "Use the given world-context and use this as the only source of truth. "
        "No explanation, no apology, no Markdown."
    )


def _assess_response(parsed_response, context):
    if parsed_response is None:
        return False, "Output must be exactly one JSON object."
    return verify(parsed_response, context)


def _run_loop(messages, context, inference=None):
    if _state.llm is None:
        raise RuntimeError("Planner not initialized. Call setup_planner() first.")

    inference = inference or _state.inference
    model_messages, history_messages = _prepare_messages(messages, context)

    attempt_reasons = []
    raw_response = ""
    outcome, payload_or_reason = None, None
    parsed_response = None
    attempts = 0

    for attempt in range(1, inference.max_attempts + 1):
        attempts = attempt
        raw_response = ""
        for delta in _generate_response(model_messages, inference):
            raw_response += delta
            yield {"type": "token", "text": delta}

        parsed_response = _parse_json_response(raw_response)
        is_valid, rejection_reason = _assess_response(parsed_response, context)

        if is_valid:
            if "clarification" in parsed_response:
                outcome = "clarification"
                payload_or_reason = parsed_response["clarification"]
            else:
                outcome = "plan"
                payload_or_reason = parsed_response
            break

        attempt_reasons.append(rejection_reason)
        if attempt < inference.max_attempts:
            model_messages = model_messages + [
                {"role": "assistant", "content": raw_response},
                {"role": "user", "content": _correction_message(rejection_reason)},
            ]

    if outcome is None:
        payload_or_reason = attempt_reasons[-1]

    schema_valid = isinstance(parsed_response, dict)
    if schema_valid:
        try:
            if set(parsed_response) == {"clarification"}:
                parse_clarification(parsed_response)
            else:
                parse_plan(parsed_response)
        except (KeyError, TypeError, ValueError):
            schema_valid = False

    response_is_valid = outcome is not None
    metrics = PlannerMetrics(
        json_valid=isinstance(parsed_response, dict),
        schema_valid=schema_valid,
        guard_valid=response_is_valid,
        rejection_reason=None if response_is_valid else payload_or_reason,
        raw_response=raw_response,
        attempts=attempts,
        attempt_reasons=attempt_reasons,
    )

    history = history_messages
    if response_is_valid:
        history = history_messages + [{"role": "assistant", "content": raw_response}]

    yield {
        "type": "done",
        "outcome": outcome or "error",
        "payload": payload_or_reason,
        "history": history,
        "metrics": metrics,
    }
