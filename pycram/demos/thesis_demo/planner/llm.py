import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from ..config import select_backend
from .prompt import system_prompt, user_instruction, user_turn
from ..validation.guard import verify
from ..validation.schema import parse_clarification, parse_plan

MAX_NEW_TOKENS = 2048
N_CTX = 8192


@dataclass
class _PlannerState:
    llm: object = field(default=None)
    error: str | None = field(default=None)
    last_run: dict = field(default_factory=dict)


_state = _PlannerState()


def setup_planner(
    model,
    gguf_file=None,
    n_gpu_layers=None,
    n_ctx=N_CTX,
):
    _state.llm = None
    _state.error = None
    _state.last_run = {}
    try:
        print(f"[planner] loading GGUF {model!r} ({gguf_file!r}) ...", flush=True)
        _load_gguf(model, gguf_file, n_gpu_layers, n_ctx)
        print(f"[planner] GGUF {model!r} ready", flush=True)
    except (ImportError, RuntimeError, OSError) as gguf_error:
        _state.error = f"GGUF load failed for {model!r}: {gguf_error!r}"
        print(f"[planner] {_state.error}", flush=True)
        raise


def load_planner():
    gguf_model = os.environ.get("PLANNER_GGUF_MODEL", "wijan/Robot-Action-Planner-gguf")
    gguf_file = os.environ.get(
        "PLANNER_GGUF_FILE", "meta-llama-3.1-8b-instruct.Q4_K_M.gguf"
    )
    try:
        setup_planner(gguf_model, gguf_file)
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"[planner] could not load model: {exc!r}", flush=True)


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


def get_last_run_metrics():
    return dict(_state.last_run)


def plan(
    transcript,
    conversation=None,
    context=None,
):
    messages = (conversation or []) + [{"role": "user", "content": transcript}]
    for event in _run_loop(messages, context or {}):
        if event["type"] == "done":
            return (
                event["outcome"],
                event["payload"],
                event["history"],
            )


def plan_stream(transcript, conversation=None, context=None):
    messages = (conversation or []) + [{"role": "user", "content": transcript}]
    return _run_loop(messages, context or {})


def _parse_json_response_with_mode(text):
    try:
        return json.loads(text.strip()), "exact"
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, "embedded_json"
    return None, "invalid"


def _generate_response(messages):
    stream = _state.llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        stream=True,
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


def _run_loop(messages, context):
    if _state.llm is None:
        raise RuntimeError("Planner not initialized. Call setup_planner() first.")

    model_messages, history_messages = _prepare_messages(messages, context)

    raw_response = ""
    for delta in _generate_response(model_messages):
        raw_response += delta
        yield {"type": "token", "text": delta}

    parsed_response, parse_mode = _parse_json_response_with_mode(raw_response)

    if parsed_response is None:
        outcome, payload_or_reason = None, "Output must be exactly one JSON object."
    else:
        is_valid, rejection_reason = verify(parsed_response, context)
        if not is_valid:
            outcome, payload_or_reason = None, rejection_reason
        elif "clarification" in parsed_response:
            outcome = "clarification"
            payload_or_reason = parsed_response["clarification"]
        else:
            outcome, payload_or_reason = "plan", parsed_response

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
    _state.last_run = {
        "json_valid": parse_mode == "exact" and isinstance(parsed_response, dict),
        "schema_valid": schema_valid,
        "guard_valid": response_is_valid,
        "rejection_reason": None if response_is_valid else payload_or_reason,
        "raw_response": raw_response,
    }

    history = history_messages
    if response_is_valid:
        history = history_messages + [{"role": "assistant", "content": raw_response}]

    yield {
        "type": "done",
        "outcome": outcome or "error",
        "payload": payload_or_reason,
        "history": history,
    }
