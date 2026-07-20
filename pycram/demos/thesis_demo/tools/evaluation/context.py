import json
import os
from pathlib import Path

DEFAULT_RUN_DIR = "~/nlp-binder-run"


def run_dir():
    """Return the run directory holding the demo's JSON logs."""
    path = Path(os.environ.get("NLP_RUN_DIR", DEFAULT_RUN_DIR)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def context_file():
    """Return the path of the logged planner world context."""
    return run_dir() / "world_context.json"


def load_context():
    """Load the planner world context that the demo logged to disk."""
    try:
        with open(context_file(), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load world context: {error}") from error
