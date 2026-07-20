import json
from pathlib import Path

from datasets import Dataset


def load_jsonl(path):
    """Load non-empty JSONL rows."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_direct_conversations(samples):
    """Validate direct conversation messages before formatting."""
    for sample_index, sample in enumerate(samples):
        messages = sample.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Sample {sample_index} has no conversation messages.")
        for message_index, message in enumerate(messages):
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Sample {sample_index}, message {message_index} has "
                    f"unsupported role {role!r}."
                )


def load_dataset(data_dir, tokenizer):
    """Load, validate, and render training and validation datasets."""
    directory = Path(data_dir)
    train_raw = load_jsonl(directory / "train.jsonl")
    val_raw = load_jsonl(directory / "val.jsonl")
    validate_direct_conversations(train_raw + val_raw)

    def render(sample):
        return {
            "text": tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        }

    print(f">>> Formatting {len(train_raw)} train + {len(val_raw)} val samples ...")
    train_dataset = Dataset.from_list(train_raw).map(render, desc="Formatting train")
    val_dataset = Dataset.from_list(val_raw).map(render, desc="Formatting val")
    print(f">>> Done: {len(train_dataset)} train | {len(val_dataset)} val")
    return train_dataset, val_dataset
