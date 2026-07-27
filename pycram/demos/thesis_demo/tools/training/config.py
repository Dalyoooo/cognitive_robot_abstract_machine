import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from datasets import Dataset
from unsloth.chat_templates import get_chat_template

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

SUPPORTED_ROLES = frozenset({"system", "user", "assistant"})


class PeftStrategy(Enum):
    """Selects which layers get LoRA adapters for a model family."""

    TEXT_MODULES = "text_modules"
    MULTIMODAL_LAYERS = "multimodal_layers"


@dataclass
class ModelFamilyConfig:
    match_keys: object
    template: object
    instruction_part: object
    response_part: object
    peft_strategy: object
    load_in_4bit: object = True

    def matches(self, model_name):
        lowered = model_name.lower()
        return any(key in lowered for key in self.match_keys)


FAMILY_CONFIGS = (
    ModelFamilyConfig(
        match_keys=("llama-3",),
        template="llama-3.1",
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        peft_strategy=PeftStrategy.TEXT_MODULES,
    ),
    ModelFamilyConfig(
        match_keys=("qwen",),
        template=None,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        peft_strategy=PeftStrategy.TEXT_MODULES,
    ),
    ModelFamilyConfig(
        match_keys=("gemma-4",),
        template="gemma-4",
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
        peft_strategy=PeftStrategy.MULTIMODAL_LAYERS,
    ),
)

FALLBACK_FAMILY = FAMILY_CONFIGS[0]

EXAMPLE_MODELS = (
    "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/Qwen3.5-4B",
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
)


def parse_args():
    """Parse fine-tuning command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune the grounded planner with Unsloth and LoRA."
    )
    parser.add_argument(
        "--model",
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="Base model for Unsloth. Examples: " + ", ".join(EXAMPLE_MODELS),
    )
    parser.add_argument(
        "--data-dir",
        default="ft-dataset",
        help="Directory with train.jsonl and val.jsonl",
    )
    parser.add_argument("--out-dir", default="models/pycram-planner")
    parser.add_argument("--merge", action="store_true", help="Save merged 16-bit model")
    parser.add_argument("--gguf", action="store_true", help="Export GGUF")
    parser.add_argument(
        "--gguf-method",
        default="q4_k_m",
        choices=[
            "q4_k_m",
            "q5_k_m",
            "q8_0",
            "q3_k_m",
            "q2_k",
            "f16",
            "q4_0",
            "q6_k",
            "q5_0",
            "q4_k_s",
            "q5_k_s",
        ],
    )
    parser.add_argument("--max-shard-size", default="20GB")
    parser.add_argument("--template", help="Override the tokenizer chat template")
    parser.add_argument("--push-to-hub", metavar="REPO_ID")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=25,
        help="Evaluate validation loss every N steps (0 = once per epoch)",
    )
    parser.add_argument("--save-steps", type=int, default=0)
    return parser.parse_args()


def detect_config(model_name):
    """Return the response markers and LoRA strategy for a model family."""
    for config in FAMILY_CONFIGS:
        if config.matches(model_name):
            return config
    print(
        f">>> WARNING: Unknown architecture for {model_name}; "
        "defaulting to llama-3 template."
    )
    return FALLBACK_FAMILY


def text_tokenizer(tokenizer):
    if hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    return tokenizer


def _token_is_known(tokenizer, token):
    if not token:
        return False
    lookup = text_tokenizer(tokenizer)
    if hasattr(lookup, "convert_tokens_to_ids"):
        token_id = lookup.convert_tokens_to_ids(token)
        return token_id is not None and token_id != lookup.unk_token_id
    if hasattr(lookup, "get_vocab"):
        return token in lookup.get_vocab()
    # No way to verify; trust the token Unsloth configured.
    return True


def configure_chat_template(tokenizer, template=None):
    """Configure and validate tokenizer chat, EOS, and padding settings."""
    original_eos = tokenizer.eos_token
    if template:
        tokenizer = get_chat_template(tokenizer, chat_template=template)

    eos = tokenizer.eos_token
    if not _token_is_known(tokenizer, eos):
        fallback = next(
            (
                token
                for token in (original_eos, "<|im_end|>", "<|endoftext|>")
                if _token_is_known(tokenizer, token)
            ),
            None,
        )
        if fallback is None:
            raise ValueError(
                f"Tokenizer EOS token {eos!r} is not a known token and no "
                "known fallback token is available."
            )
        tokenizer.eos_token = fallback
        eos = fallback

    if not tokenizer.chat_template:
        raise ValueError(
            "Tokenizer has no built-in chat template. Pass --template with a "
            "template supported by the installed Unsloth version."
        )
    if "<EOS_TOKEN>" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace("<EOS_TOKEN>", eos)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = eos
    return tokenizer


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
            if role not in SUPPORTED_ROLES:
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
