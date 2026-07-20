import argparse

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

MODEL_CONFIGS = {
    "qwen": {
        "template": None,
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "load_in_4bit": True,
        "target_modules": list(DEFAULT_TARGET_MODULES),
    },
    "gemma-4": {
        "template": None,
        "instruction_part": "<|turn>user\n",
        "response_part": "<|turn>model\n",
        "load_in_4bit": True,
        "target_modules": list(DEFAULT_TARGET_MODULES),
    },
    "llama-3": {
        "template": "llama-3.1",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "load_in_4bit": True,
        "target_modules": list(DEFAULT_TARGET_MODULES),
    },
}


def parse_args():
    """Parse fine-tuning command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune the grounded planner with Unsloth and LoRA."
    )
    parser.add_argument(
        "--model",
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="Base model for Unsloth",
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
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Evaluate validation loss every N steps (0 = once per epoch)",
    )
    parser.add_argument("--save-steps", type=int, default=0)
    return parser.parse_args()


def detect_config(model_name):
    """Return the response markers for a supported model family."""
    name_lower = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if key in name_lower:
            return config
    print(
        f">>> WARNING: Unknown architecture for {model_name}; "
        "defaulting to llama-3 template."
    )
    return MODEL_CONFIGS["llama-3"]


def configure_chat_template(tokenizer, template=None):
    """Configure and validate tokenizer chat, EOS, and padding settings."""
    original_eos = tokenizer.eos_token
    if template:
        tokenizer = get_chat_template(tokenizer, chat_template=template)

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    eos = tokenizer.eos_token
    if eos not in vocab:
        fallback = next(
            (
                token
                for token in (original_eos, "<|im_end|>", "<|endoftext|>")
                if token and token in vocab
            ),
            None,
        )
        if fallback is None:
            raise ValueError(
                f"Tokenizer EOS token {eos!r} is not in its vocabulary and no "
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
