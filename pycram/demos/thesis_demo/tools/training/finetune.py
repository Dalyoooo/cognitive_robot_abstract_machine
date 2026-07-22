import gc
import json
import sys
from pathlib import Path

_DEMOS_DIR = Path(__file__).resolve().parents[3]
if str(_DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMOS_DIR))

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import unsloth
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

from thesis_demo.planner.prompt import system_prompt, user_turn
from thesis_demo.tools.training.artifacts import (
    push_to_hub,
    save_gguf,
    save_merged,
    save_training_report,
)
from thesis_demo.tools.training.config import (
    DEFAULT_TARGET_MODULES,
    PeftStrategy,
    configure_chat_template,
    detect_config,
    load_dataset,
    parse_args,
    validate_direct_conversations as _validate_direct_conversations,
)
from thesis_demo.validation.guard import verify


class SanityCheckError(RuntimeError):
    """Expected failure of a learned-behavior check."""


SANITY_CASES = (
    {
        "name": "Fridge source",
        "instruction": "bring the milk from the fridge to the table",
        "context": {
            "objects": ["milk"],
            "object_locations": {"milk": ["fridge"]},
            "surfaces": ["table"],
            "containers": ["fridge"],
            "openables": ["fridge"],
            "furniture": [],
            "rooms": ["kitchen"],
            "types": {
                "milk": "Milk",
                "table": "Table",
                "fridge": "Fridge",
                "kitchen": "Kitchen",
            },
        },
        "container_actions": {"OpenAction", "CloseAction"},
        "source": "fridge",
        "destination": "table",
    },
    {
        "name": "Fridge source rephrase",
        "instruction": "fetch the milk from the fridge and place it on the table",
        "context": {
            "objects": ["milk"],
            "object_locations": {"milk": ["fridge"]},
            "surfaces": ["table"],
            "containers": ["fridge"],
            "openables": ["fridge"],
            "furniture": [],
            "rooms": ["kitchen"],
            "types": {
                "milk": "Milk",
                "table": "Table",
                "fridge": "Fridge",
                "kitchen": "Kitchen",
            },
        },
        "container_actions": {"OpenAction", "CloseAction"},
        "source": "fridge",
        "destination": "table",
    },
    {
        "name": "surface transport",
        "instruction": "move the milk from the counter to the table",
        "context": {
            "objects": ["milk"],
            "object_locations": {"milk": ["counter"]},
            "surfaces": ["counter", "table"],
            "containers": [],
            "openables": [],
            "furniture": [],
            "rooms": ["kitchen"],
            "types": {
                "milk": "Milk",
                "counter": "CounterTop",
                "table": "Table",
                "kitchen": "Kitchen",
            },
        },
        "container_actions": set(),
        "source": "counter",
        "destination": "table",
    },
)


def generate(model, tokenizer, messages, max_new_tokens=2048):
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    attention_mask = tokens.new_ones(tokens.shape)
    output = model.generate(
        tokens,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(
        output[0][tokens.shape[1] :], skip_special_tokens=True
    ).strip()


def _generate(model, tokenizer, messages, max_new_tokens=2048):
    return generate(model, tokenizer, messages, max_new_tokens)


def _sanity_transport(plan, case):
    transports = [
        step for step in plan["plan"] if step.get("action") == "TransportAction"
    ]
    matches = (
        transports[0].get("object"),
        transports[0].get("source"),
        transports[0].get("location"),
    ) == ("milk", case["source"], case["destination"])
    if len(transports) != 1 or not matches:
        raise SanityCheckError(
            f"{case['name']} failed: expected one matching TransportAction."
        )


def _sanity_container_actions(plan, case):
    container_actions = {
        step["action"]
        for step in plan["plan"]
        if step["action"] in {"OpenAction", "CloseAction"}
    }
    if container_actions != case["container_actions"]:
        raise SanityCheckError(
            f"{case['name']} failed: expected container actions "
            f"{sorted(case['container_actions'])}, got "
            f"{sorted(container_actions)}."
        )


def _check_sanity_case(model, tokenizer, generate_response, case):
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_turn(case["instruction"], case["context"])},
    ]
    text = generate_response(model, tokenizer, messages)
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as error:
        raise SanityCheckError(
            f"{case['name']} failed: response is not JSON: {text[:200]!r}."
        ) from error
    valid, reason = verify(plan, case["context"])
    if not valid or not plan.get("plan"):
        raise SanityCheckError(f"{case['name']} failed: invalid plan: {reason}.")
    _sanity_transport(plan, case)
    _sanity_container_actions(plan, case)


def run_sanity_check(model, tokenizer, generate_response, fast_model):
    fast_model.for_inference(model)
    for case in SANITY_CASES:
        _check_sanity_case(model, tokenizer, generate_response, case)
    print(">>> Sanity check passed: all 3 planner behavior cases are valid.")


def _run_sanity_check(model, tokenizer):
    return run_sanity_check(model, tokenizer, _generate, unsloth.FastModel)


def _run_sanity_check_nonfatal(model, tokenizer):
    try:
        _run_sanity_check(model, tokenizer)
    except SanityCheckError as error:
        print(f">>> WARNING: {error}")
        print(">>> WARNING: Continuing exports after failed sanity check.")
        return False
    return True


def _build_trainer(args, model, tokenizer, train_dataset, val_dataset, output_dir):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    estimated_steps = (
        args.epochs * len(train_dataset) / (args.batch_size * args.grad_accum)
    )
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else max(1, round(0.03 * estimated_steps))
    )
    print(
        f">>> Precision: {'bf16' if use_bf16 else 'fp16'} | "
        f"Batch: {args.batch_size} x {args.grad_accum} accum | "
        f"Epochs: {args.epochs} | LR: {args.lr} | "
        f"LoRA r={args.lora_r} alpha={args.lora_alpha} | "
        f"Warmup: {warmup_steps} steps"
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            eos_token=tokenizer.eos_token,
            dataset_text_field="text",
            dataset_num_proc=2,
            max_length=args.max_seq_len,
            packing=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=0.3,
            bf16=use_bf16,
            fp16=not use_bf16,
            eval_strategy="steps" if args.eval_steps > 0 else "epoch",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "epoch",
            save_steps=args.save_steps if args.save_steps > 0 else None,
            load_best_model_at_end=False,
            save_total_limit=args.epochs,
            logging_steps=10,
            seed=args.seed,
            report_to="none",
            dataloader_num_workers=2,
        ),
    )
    return unsloth.train_on_responses_only(
        trainer,
        instruction_part=args.instruction_part,
        response_part=args.response_part,
    )


def _check_response_mask(trainer, tokenizer):
    space_id = tokenizer(" ", add_special_tokens=False).input_ids[0]
    sample_labels = trainer.train_dataset[0]["labels"]
    supervised_tokens = sum(label != -100 for label in sample_labels)
    if supervised_tokens == 0:
        raise RuntimeError(
            "Response-only masking removed every label. Check the chat template "
            "markers before training."
        )
    visible_labels = [space_id if label == -100 else label for label in sample_labels]
    print(f">>> Response-only check: {supervised_tokens} supervised tokens in sample 0")
    print(tokenizer.decode(visible_labels[:300]))


def _apply_peft(model, args, peft_strategy):
    if peft_strategy is PeftStrategy.MULTIMODAL_LAYERS:
        return unsloth.FastModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=args.seed,
            use_gradient_checkpointing="unsloth",
            use_rslora=False,
            loftq_config=None,
        )
    return unsloth.FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=list(DEFAULT_TARGET_MODULES),
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        loftq_config=None,
    )


def _load_model(args, template, config):
    """Load the base model and attach its LoRA adapter."""
    load_in_4bit = config.load_in_4bit
    print(f">>> Loading model: {args.model}")
    model, tokenizer = unsloth.FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=load_in_4bit,
        load_in_16bit=not load_in_4bit,
    )
    model.generation_config.max_length = None
    model = _apply_peft(model, args, config.peft_strategy)
    tokenizer = configure_chat_template(tokenizer, template)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    """Run fine-tuning and requested model exports."""
    args = parse_args()

    config = detect_config(args.model)
    template = args.template or config.template
    args.instruction_part = config.instruction_part
    args.response_part = config.response_part

    model, tokenizer = _load_model(args, template, config)
    train_dataset, val_dataset = load_dataset(args.data_dir, tokenizer)
    print(">>> Sample formatted text:")
    print(train_dataset[0]["text"])

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = _build_trainer(
        args, model, tokenizer, train_dataset, val_dataset, output_dir
    )
    _check_response_mask(trainer, tokenizer)

    gpu = torch.cuda.get_device_properties(0)
    start_memory = torch.cuda.max_memory_reserved() / 1024**3
    max_memory = gpu.total_memory / 1024**3
    print(
        f">>> GPU: {gpu.name}; reserved={start_memory:.3f} GB; "
        f"total={max_memory:.3f} GB"
    )

    print(">>> Training ...")
    result = trainer.train()
    runtime = result.metrics.get("train_runtime", 0)
    print(f">>> Training finished in {runtime:.0f}s ({runtime / 60:.1f} min)")
    save_training_report(trainer, output_dir)

    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f">>> Adapter saved: {adapter_path}")

    print(">>> Running minimal learned-behavior check ...")
    _run_sanity_check_nonfatal(model, tokenizer)

    merged_path = output_dir / "merged"
    merge_ok = args.merge and save_merged(
        model,
        tokenizer,
        adapter_path,
        merged_path,
        args.model,
        args.max_shard_size,
    )
    gguf_path = output_dir / "gguf"
    gguf_ok = args.gguf and save_gguf(
        model,
        tokenizer,
        gguf_path,
        args.gguf_method,
    )
    if args.merge and not merge_ok:
        raise RuntimeError(
            "Merged export was requested but failed. The adapter remains available."
        )
    if args.gguf and not gguf_ok:
        raise RuntimeError(
            "GGUF export was requested but failed. Hub upload was not started."
        )
    if args.push_to_hub:
        push_to_hub(
            args.push_to_hub,
            adapter_path,
            merged_path,
            merge_ok,
            gguf_path,
            gguf_ok,
        )

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print(">>> Done.")


if __name__ == "__main__":
    main()
