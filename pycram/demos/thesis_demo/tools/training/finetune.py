import gc
import sys
from pathlib import Path

_DEMOS_DIR = Path(__file__).resolve().parents[3]
if str(_DEMOS_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMOS_DIR))

import torch
import unsloth
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

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
    text_tokenizer,
)


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
            optim="adamw_8bit",
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
    # A multimodal processor routes a positional argument to its image channel,
    # so tokenize the space through the underlying text tokenizer.
    text_only = text_tokenizer(tokenizer)
    space_id = text_only(" ", add_special_tokens=False).input_ids[0]
    sample_labels = trainer.train_dataset[0]["labels"]
    supervised_tokens = sum(label != -100 for label in sample_labels)
    if supervised_tokens == 0:
        raise RuntimeError(
            "Response-only masking removed every label. Check the chat template "
            "markers before training."
        )
    visible_labels = [space_id if label == -100 else label for label in sample_labels]
    print(f">>> Response-only check: {supervised_tokens} supervised tokens in sample 0")
    print(text_only.decode(visible_labels[:300]))


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
