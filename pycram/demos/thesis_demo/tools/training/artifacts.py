import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM


def save_merged(
    model,
    tokenizer,
    adapter_path,
    merged_path,
    model_name,
    max_shard_size,
):
    merged_path.mkdir(parents=True, exist_ok=True)
    try:
        print(">>> Merging LoRA into base model (16-bit) ...")
        model.save_pretrained_merged(
            str(merged_path),
            tokenizer,
            save_method="merged_16bit",
            max_shard_size=max_shard_size,
            maximum_memory_usage=0.9,
        )
        print(f">>> Merged model saved: {merged_path}")
        return True
    except Exception as unsloth_error:
        print(
            f">>> WARNING: Unsloth merge failed ({unsloth_error}); "
            "trying manual PEFT merge ..."
        )

    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_path))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(
            str(merged_path),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        tokenizer.save_pretrained(str(merged_path))
        del base, peft_model, merged
        torch.cuda.empty_cache()
        print(f">>> Merged model saved: {merged_path}")
        return True
    except Exception as peft_error:
        print(
            f">>> ERROR: Merge failed ({peft_error}). "
            f"Adapter remains available at {adapter_path}."
        )
        return False


def save_gguf(model, tokenizer, gguf_path, gguf_method):
    gguf_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".gguf-export-",
        dir=gguf_path.parent,
    ) as temporary_directory:
        export_path = Path(temporary_directory) / "model"
        try:
            model.save_pretrained_gguf(
                str(export_path),
                tokenizer,
                quantization_method=gguf_method,
            )
        except Exception as error:
            print(f">>> ERROR: GGUF export failed: {error}")
            return False

        exported_files = sorted(Path(temporary_directory).rglob("*.gguf"))
        if not exported_files:
            print(">>> ERROR: GGUF export did not produce any .gguf files")
            return False
        for stale_file in gguf_path.glob("*.gguf"):
            stale_file.unlink()
        for source in exported_files:
            destination = gguf_path / source.name
            shutil.move(str(source), str(destination))

    print(f">>> GGUF saved: {gguf_path} ({len(exported_files)} file(s))")
    return True


def push_to_hub(repo_id, adapter_path, merged_path, merge_ok, gguf_path, gguf_ok):
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    artifact_path = merged_path if merge_ok else adapter_path
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(artifact_path),
        commit_message="Add fine-tuned planner",
    )
    if gguf_ok:
        gguf_repo = repo_id + "-gguf"
        gguf_files = sorted(gguf_path.glob("*.gguf"))
        if not gguf_files:
            raise RuntimeError("GGUF upload requested, but no local GGUF file exists.")

        api.create_repo(repo_id=gguf_repo, exist_ok=True)
        for gguf_file in gguf_files:
            api.upload_file(
                path_or_fileobj=str(gguf_file),
                path_in_repo=gguf_file.name,
                repo_id=gguf_repo,
                commit_message=f"Add {gguf_file.name}",
            )

        legacy_weights = [
            path
            for path in api.list_repo_files(repo_id=gguf_repo)
            if path.endswith(
                (
                    ".safetensors",
                    ".safetensors.index.json",
                    ".bin",
                    ".bin.index.json",
                )
            )
        ]
        for path in legacy_weights:
            api.delete_file(
                path_in_repo=path,
                repo_id=gguf_repo,
                commit_message=f"Remove non-GGUF weight {path}",
            )

        remote_files = set(api.list_repo_files(repo_id=gguf_repo))
        missing = [path.name for path in gguf_files if path.name not in remote_files]
        if missing:
            raise RuntimeError(
                f"GGUF upload verification failed for {gguf_repo}: {missing}"
            )
        print(f">>> GGUF pushed: {gguf_repo}")
    print(">>> Push complete.")


def _combine_legends(primary_axis, secondary_axis=None):
    handles, labels = primary_axis.get_legend_handles_labels()
    if secondary_axis is not None:
        secondary_handles, secondary_labels = secondary_axis.get_legend_handles_labels()
        handles += secondary_handles
        labels += secondary_labels
    if handles:
        primary_axis.legend(handles, labels, loc="center right")


def _history_points(history, column):
    """Return the recorded step/value pairs for one training metric."""
    columns = ["step", column]
    if not set(columns).issubset(history.columns):
        return history.reindex(columns=columns).iloc[:0]
    return history[columns].dropna().sort_values("step")


def _plot_loss_history(axis, train, validation):
    gap_axis = None
    if not train.empty:
        train["smoothed_loss"] = train["loss"].rolling(5, min_periods=1).mean()
        axis.plot(
            train["step"],
            train["loss"],
            color="tab:blue",
            alpha=0.25,
            label="Training loss",
        )
        axis.plot(
            train["step"],
            train["smoothed_loss"],
            color="tab:blue",
            label="Training loss (rolling mean)",
        )
    if not validation.empty:
        axis.plot(
            validation["step"],
            validation["eval_loss"],
            color="tab:orange",
            marker="o",
            label="Validation loss",
        )
        best = validation.loc[validation["eval_loss"].idxmin()]
        axis.annotate(
            f"best={best['eval_loss']:.4f}",
            (best["step"], best["eval_loss"]),
            xytext=(8, 8),
            textcoords="offset points",
        )
    if not train.empty and not validation.empty:
        aligned = pd.merge_asof(
            validation,
            train[["step", "smoothed_loss"]],
            on="step",
            direction="backward",
        ).dropna()
        if not aligned.empty:
            aligned["gap"] = aligned["eval_loss"] - aligned["smoothed_loss"]
            gap_axis = axis.twinx()
            gap_axis.plot(
                aligned["step"],
                aligned["gap"],
                color="tab:red",
                linestyle="--",
                marker="x",
                label="Generalization gap",
            )
            gap_axis.axhline(0, color="tab:red", linewidth=0.8, alpha=0.3)
            gap_axis.set_ylabel("Validation - training loss")

    axis.set_ylabel("Loss")
    axis.set_title("Training and validation")
    axis.grid(True, alpha=0.25)
    _combine_legends(axis, gap_axis)


def _plot_learning_rate_history(axis, learning_rate):
    if not learning_rate.empty:
        axis.plot(
            learning_rate["step"],
            learning_rate["learning_rate"],
            color="tab:green",
            label="Learning rate",
        )
        axis.set_ylabel("Learning rate")
    else:
        axis.text(
            0.5,
            0.5,
            "No learning-rate logs",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("Optimizer step")
    axis.set_title("Learning-rate schedule")
    axis.grid(True, alpha=0.25)
    _combine_legends(axis, None)


def save_training_report(trainer, output_dir):
    history = pd.DataFrame(trainer.state.log_history)
    if history.empty:
        return

    history_path = output_dir / "training_history.csv"
    history.to_csv(history_path, index=False)

    train = _history_points(history, "loss")
    validation = _history_points(history, "eval_loss")
    learning_rate = _history_points(history, "learning_rate")

    figure, (loss_axis, diagnostic_axis) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
    )
    _plot_loss_history(loss_axis, train, validation)
    _plot_learning_rate_history(diagnostic_axis, learning_rate)

    output_path = output_dir / "loss.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f">>> Loss plot saved: {output_path}")
    print(f">>> Training history saved: {history_path}")
