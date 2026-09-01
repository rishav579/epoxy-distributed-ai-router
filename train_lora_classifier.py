"""Fine-tune a Hugging Face sequence classifier with LoRA from CSV files.

Expected columns: `text` and `label` (integer class IDs).  Install: torch,
transformers, peft, wandb.  Example:
python train_lora_classifier.py --train-csv train.csv --val-csv val.csv --num-labels 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeAlias

import torch
import wandb
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase


Example: TypeAlias = tuple[str, int]
Batch: TypeAlias = dict[str, Tensor]


@dataclass(frozen=True)
class Config:
    train_csv: Path
    val_csv: Path
    output_dir: Path
    model_name: str
    num_labels: int
    text_column: str
    label_column: str
    batch_size: int
    epochs: int
    learning_rate: float
    max_length: int
    patience: int
    lora_rank: int
    seed: int
    wandb_project: str
    wandb_run_name: str | None


class CsvClassificationDataset(Dataset[Example]):
    def __init__(self, path: Path, text_column: str, label_column: str, num_labels: int) -> None:
        # `utf-8-sig` accepts ordinary UTF-8 and strips a BOM added by Windows PowerShell.
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or text_column not in rows[0] or label_column not in rows[0]:
            raise ValueError(f"{path} must contain non-empty '{text_column}' and '{label_column}' columns")
        try:
            self.examples: list[Example] = [(row[text_column], int(row[label_column])) for row in rows]
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid label in {path}; labels must be integer class IDs") from error
        invalid_labels = sorted({label for _, label in self.examples if not 0 <= label < num_labels})
        if invalid_labels:
            raise ValueError(
                f"Labels in {path} must be in [0, {num_labels - 1}]; found {invalid_labels}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collator(tokenizer: PreTrainedTokenizerBase, max_length: int) -> Callable[[Sequence[Example]], Batch]:
    def collate(examples: Sequence[Example]) -> Batch:
        texts, labels = zip(*examples, strict=True)
        encoded = tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return dict(encoded)

    return collate


def move_to_device(batch: Batch, device: torch.device) -> Batch:
    return {name: tensor.to(device, non_blocking=device.type == "cuda") for name, tensor in batch.items()}


def release_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def is_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def find_lora_target_modules(model: nn.Module) -> list[str]:
    """Return the query/value projection names used by common encoder models.

    PEFT accepts leaf module names, and applies them to every matching layer.
    Keeping this discovery small and explicit avoids silently adapting an
    unintended module for an unsupported architecture.
    """
    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    for targets in (("q_lin", "v_lin"), ("query", "value"), ("q_proj", "v_proj")):
        if set(targets).issubset(leaf_names):
            return list(targets)
    raise ValueError(
        "Could not identify query/value projections for this model. "
        "Use an encoder with q_lin/v_lin, query/value, or q_proj/v_proj layers, "
        "or extend find_lora_target_modules()."
    )


def classifier_modules_to_save(model: nn.Module) -> list[str] | None:
    """Keep DistilBERT's extra classification projection trainable.

    For sequence-classification tasks PEFT itself saves `classifier` (or
    `score`).  DistilBERT also has `pre_classifier`; naming it here lets PEFT
    save that layer without requiring every backbone to expose it.
    """
    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    return ["pre_classifier"] if "pre_classifier" in leaf_names else None


def evaluate(model: nn.Module, loader: DataLoader[Batch], device: torch.device) -> float:
    model.eval()
    total_loss, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            outputs = model(**move_to_device(batch, device))
            if outputs.loss is None:
                raise RuntimeError("Model did not return a loss; labels are required.")
            total_loss += outputs.loss.item()
            count += 1
    if count == 0:
        raise ValueError("Validation DataLoader is empty")
    return total_loss / count


def save_checkpoint(
    model: PeftModel,
    optimizer: Optimizer,
    epoch: int,
    val_loss: float,
    output_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_checkpoint.pt"
    temporary_checkpoint = checkpoint.with_suffix(".tmp")
    torch.save(
        {"epoch": epoch, "val_loss": val_loss, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")


def train(config: Config) -> None:
    validate_config(config)
    seed_everything(config.seed)
    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=find_lora_target_modules(base_model),
        modules_to_save=classifier_modules_to_save(base_model),
    )
    model = get_peft_model(base_model, lora_config).to(device)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate)

    collate = make_collator(tokenizer, config.max_length)
    train_loader = DataLoader(
        CsvClassificationDataset(config.train_csv, config.text_column, config.label_column, config.num_labels),
        batch_size=config.batch_size, shuffle=True, collate_fn=collate, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CsvClassificationDataset(config.val_csv, config.text_column, config.label_column, config.num_labels),
        batch_size=config.batch_size, shuffle=False, collate_fn=collate, pin_memory=device.type == "cuda",
    )

    run = wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))
    best_val_loss, stalled_epochs, global_step = float("inf"), 0, 0
    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss, processed = 0.0, 0
            for batch in train_loader:
                try:
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(**move_to_device(batch, device))
                    if outputs.loss is None:
                        raise RuntimeError("Model did not return a loss; labels are required.")
                    outputs.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                except RuntimeError as error:
                    if not is_oom(error):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    release_memory(device)
                    wandb.log({"oom_batches": 1, "step": global_step})
                    continue
                running_loss += outputs.loss.item()
                processed += 1
                global_step += 1
                wandb.log({"train/loss": outputs.loss.item(), "step": global_step})

            if processed == 0:
                raise RuntimeError("Every training batch OOMed; lower --batch-size or --max-length.")
            val_loss = evaluate(model, val_loader, device)
            train_loss = running_loss / processed
            wandb.log({"epoch": epoch, "train/epoch_loss": train_loss, "val/loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss, stalled_epochs = val_loss, 0
                save_checkpoint(model, optimizer, epoch, val_loss, config.output_dir, tokenizer)
            else:
                stalled_epochs += 1
                if stalled_epochs >= config.patience:
                    break
    finally:
        run.finish()


def validate_config(config: Config) -> None:
    if config.num_labels < 2:
        raise ValueError("--num-labels must be at least 2")
    for name, value in (
        ("--batch-size", config.batch_size),
        ("--epochs", config.epochs),
        ("--max-length", config.max_length),
        ("--patience", config.patience),
        ("--lora-rank", config.lora_rank),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero")
    for path, option in ((config.train_csv, "--train-csv"), (config.val_csv, "--val-csv")):
        if not path.is_file():
            raise FileNotFoundError(f"{option} does not exist or is not a file: {path}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--num-labels", type=int, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="lora-classifier")
    parser.add_argument("--wandb-run-name")
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
