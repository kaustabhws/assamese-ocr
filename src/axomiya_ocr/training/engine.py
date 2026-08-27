from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from axomiya_ocr.data.dataset import HFDiskOCRDataset
from axomiya_ocr.data.image import CTCCollator
from axomiya_ocr.data.vocab import Vocabulary
from axomiya_ocr.models.recognizer import AssameseCRNN, RecognizerConfig, output_lengths
from axomiya_ocr.training.metrics import OCRMetrics, character_recall


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mixed_sampler(real_size: int, synthetic_size: int, synthetic_weight: float) -> WeightedRandomSampler:
    if not 0.0 <= synthetic_weight < 1.0:
        raise ValueError("synthetic_weight must be in [0, 1)")
    real_weight = (1.0 - synthetic_weight) / max(1, real_size)
    synth_weight = synthetic_weight / max(1, synthetic_size)
    weights = [real_weight] * real_size + [synth_weight] * synthetic_size
    return WeightedRandomSampler(weights, num_samples=real_size, replacement=True)


def build_loaders(config: dict[str, Any], vocab: Vocabulary) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    real_train = HFDiskOCRDataset(data_config["dataset_path"], "train")
    validation = HFDiskOCRDataset(data_config["dataset_path"], "validation")
    synthetic_path = Path(data_config["synthetic_path"])
    sampler = None
    train_dataset: Any = real_train
    if synthetic_path.exists() and data_config.get("synthetic_weight", 0) > 0:
        synthetic = HFDiskOCRDataset(synthetic_path, "train")
        train_dataset = ConcatDataset([real_train, synthetic])
        sampler = _mixed_sampler(
            len(real_train), len(synthetic), float(data_config["synthetic_weight"])
        )

    image_height = int(data_config["image_height"])
    min_image_width = int(data_config["min_image_width"])
    max_image_width = int(data_config["max_image_width"])
    batch_size = int(config["training"]["batch_size"])
    workers = min(int(data_config["num_workers"]), os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=CTCCollator(
            vocab=vocab,
            image_height=image_height,
            min_image_width=min_image_width,
            max_image_width=max_image_width,
            augment=True,
            seed=int(config["seed"]),
        ),
    )
    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=CTCCollator(
            vocab=vocab,
            image_height=image_height,
            min_image_width=min_image_width,
            max_image_width=max_image_width,
            augment=False,
            seed=int(config["seed"]),
        ),
    )
    return train_loader, validation_loader


def build_model(config: dict[str, Any], vocab: Vocabulary) -> AssameseCRNN:
    model_config = config["model"]
    return AssameseCRNN(
        RecognizerConfig(
            num_classes=vocab.size,
            input_channels=int(model_config["input_channels"]),
            cnn_channels=int(model_config["cnn_channels"]),
            rnn_hidden=int(model_config["rnn_hidden"]),
            rnn_layers=int(model_config["rnn_layers"]),
            dropout=float(model_config["dropout"]),
        )
    )


def _decode_batch(
    logits: torch.Tensor, input_lengths: torch.Tensor, vocab: Vocabulary
) -> list[str]:
    best = logits.argmax(dim=-1).detach().cpu()
    lengths = input_lengths.detach().cpu().tolist()
    return [
        vocab.decode_ctc(row[:length].tolist())
        for row, length in zip(best, lengths, strict=True)
    ]


@torch.inference_mode()
def evaluate(
    model: AssameseCRNN,
    loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    metrics = OCRMetrics()
    predictions: list[dict[str, Any]] = []
    references: list[str] = []
    hypotheses: list[str] = []
    for batch in tqdm(loader, desc="validation", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        logits = model(images)
        lengths = output_lengths(batch["input_widths"])
        decoded = _decode_batch(logits, lengths, vocab)
        for reference, hypothesis in zip(batch["texts"], decoded, strict=True):
            metrics.update(reference, hypothesis)
            references.append(reference)
            hypotheses.append(hypothesis)
            if len(predictions) < 200:
                predictions.append({"reference": reference, "hypothesis": hypothesis})
    result: dict[str, Any] = metrics.to_dict()
    result["assamese_specific_recall"] = {
        char: character_recall(references, hypotheses, char) for char in ("ৰ", "ৱ")
    }
    return result, predictions


def _save_checkpoint(
    path: Path,
    model: AssameseCRNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    vocab: Vocabulary,
    config: dict[str, Any],
    epoch: int,
    best_cer: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "best_cer": best_cer,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": asdict(model.config),
            "vocab": vocab.to_dict(),
            "training_config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        temporary,
    )
    temporary.replace(path)


def train(config: dict[str, Any], resume_from: str | Path | None = None) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = _device()
    dataset_path = Path(config["data"]["dataset_path"])
    audit_path = dataset_path / "audit.json"
    if audit_path.exists():
        config["data_provenance"] = json.loads(audit_path.read_text(encoding="utf-8"))
    synthetic_path = Path(config["data"]["synthetic_path"])
    synthetic_metadata = synthetic_path / "generation.json"
    if synthetic_metadata.exists():
        config["synthetic_provenance"] = json.loads(
            synthetic_metadata.read_text(encoding="utf-8")
        )
    vocab = Vocabulary.load(config["data"]["vocab_path"])
    train_loader, validation_loader = build_loaders(config, vocab)
    model = build_model(config, vocab).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training["epochs"]))
    )
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_cer = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    resume_path = Path(resume_from) if resume_from else None
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint["vocab"].get("sha256") != vocab.sha256:
            raise ValueError("Cannot resume with a different vocabulary")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_cer = float(checkpoint.get("best_cer", math.inf))
        start_epoch = int(checkpoint["epoch"]) + 1
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"].cpu())
            if torch.cuda.is_available() and rng_state.get("cuda"):
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            input_lengths = output_lengths(batch["input_widths"]).to(device)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            if step % int(training["log_every"]) == 0:
                progress.set_postfix(loss=f"{running_loss / step:.4f}")
        scheduler.step()

        validation_metrics, sample_predictions = evaluate(
            model, validation_loader, vocab, device
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **validation_metrics,
        }
        history.append(epoch_record)
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "validation_samples.json").write_text(
            json.dumps(sample_predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        improved = float(validation_metrics["cer"]) < best_cer
        if improved:
            best_cer = float(validation_metrics["cer"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        _save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            vocab,
            config,
            epoch,
            best_cer,
        )
        if improved:
            _save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                vocab,
                config,
                epoch,
                best_cer,
            )
        if epochs_without_improvement >= int(training["patience"]):
            break

    return {
        "device": str(device),
        "parameters": model.parameter_count(),
        "best_cer": best_cer,
        "epochs_completed": len(history),
        "checkpoint": str(output_dir / "best.pt"),
    }


def model_from_checkpoint(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[AssameseCRNN, Vocabulary, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_payload = checkpoint["vocab"]
    vocab = Vocabulary(tuple(vocab_payload["characters"]))
    if vocab_payload.get("sha256") != vocab.sha256:
        raise ValueError("Checkpoint vocabulary checksum mismatch")
    model = AssameseCRNN(RecognizerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, vocab, checkpoint
