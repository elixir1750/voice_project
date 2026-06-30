from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from config import save_config
from data.dataset import LibriSpeechDataModule
from models.interfaces import DecoderOutput
from models.registry import build_decoder, build_representation, build_ssl
from utils.checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from utils.device import resolve_device, supports_amp
from utils.metrics import compute_error_rates, compute_rtf
from utils.tokenizer import CharacterCTCTokenizer


class ASRModel(nn.Module):
    def __init__(
        self,
        ssl: nn.Module,
        representation: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.ssl = ssl
        self.representation = representation
        self.decoder = decoder

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> DecoderOutput:
        speech = self.ssl(waveforms, waveform_lengths)
        represented = self.representation(speech)
        return self.decoder(represented.values, represented.lengths)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_from_config(
    config: Mapping[str, Any],
    tokenizer: CharacterCTCTokenizer,
) -> ASRModel:
    ssl_config = dict(config["ssl"])
    cache_dir = config.get("runtime", {}).get("cache_dir")
    if cache_dir and "cache_dir" not in ssl_config:
        ssl_config["cache_dir"] = cache_dir
    ssl = build_ssl(ssl_config)
    representation = build_representation(
        config["representation"],
        input_dim=ssl.output_dim,
    )
    decoder = build_decoder(
        config["decoder"],
        input_dim=representation.output_dim,
        vocab_size=len(tokenizer),
    )
    return ASRModel(ssl=ssl, representation=representation, decoder=decoder)


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    decoder = getattr(model, "decoder", None)
    decoder_parameters = list(decoder.parameters()) if decoder is not None else []
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "decoder": sum(parameter.numel() for parameter in decoder_parameters),
        "decoder_trainable": sum(
            parameter.numel()
            for parameter in decoder_parameters
            if parameter.requires_grad
        ),
    }


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["waveforms"].to(device),
        batch["waveform_lengths"].to(device),
        batch["targets"].to(device),
        batch["target_lengths"].to(dtype=torch.long),
    )


def ctc_compute_device(model_device: torch.device) -> torch.device:
    if model_device.type == "mps":
        return torch.device("cpu")
    return model_device


def should_stop_early(
    epochs_without_improvement: int,
    patience: int,
) -> bool:
    return patience > 0 and epochs_without_improvement >= patience


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: Mapping[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_type = str(training_config.get("scheduler", "none")).lower()
    if scheduler_type in {"none", "off", "disabled"}:
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=float(training_config.get("min_learning_rate", 0.0)),
        )
    raise ValueError(f"Unknown scheduler: {scheduler_type}")


def _ctc_loss(
    output: DecoderOutput,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> torch.Tensor:
    output_lengths = output.lengths.to(dtype=torch.long)
    cpu_output_lengths = output_lengths.detach().cpu()
    cpu_target_lengths = target_lengths.detach().cpu()
    if torch.any(cpu_target_lengths > cpu_output_lengths):
        raise ValueError(
            "CTC target length exceeds available output frames; "
            "use longer audio or shorter normalized transcripts"
        )
    compute_device = ctc_compute_device(output.logits.device)
    log_probs = (
        output.logits.log_softmax(dim=-1)
        .transpose(0, 1)
        .to(compute_device)
    )
    return F.ctc_loss(
        log_probs,
        targets.to(compute_device),
        cpu_output_lengths,
        cpu_target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    tokenizer: CharacterCTCTokenizer,
    device: torch.device,
    grad_clip: float = 1.0,
    amp: bool = False,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float | int]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train()
    use_amp = bool(amp and supports_amp(device))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = 0.0
    batches = 0
    optimizer_steps = 0
    batches_since_step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(dataloader, desc="train", leave=False):
        waveforms, waveform_lengths, targets, target_lengths = _move_batch(
            batch,
            device,
        )
        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            output = model(waveforms, waveform_lengths)
            loss = _ctc_loss(output, targets, target_lengths, tokenizer.blank_id)
        scaler.scale(loss / gradient_accumulation_steps).backward()

        total_loss += float(loss.detach().cpu())
        batches += 1
        batches_since_step += 1
        if batches_since_step == gradient_accumulation_steps:
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            batches_since_step = 0

    if batches == 0:
        raise RuntimeError("Training dataloader produced no batches")
    if batches_since_step:
        scaler.unscale_(optimizer)
        correction = gradient_accumulation_steps / batches_since_step
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    return {
        "loss": total_loss / batches,
        "batches": batches,
        "optimizer_steps": optimizer_steps,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    tokenizer: CharacterCTCTokenizer,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    batches = 0
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0
    references: list[str] = []
    hypotheses: list[str] = []

    for batch in tqdm(dataloader, desc="validate", leave=False):
        waveforms, waveform_lengths, targets, target_lengths = _move_batch(
            batch,
            device,
        )
        _synchronize(device)
        started_at = time.perf_counter()
        output = model(waveforms, waveform_lengths)
        _synchronize(device)
        total_inference_seconds += time.perf_counter() - started_at

        loss = _ctc_loss(output, targets, target_lengths, tokenizer.blank_id)
        predicted_ids = output.logits.argmax(dim=-1).detach().cpu()
        for row, length in zip(predicted_ids, output.lengths.detach().cpu()):
            hypotheses.append(tokenizer.decode_ctc(row[: int(length)].tolist()))
        references.extend(str(text) for text in batch["texts"])
        total_audio_seconds += float(batch["audio_seconds"].sum())
        total_loss += float(loss.detach().cpu())
        batches += 1

    if batches == 0 or not references:
        raise RuntimeError("Validation dataloader produced no samples")
    rates = compute_error_rates(references, hypotheses)
    return {
        "loss": total_loss / batches,
        **rates,
        "rtf": compute_rtf(total_audio_seconds, total_inference_seconds),
        "num_samples": len(references),
        "references": references,
        "hypotheses": hypotheses,
    }


def fit(
    config: Mapping[str, Any],
    resume: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(int(config["experiment"].get("seed", 42)))
    device = resolve_device(str(config["runtime"].get("device", "auto")))
    tokenizer = CharacterCTCTokenizer()
    model = build_model_from_config(config, tokenizer).to(device)
    parameter_counts = count_model_parameters(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Model has no trainable parameters")
    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    epochs = int(training_config.get("epochs", 1))
    scheduler = build_scheduler(optimizer, training_config, epochs)
    state = {
        "epoch": 0,
        "global_step": 0,
        "optimizer_step": 0,
        "best_wer": math.inf,
        "epochs_without_improvement": 0,
    }
    if resume is not None:
        payload = load_checkpoint(resume, map_location=device)
        restore_checkpoint(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        state.update(payload["training_state"])

    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    data_module = LibriSpeechDataModule(
        config=config["data"],
        tokenizer=tokenizer,
        cache_dir=config["runtime"].get("cache_dir"),
    )

    for epoch in range(int(state["epoch"]), epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=data_module.train_dataloader(training_config),
            optimizer=optimizer,
            tokenizer=tokenizer,
            device=device,
            grad_clip=float(training_config.get("grad_clip", 1.0)),
            amp=bool(training_config.get("amp", False)),
            gradient_accumulation_steps=int(
                training_config.get("gradient_accumulation_steps", 1)
            ),
        )
        validation_metrics = validate(
            model=model,
            dataloader=data_module.validation_dataloader(training_config),
            tokenizer=tokenizer,
            device=device,
        )
        state["epoch"] = epoch + 1
        state["global_step"] = int(state["global_step"]) + int(
            train_metrics["batches"]
        )
        state["optimizer_step"] = int(
            state.get("optimizer_step", 0)
        ) + int(
            train_metrics["optimizer_steps"]
        )
        current_wer = float(validation_metrics["wer"])
        improved = current_wer < float(state["best_wer"])
        if improved:
            state["best_wer"] = current_wer
            state["epochs_without_improvement"] = 0
        else:
            state["epochs_without_improvement"] = int(
                state.get("epochs_without_improvement", 0)
            ) + 1
        if scheduler is not None:
            scheduler.step()

        summary = {
            "epoch": epoch + 1,
            "device": str(device),
            "parameters": parameter_counts,
            "train": train_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": {
                key: value
                for key, value in validation_metrics.items()
                if key not in {"references", "hypotheses"}
            },
        }
        print(json.dumps(summary, ensure_ascii=False))
        (output_dir / f"epoch_{epoch + 1}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            training_state=state,
            scheduler=scheduler,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                training_state=state,
                scheduler=scheduler,
            )
        if should_stop_early(
            int(state["epochs_without_improvement"]),
            int(training_config.get("early_stopping_patience", 0)),
        ):
            break

    return {
        "device": str(device),
        "output_dir": str(output_dir),
        "training_state": state,
    }
