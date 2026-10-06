from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import build_mnist_loaders
from .model import CaffeLeNet, build_caffe_parameter_groups, count_parameters
from .schedules import InverseSchedule, OneCycleSchedule, SchedulePoint, apply_schedule


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "train_seconds",
    "eval_seconds",
    "epoch_seconds",
    "elapsed_seconds",
    "lr_start",
    "lr_end",
    "momentum_start",
    "momentum_end",
    "peak_gpu_memory_mib",
]

SCHEDULE_FIELDS = ["global_step", "epoch", "batch", "lr", "momentum", "phase"]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("В конфигурации выбрана CUDA, но torch.cuda.is_available() == False")
    return device


def collect_environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torchvision

        environment["torchvision"] = torchvision.__version__
    except Exception as error:  # pragma: no cover - только диагностика окружения
        environment["torchvision_error"] = repr(error)
    if device.type == "cuda":
        environment["gpu_name"] = torch.cuda.get_device_name(device)
        environment["gpu_memory_mib"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024**2, 1
        )
        environment["cudnn"] = torch.backends.cudnn.version()
    return environment


def build_optimizer(model: nn.Module, optimizer_config: dict[str, Any]) -> Optimizer:
    if optimizer_config.get("kind") != "sgd":
        raise ValueError("Для воспроизведения статьи поддерживается только SGD")
    if float(optimizer_config.get("caffe_bias_lr_multiplier", 2.0)) != 2.0:
        raise ValueError("Исходный LeNet Caffe использует множитель скорости 2 для смещений")

    groups = build_caffe_parameter_groups(
        model,
        base_lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    return SGD(groups, momentum=float(optimizer_config["momentum"]))


def build_schedule(
    scheduler_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    steps_per_epoch: int,
):
    kind = scheduler_config["kind"]
    if kind == "inverse":
        return InverseSchedule(
            base_lr=float(optimizer_config["lr"]),
            momentum=float(optimizer_config["momentum"]),
            gamma=float(scheduler_config["gamma"]),
            power=float(scheduler_config["power"]),
        )
    if kind == "onecycle":
        return OneCycleSchedule(
            up_steps=int(scheduler_config["up_epochs"]) * steps_per_epoch,
            down_steps=int(scheduler_config["down_epochs"]) * steps_per_epoch,
            final_steps=int(scheduler_config["final_epochs"]) * steps_per_epoch,
            min_lr=float(scheduler_config["min_lr"]),
            max_lr=float(scheduler_config["max_lr"]),
            final_lr=float(scheduler_config["final_lr"]),
            min_momentum=float(scheduler_config["min_momentum"]),
            max_momentum=float(scheduler_config["max_momentum"]),
        )
    raise ValueError(f"Неизвестное расписание: {kind!r}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    limit_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += targets.size(0)
    if total_examples == 0:
        raise RuntimeError("Проверочный загрузчик не выдал ни одного примера")
    return total_loss / total_examples, 100.0 * total_correct / total_examples


def _save_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    config: ExperimentConfig,
    *,
    epochs_override: int | None = None,
    seed_override: int | None = None,
    output_dir_override: str | Path | None = None,
    limit_train_batches: int | None = None,
    limit_test_batches: int | None = None,
) -> dict[str, Any]:
    raw = deepcopy(config.raw)
    data_config = raw["data"]
    optimizer_config = raw["optimizer"]
    scheduler_config = raw["scheduler"]
    training_config = raw["training"]

    epochs = int(epochs_override or training_config["epochs"])
    max_steps = int(training_config["max_steps"]) if training_config.get("max_steps") else None
    seed = int(config.seed if seed_override is None else seed_override)
    raw["seed"] = seed
    if scheduler_config["kind"] == "onecycle" and epochs != int(training_config["epochs"]):
        raise ValueError("Для 1cycle нельзя менять число эпох без изменения длительности его этапов")

    output_dir = (
        Path(output_dir_override).resolve()
        if output_dir_override is not None
        else config.resolve_path(training_config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    deterministic = bool(training_config.get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)
    device = resolve_device(str(training_config.get("device", "auto")))

    loaders = build_mnist_loaders(
        root=config.resolve_path(data_config["root"]),
        batch_size=int(data_config["batch_size"]),
        test_batch_size=int(data_config["test_batch_size"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]) and device.type == "cuda",
        download=bool(data_config["download"]),
        drop_last=bool(data_config["drop_last"]),
        seed=seed,
        caffe_pixel_scale=bool(data_config["caffe_pixel_scale"]),
    )
    actual_steps_per_epoch = len(loaders.train)
    steps_per_epoch = min(actual_steps_per_epoch, limit_train_batches or actual_steps_per_epoch)

    model = CaffeLeNet().to(device)
    optimizer = build_optimizer(model, optimizer_config)
    schedule = build_schedule(scheduler_config, optimizer_config, steps_per_epoch)
    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(training_config.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    environment = collect_environment(device)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    schedule_history: list[dict[str, Any]] = []
    global_step = 0
    total_start = time.perf_counter()
    log_every = int(training_config.get("log_every_steps", 25))

    print(f"Запуск: {config.name}")
    print(f"Устройство: {environment.get('gpu_name', device)}")
    print(f"Параметров модели: {count_parameters(model):,}")
    print(f"Шагов в эпохе: {steps_per_epoch}; эпох: {epochs}")

    for epoch_index in range(epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        epoch_start = time.perf_counter()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        first_point: SchedulePoint | None = None
        last_point: SchedulePoint | None = None

        for batch_index, (inputs, targets) in enumerate(loaders.train):
            if limit_train_batches is not None and batch_index >= limit_train_batches:
                break
            if max_steps is not None and global_step >= max_steps:
                break

            point = schedule(global_step)
            apply_schedule(optimizer, point)
            if first_point is None:
                first_point = point
            last_point = point
            schedule_history.append(
                {
                    "global_step": global_step,
                    "epoch": epoch_index + 1,
                    "batch": batch_index + 1,
                    "lr": point.lr,
                    "momentum": point.momentum,
                    "phase": point.phase,
                }
            )

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += targets.size(0)
            global_step += 1

            if log_every > 0 and (batch_index + 1) % log_every == 0:
                print(
                    f"  эпоха {epoch_index + 1:02d}/{epochs}, "
                    f"шаг {batch_index + 1:03d}/{steps_per_epoch}, "
                    f"потери {total_loss / total_examples:.4f}, "
                    f"LR {point.lr:.6g}"
                )

        if total_examples == 0 or first_point is None or last_point is None:
            raise RuntimeError("Обучающий загрузчик не выдал ни одного примера")

        _sync(device)
        train_seconds = time.perf_counter() - epoch_start
        eval_start = time.perf_counter()
        test_loss, test_accuracy = _evaluate(
            model, loaders.test, criterion, device, limit_batches=limit_test_batches
        )
        _sync(device)
        eval_seconds = time.perf_counter() - eval_start
        epoch_seconds = train_seconds + eval_seconds
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        )
        row = {
            "epoch": epoch_index + 1,
            "train_loss": total_loss / total_examples,
            "train_accuracy": 100.0 * total_correct / total_examples,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": time.perf_counter() - total_start,
            "lr_start": first_point.lr,
            "lr_end": last_point.lr,
            "momentum_start": first_point.momentum,
            "momentum_end": last_point.momentum,
            "peak_gpu_memory_mib": peak_memory,
        }
        history.append(row)
        print(
            f"Эпоха {epoch_index + 1:02d}/{epochs}: "
            f"train={row['train_accuracy']:.3f}%, "
            f"test={test_accuracy:.3f}%, "
            f"время={epoch_seconds:.2f} с"
        )

        _save_csv(output_dir / "history.csv", history, HISTORY_FIELDS)
        _save_csv(output_dir / "schedule.csv", schedule_history, SCHEDULE_FIELDS)
        if max_steps is not None and global_step >= max_steps:
            break

    total_seconds = time.perf_counter() - total_start
    best_row = max(history, key=lambda item: float(item["test_accuracy"]))
    summary = {
        "experiment": config.name,
        "seed": seed,
        "epochs": len(history),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": global_step,
        "parameters": count_parameters(model),
        "best_epoch": int(best_row["epoch"]),
        "best_test_accuracy": float(best_row["test_accuracy"]),
        "final_test_accuracy": float(history[-1]["test_accuracy"]),
        "final_test_loss": float(history[-1]["test_loss"]),
        "train_seconds": sum(float(row["train_seconds"]) for row in history),
        "eval_seconds": sum(float(row["eval_seconds"]) for row in history),
        "total_seconds": total_seconds,
        "peak_gpu_memory_mib": max(float(row["peak_gpu_memory_mib"]) for row in history),
        "paper_target_accuracy": 99.03 if scheduler_config["kind"] == "inverse" else 99.25,
        "paper_target_epochs": 85 if scheduler_config["kind"] == "inverse" else 12,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": raw,
            "summary": summary,
        },
        output_dir / "model.pt",
    )
    print(
        f"Готово: лучшая точность {summary['best_test_accuracy']:.3f}% "
        f"на эпохе {summary['best_epoch']}; результаты: {output_dir}"
    )
    return summary
