#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch import nn

HW2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW2_ROOT.parent
sys.path.insert(0, str(HW2_ROOT / "src"))

from superconvergence.config import load_config
from superconvergence.data import build_mnist_loaders
from superconvergence.engine import build_optimizer, resolve_device, seed_everything
from superconvergence.model import CaffeLeNet
from superconvergence.schedules import SchedulePoint, apply_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка диапазона скорости обучения")
    parser.add_argument(
        "--config",
        type=Path,
        default=HW2_ROOT / "configs" / "baseline.json",
        help="Базовая конфигурация данных, модели и оптимизатора",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--max-lr", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=HW2_ROOT / "results" / "lr_range")
    parser.add_argument("--max-steps", type=int, help="По умолчанию используется одна эпоха")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.min_lr < args.max_lr:
        raise ValueError("Требуется 0 < min-lr < max-lr")

    config = load_config(args.config, repo_root=REPO_ROOT)
    data_config = config.section("data")
    training_config = config.section("training")
    optimizer_config = dict(config.section("optimizer"))
    seed_everything(config.seed)
    device = resolve_device(str(training_config.get("device", "auto")))
    loaders = build_mnist_loaders(
        root=config.resolve_path(data_config["root"]),
        batch_size=int(data_config["batch_size"]),
        test_batch_size=int(data_config["test_batch_size"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]) and device.type == "cuda",
        download=bool(data_config["download"]),
        drop_last=bool(data_config["drop_last"]),
        seed=config.seed,
        caffe_pixel_scale=bool(data_config["caffe_pixel_scale"]),
    )

    model = CaffeLeNet().to(device)
    optimizer_config["lr"] = args.min_lr
    optimizer = build_optimizer(model, optimizer_config)
    criterion = nn.CrossEntropyLoss()
    total_steps = min(args.max_steps or len(loaders.train), len(loaders.train))
    rows: list[dict[str, float | int]] = []
    smoothed_loss = 0.0
    best_loss = math.inf
    smoothing = 0.98

    model.train()
    for step, (inputs, targets) in enumerate(loaders.train):
        if step >= total_steps:
            break
        fraction = step / max(1, total_steps - 1)
        lr = args.min_lr + (args.max_lr - args.min_lr) * fraction
        apply_schedule(
            optimizer,
            SchedulePoint(lr=lr, momentum=float(optimizer_config["momentum"]), phase="range"),
        )
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        value = float(loss.item())
        smoothed_loss = smoothing * smoothed_loss + (1.0 - smoothing) * value
        corrected_loss = smoothed_loss / (1.0 - smoothing ** (step + 1))
        best_loss = min(best_loss, corrected_loss)
        rows.append({"step": step, "lr": lr, "loss": value, "smoothed_loss": corrected_loss})
        print(f"Шаг {step + 1:03d}/{total_steps}: LR={lr:.6f}, потери={corrected_loss:.4f}")
        if step > 10 and corrected_loss > 4.0 * best_loss:
            print("Проверка остановлена: функция потерь заметно выросла")
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "lr_range.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "lr", "loss", "smoothed_loss"])
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row["lr"] for row in rows], [row["smoothed_loss"] for row in rows])
    axis.axvline(0.01, color="C1", linestyle="--", label="нижняя граница статьи: 0,01")
    axis.axvline(0.1, color="C2", linestyle="--", label="верхняя граница статьи: 0,1")
    axis.set_xlabel("Скорость обучения")
    axis.set_ylabel("Сглаженная функция потерь")
    axis.set_title("Проверка диапазона скорости обучения")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plot_path = args.output_dir / "lr_range.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    print(f"Данные: {csv_path}")
    print(f"График: {plot_path}")


if __name__ == "__main__":
    main()

