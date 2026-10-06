from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            {key: float(value) for key, value in row.items() if key != ""}
            for row in csv.DictReader(file)
        ]


def plot_run(result_dir: str | Path) -> Path:
    result_dir = Path(result_dir)
    history = _read_history(result_dir / "history.csv")
    epochs = [row["epoch"] for row in history]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(
        epochs,
        [row["train_loss"] for row in history],
        marker="o",
        markersize=4,
        label="обучение",
    )
    axes[0, 0].plot(
        epochs,
        [row["test_loss"] for row in history],
        marker="o",
        markersize=4,
        label="тест",
    )
    axes[0, 0].set_title("Функция потерь")
    axes[0, 0].legend()

    axes[0, 1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        marker="o",
        markersize=4,
        label="обучение",
    )
    axes[0, 1].plot(
        epochs,
        [row["test_accuracy"] for row in history],
        marker="o",
        markersize=4,
        label="тест",
    )
    axes[0, 1].set_title("Точность, %")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["lr_end"] for row in history], marker="o", markersize=4)
    axes[1, 0].set_title("Скорость обучения в конце эпохи")
    axes[1, 0].set_yscale("log")

    axes[1, 1].plot(
        epochs,
        [row["momentum_end"] for row in history],
        marker="o",
        markersize=4,
    )
    axes[1, 1].set_title("Инерция в конце эпохи")

    for axis in axes.flat:
        axis.set_xlabel("Эпоха")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    output = result_dir / "training_curves.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_comparison(
    baseline_dir: str | Path,
    onecycle_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    baseline_dir = Path(baseline_dir)
    onecycle_dir = Path(onecycle_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _read_history(baseline_dir / "history.csv")
    onecycle = _read_history(onecycle_dir / "history.csv")
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    onecycle_summary = json.loads((onecycle_dir / "summary.json").read_text(encoding="utf-8"))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        [row["epoch"] for row in baseline],
        [row["test_accuracy"] for row in baseline],
        label="обычное обучение",
    )
    axes[0].plot(
        [row["epoch"] for row in onecycle],
        [row["test_accuracy"] for row in onecycle],
        label="1cycle",
    )
    axes[0].axhline(99.03, color="C0", linestyle="--", alpha=0.5, label="статья: 99,03%")
    axes[0].axhline(99.25, color="C1", linestyle="--", alpha=0.5, label="статья: 99,25%")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Тестовая точность, %")
    axes[0].set_title("Качество по эпохам")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    labels = ["обычное", "1cycle"]
    seconds = [baseline_summary["total_seconds"], onecycle_summary["total_seconds"]]
    bars = axes[1].bar(labels, seconds, color=["C0", "C1"])
    axes[1].set_ylabel("Секунды")
    axes[1].set_title("Полное время обучения")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, seconds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom")

    figure.tight_layout()
    output = output_dir / "comparison.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_mnist_suite(
    aggregate: list[dict[str, float | int | str]],
    output_dir: str | Path,
) -> Path:
    """Строит общее сравнение всех шести строк MNIST–LeNet."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in aggregate]
    positions = list(range(len(labels)))
    our_accuracy = [float(row["our_mean_accuracy"]) for row in aggregate]
    our_std = [float(row["our_std_accuracy"]) for row in aggregate]
    paper_accuracy = [float(row["paper_accuracy"]) for row in aggregate]
    paper_std = [float(row["paper_std"]) for row in aggregate]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    width = 0.38
    axes[0, 0].bar(
        [position - width / 2 for position in positions],
        paper_accuracy,
        width,
        yerr=paper_std,
        capsize=3,
        label="статья",
    )
    axes[0, 0].bar(
        [position + width / 2 for position in positions],
        our_accuracy,
        width,
        yerr=our_std,
        capsize=3,
        label="воспроизведение",
    )
    lower = min(paper_accuracy + our_accuracy) - 0.2
    upper = max(paper_accuracy + our_accuracy) + 0.12
    axes[0, 0].set_ylim(lower, upper)
    axes[0, 0].set_ylabel("Тестовая точность, %")
    axes[0, 0].set_title("Финальная точность")
    axes[0, 0].legend()

    epochs = [float(row["epochs"]) for row in aggregate]
    axes[0, 1].bar(positions, epochs, color="C2")
    axes[0, 1].set_ylabel("Эпохи")
    axes[0, 1].set_title("Продолжительность обучения")

    steps = [float(row["mean_total_steps"]) for row in aggregate]
    axes[1, 0].bar(positions, steps, color="C3")
    axes[1, 0].set_ylabel("Обновления весов")
    axes[1, 0].set_title("Число обновлений")

    seconds = [float(row["mean_total_seconds"]) for row in aggregate]
    axes[1, 1].bar(positions, seconds, color="C4")
    axes[1, 1].set_ylabel("Секунды")
    axes[1, 1].set_title("Измеренное время")

    for axis in axes.flat:
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.3)

    figure.suptitle("Воспроизведение опытов MNIST–LeNet", fontsize=15)
    figure.tight_layout()
    output = output_dir / "mnist_comparison.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
