from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig, load_config


RUN_FIELDS = [
    "order",
    "experiment",
    "label",
    "paper_schedule",
    "seed",
    "epochs",
    "total_steps",
    "final_test_accuracy",
    "best_test_accuracy",
    "paper_accuracy",
    "paper_std",
    "accuracy_difference",
    "total_seconds",
    "peak_gpu_memory_mib",
]

AGGREGATE_FIELDS = [
    "order",
    "experiment",
    "label",
    "paper_schedule",
    "runs",
    "epochs",
    "mean_total_steps",
    "our_mean_accuracy",
    "our_std_accuracy",
    "paper_accuracy",
    "paper_std",
    "accuracy_difference",
    "mean_total_seconds",
    "mean_peak_gpu_memory_mib",
]


def discover_suite_configs(hw2_root: str | Path, repo_root: str | Path) -> list[ExperimentConfig]:
    hw2_root = Path(hw2_root)
    configs = [load_config(path, repo_root) for path in (hw2_root / "configs").glob("*.json")]
    suite_configs = [config for config in configs if isinstance(config.raw.get("paper"), dict)]
    return sorted(suite_configs, key=lambda config: int(config.raw["paper"]["order"]))


def result_dir_for_seed(config: ExperimentConfig, seed: int) -> Path:
    base = config.resolve_path(config.section("training")["output_dir"])
    if seed == config.seed:
        return base
    return base.with_name(f"{base.name}_seed{seed}")


def collect_suite_rows(configs: Iterable[ExperimentConfig], seeds: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        paper = config.raw["paper"]
        for seed in seeds:
            result_dir = result_dir_for_seed(config, int(seed))
            summary_path = result_dir / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"Нет результата {summary_path}. Сначала выполните соответствующий запуск."
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            final_accuracy = float(summary["final_test_accuracy"])
            paper_accuracy = float(paper["target_accuracy"])
            rows.append(
                {
                    "order": int(paper["order"]),
                    "experiment": config.name,
                    "label": str(paper["label"]),
                    "paper_schedule": str(paper["schedule"]),
                    "seed": int(seed),
                    "epochs": int(summary["epochs"]),
                    "total_steps": int(summary["total_steps"]),
                    "final_test_accuracy": final_accuracy,
                    "best_test_accuracy": float(summary["best_test_accuracy"]),
                    "paper_accuracy": paper_accuracy,
                    "paper_std": float(paper["target_std"]),
                    "accuracy_difference": final_accuracy - paper_accuracy,
                    "total_seconds": float(summary["total_seconds"]),
                    "peak_gpu_memory_mib": float(summary["peak_gpu_memory_mib"]),
                }
            )
    return rows


def aggregate_suite_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["experiment"]), []).append(row)

    aggregate: list[dict[str, Any]] = []
    for experiment_rows in groups.values():
        first = experiment_rows[0]
        accuracies = [float(row["final_test_accuracy"]) for row in experiment_rows]
        times = [float(row["total_seconds"]) for row in experiment_rows]
        steps = [float(row["total_steps"]) for row in experiment_rows]
        memories = [float(row["peak_gpu_memory_mib"]) for row in experiment_rows]
        mean_accuracy = statistics.mean(accuracies)
        paper_accuracy = float(first["paper_accuracy"])
        aggregate.append(
            {
                "order": int(first["order"]),
                "experiment": str(first["experiment"]),
                "label": str(first["label"]),
                "paper_schedule": str(first["paper_schedule"]),
                "runs": len(experiment_rows),
                "epochs": int(first["epochs"]),
                "mean_total_steps": statistics.mean(steps),
                "our_mean_accuracy": mean_accuracy,
                "our_std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "paper_accuracy": paper_accuracy,
                "paper_std": float(first["paper_std"]),
                "accuracy_difference": mean_accuracy - paper_accuracy,
                "mean_total_seconds": statistics.mean(times),
                "mean_peak_gpu_memory_mib": statistics.mean(memories),
            }
        )
    return sorted(aggregate, key=lambda row: int(row["order"]))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_suite_tables(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "mnist_runs.csv"
    aggregate_path = output_dir / "mnist_summary.csv"
    _write_csv(runs_path, rows, RUN_FIELDS)
    _write_csv(aggregate_path, aggregate, AGGREGATE_FIELDS)
    return runs_path, aggregate_path
