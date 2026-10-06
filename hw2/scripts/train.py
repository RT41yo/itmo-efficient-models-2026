#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW2_ROOT.parent
sys.path.insert(0, str(HW2_ROOT / "src"))

from superconvergence.config import load_config
from superconvergence.engine import run_experiment
from superconvergence.plotting import plot_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение LeNet на MNIST")
    parser.add_argument("--config", type=Path, required=True, help="Путь к JSON-конфигурации")
    parser.add_argument("--epochs", type=int, help="Замена числа эпох; только для быстрой проверки baseline")
    parser.add_argument("--seed", type=int, help="Начальное случайное число вместо значения из конфигурации")
    parser.add_argument("--output-dir", type=Path, help="Каталог результатов вместо значения из конфигурации")
    parser.add_argument("--limit-train-batches", type=int, help="Ограничение батчей в эпохе для быстрой проверки")
    parser.add_argument("--limit-test-batches", type=int, help="Ограничение тестовых батчей для быстрой проверки")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, repo_root=REPO_ROOT)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else config.resolve_path(config.section("training")["output_dir"])
    )
    run_experiment(
        config,
        epochs_override=args.epochs,
        seed_override=args.seed,
        output_dir_override=output_dir,
        limit_train_batches=args.limit_train_batches,
        limit_test_batches=args.limit_test_batches,
    )
    plot_path = plot_run(output_dir)
    print(f"График сохранён: {plot_path}")


if __name__ == "__main__":
    main()
