#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HW2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW2_ROOT.parent
sys.path.insert(0, str(HW2_ROOT / "src"))

from superconvergence.suite import discover_suite_configs, result_dir_for_seed


def run(command: list[str]) -> None:
    print("Выполняется:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск шести опытов MNIST–LeNet из таблицы 2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Не запускать опыт повторно, если его summary.json уже существует",
    )
    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="Не обучать модели, а только заново построить общую сводку",
    )
    args = parser.parse_args()

    configs = discover_suite_configs(HW2_ROOT, REPO_ROOT)
    if len(configs) != 6:
        raise RuntimeError(f"Ожидалось 6 конфигураций MNIST–LeNet, найдено {len(configs)}")

    if not args.only_summary:
        for config in configs:
            for seed in args.seeds:
                output_dir = result_dir_for_seed(config, seed)
                summary_path = output_dir / "summary.json"
                if args.skip_existing and summary_path.is_file():
                    print(f"Пропуск готового опыта: {config.name}, seed={seed}")
                    continue
                run(
                    [
                        sys.executable,
                        str(HW2_ROOT / "scripts" / "train.py"),
                        "--config",
                        str(config.source_path),
                        "--seed",
                        str(seed),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

    run(
        [
            sys.executable,
            str(HW2_ROOT / "scripts" / "plot_mnist_suite.py"),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--output-dir",
            str(HW2_ROOT / "results"),
        ]
    )


if __name__ == "__main__":
    main()
