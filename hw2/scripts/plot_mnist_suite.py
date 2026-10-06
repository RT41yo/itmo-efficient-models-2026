#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW2_ROOT.parent
sys.path.insert(0, str(HW2_ROOT / "src"))

from superconvergence.plotting import plot_mnist_suite
from superconvergence.suite import (
    aggregate_suite_rows,
    collect_suite_rows,
    discover_suite_configs,
    write_suite_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сводка опытов MNIST–LeNet")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output-dir", type=Path, default=HW2_ROOT / "results")
    args = parser.parse_args()

    configs = discover_suite_configs(HW2_ROOT, REPO_ROOT)
    rows = collect_suite_rows(configs, args.seeds)
    aggregate = aggregate_suite_rows(rows)
    runs_path, summary_path = write_suite_tables(args.output_dir, rows, aggregate)
    plot_path = plot_mnist_suite(aggregate, args.output_dir)
    print(f"Результаты отдельных запусков: {runs_path}")
    print(f"Сводная таблица: {summary_path}")
    print(f"Сводный график: {plot_path}")


if __name__ == "__main__":
    main()
