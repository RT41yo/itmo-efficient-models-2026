#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HW2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW2_ROOT / "src"))

from superconvergence.plotting import plot_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение результатов двух запусков")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--onecycle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = plot_comparison(args.baseline, args.onecycle, args.output_dir)
    print(f"Сводный график сохранён: {output}")


if __name__ == "__main__":
    main()

