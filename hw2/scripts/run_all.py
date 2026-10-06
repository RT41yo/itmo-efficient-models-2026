#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HW2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW2_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Последовательный запуск двух режимов")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-onecycle", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("Выполняется:", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    train_script = str(HW2_ROOT / "scripts" / "train.py")
    if not args.skip_onecycle:
        run(
            [
                sys.executable,
                train_script,
                "--config",
                str(HW2_ROOT / "configs" / "onecycle.json"),
            ]
        )
    if not args.skip_baseline:
        run(
            [
                sys.executable,
                train_script,
                "--config",
                str(HW2_ROOT / "configs" / "baseline.json"),
            ]
        )

    run(
        [
            sys.executable,
            str(HW2_ROOT / "scripts" / "plot_results.py"),
            "--baseline",
            str(HW2_ROOT / "results" / "baseline"),
            "--onecycle",
            str(HW2_ROOT / "results" / "onecycle"),
            "--output-dir",
            str(HW2_ROOT / "results"),
        ]
    )


if __name__ == "__main__":
    main()
