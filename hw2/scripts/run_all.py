#!/usr/bin/env python3
"""Короткое имя для полного набора опытов MNIST–LeNet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HW2_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [
        sys.executable,
        str(HW2_ROOT / "scripts" / "run_mnist_suite.py"),
        *sys.argv[1:],
    ]
    subprocess.run(command, cwd=HW2_ROOT.parent, check=True)


if __name__ == "__main__":
    main()
