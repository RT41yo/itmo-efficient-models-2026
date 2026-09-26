import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from equations import memory


root = Path(__file__).resolve().parent

with (root / "results" / "measurements.csv").open(newline="") as file:
    rows = list(csv.DictReader(file))

sizes = sorted({int(row["S"]) for row in rows})

fig, axes = plt.subplots(
    4, 3,
    figsize=(13, 14),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for s, ax in zip(sizes, axes.flat):
    # Линия: предсказание формулы для B от 1 до 256.
    batch_line = np.arange(1, 257)
    prediction_mb = memory(s, batch_line) / 1e6

    ax.plot(
        batch_line,
        prediction_mb,
        color="black",
        label="Prediction",
    )

    # Точки: реальные измерения, отдельно для калибровки и проверки.
    for is_validation, marker, color, label in [
        ("False", "o", "tab:blue", "Calibration"),
        ("True", "x", "tab:orange", "Validation"),
    ]:
        points = [
            row for row in rows
            if int(row["S"]) == s
            and row["status"] == "OK"
            and row["is_validation"] == is_validation
        ]

        ax.scatter(
            [int(row["B"]) for row in points],
            [float(row["memory_bytes"]) / 1e6 for row in points],
            marker=marker,
            color=color,
            s=25,
            label=label,
        )

    ax.set_title(f"S = {s}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(alpha=0.2)

# Последняя, двенадцатая ячейка нужна для легенды.
axes.flat[-1].axis("off")
axes.flat[-1].legend(
    *axes.flat[0].get_legend_handles_labels(),
    loc="center",
)

fig.supxlabel("Batch size B (images)")
fig.supylabel("Peak allocated GPU memory (MB)")

output = root / "results" / "figures" / "memory.png"
output.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output, dpi=160)
plt.close(fig)

print("График сохранен:", output)
