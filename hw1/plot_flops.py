import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from equations import flops


root = Path(__file__).resolve().parent

with (root / "results" / "flops_profile.csv").open(newline="") as file:
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
    batch_line = np.arange(1, 257)
    prediction_gflops = flops(s, batch_line) / 1e9

    ax.plot(
        batch_line,
        prediction_gflops,
        color="black",
        label="Analytical total",
    )

    for is_validation, marker, color, label in [
        ("False", "o", "tab:blue", "Profiler: calibration"),
        ("True", "x", "tab:orange", "Profiler: validation"),
    ]:
        points = [
            row for row in rows
            if int(row["S"]) == s
            and row["status"] == "OK"
            and row["is_validation"] == is_validation
        ]

        ax.scatter(
            [int(row["B"]) for row in points],
            [int(row["profiler_flops"]) / 1e9 for row in points],
            marker=marker,
            color=color,
            s=25,
            label=label,
        )

    ax.set_title(f"S = {s}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(alpha=0.2)

axes.flat[-1].axis("off")
axes.flat[-1].legend(
    *axes.flat[0].get_legend_handles_labels(),
    loc="center",
)

fig.supxlabel("Batch size B (images)")
fig.supylabel("Operations per forward pass (GFLOPs)")

output = root / "results" / "figures" / "flops.png"
output.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output, dpi=160)
plt.close(fig)

print("График сохранен:", output)
