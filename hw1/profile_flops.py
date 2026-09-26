import csv
import gc
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from equations import flops
from models import SmallCNN


root = Path(__file__).resolve().parent
measurements_path = root / "results" / "measurements.csv"
output_path = root / "results" / "flops_profile.csv"

with measurements_path.open(newline="") as file:
    measurements = list(csv.DictReader(file))

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

model = SmallCNN().cuda().eval()

columns = [
    "S", "B", "profiler_flops", "formula_flops",
    "missing_flops_expected", "is_validation", "status",
]

with output_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for index, measurement in enumerate(measurements, start=1):
        s = int(measurement["S"])
        b = int(measurement["B"])
        x = None
        y = None
        prof = None

        try:
            x = torch.randn(b, 3, s, s, device="cuda")

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as prof:
                with torch.inference_mode():
                    y = model(x)
                torch.cuda.synchronize()

            profiler_flops = sum(
                event.flops or 0 for event in prof.key_averages()
            )

            result = {
                "S": s,
                "B": b,
                "profiler_flops": int(profiler_flops),
                "formula_flops": int(flops(s, b)),
                "missing_flops_expected": 2 * b * s * s + 356 * b,
                "is_validation": measurement["is_validation"],
                "status": "OK",
            }

        except torch.cuda.OutOfMemoryError:
            result = {
                "S": s,
                "B": b,
                "profiler_flops": "",
                "formula_flops": int(flops(s, b)),
                "missing_flops_expected": 2 * b * s * s + 356 * b,
                "is_validation": measurement["is_validation"],
                "status": "OOM",
            }

        finally:
            del x, y, prof
            gc.collect()
            torch.cuda.empty_cache()

        writer.writerow(result)
        file.flush()

        if index % 10 == 0 or index == len(measurements):
            print(f"Проверено: {index}/{len(measurements)}", flush=True)

print("Результаты сохранены:", output_path)
