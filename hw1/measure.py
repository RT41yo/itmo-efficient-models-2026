import csv
import gc
import random
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from models import SmallCNN

import pynvml as nvml


rng = random.Random(2026)

base_sizes = [32, 64, 128, 224, 256, 384, 512]
base_batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]

extra_sizes = sorted(rng.sample(
    [s for s in range(32, 513, 16) if s not in base_sizes], 4
))
extra_batches = sorted(rng.sample(
    [b for b in range(1, 257) if b not in base_batches], 3
))

sizes = sorted(base_sizes + extra_sizes)
batches = sorted(base_batches + extra_batches)

configs = [
    (s, b, s in extra_sizes or b in extra_batches)
    for s in sizes
    for b in batches
]

print("Дополнительные S:", extra_sizes)
print("Дополнительные B:", extra_batches)
print("Всего точек:", len(configs))
print("Для калибровки:", sum(not validation for _, _, validation in configs))
print("Для проверки:", sum(validation for _, _, validation in configs))

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

nvml.nvmlInit()
gpu = nvml.nvmlDeviceGetHandleByIndex(0)

model = SmallCNN().cuda().eval()


def measure_one(s, b):
    x = None
    y = None
    try:
        x = torch.randn(b, 3, s, s, device="cuda")

        with torch.inference_mode():
            for _ in range(10):
                model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            y = model(x)
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        y = None

        times = []
        with torch.inference_mode():
            for _ in range(30):
                torch.cuda.synchronize()
                start = perf_counter()
                model(x)
                torch.cuda.synchronize()
                times.append(perf_counter() - start)

        torch.cuda.synchronize()
        energy_before_mj = nvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
        energy_start = perf_counter()
        repetitions = 0

        with torch.inference_mode():
            while perf_counter() - energy_start < 3.0:
                model(x)
                torch.cuda.synchronize()
                repetitions += 1

        energy_after_mj = nvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
        energy_j = (energy_after_mj - energy_before_mj) / 1000 / repetitions

        return median(times), peak_bytes, energy_j

    finally:
        y = None
        x = None
        gc.collect()
        torch.cuda.empty_cache()


output = Path(__file__).resolve().parent / "results" / "measurements.csv"
output.parent.mkdir(parents=True, exist_ok=True)

columns = [
    "S", "B", "latency_s", "memory_bytes",
    "energy_j", "is_validation", "status",
]

with output.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    # Проверяем запись CSV.
    for s, b, is_validation in configs:
        try:
            seconds, peak_bytes, energy_j = measure_one(s, b)
            row = {
                "S": s, "B": b,
                "latency_s": seconds,
                "memory_bytes": peak_bytes,
                "energy_j": energy_j,
                "is_validation": is_validation,
                "status": "OK",
            }
        except torch.cuda.OutOfMemoryError:
            row = {
                "S": s, "B": b,
                "latency_s": "",
                "memory_bytes": "OOM",
                "energy_j": "",
                "is_validation": is_validation,
                "status": "OOM",
            }

        writer.writerow(row)
        file.flush()
        print(row, flush=True)

print("CSV сохранен:", output)
nvml.nvmlShutdown()
