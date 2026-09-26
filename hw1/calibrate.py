import numpy as np
import csv
import json

from scipy.optimize import least_squares
from equations import latency
from pathlib import Path
from equations import energy


csv_path = Path(__file__).resolve().parent / "results" / "measurements.csv"

with csv_path.open(newline="") as file:
    train = [
        row for row in csv.DictReader(file)
        if row["status"] == "OK" and row["is_validation"] == "False"
    ]

print("Точек для калибровки:", len(train))

sizes = np.array([int(row["S"]) for row in train])
batches = np.array([int(row["B"]) for row in train])
measured = np.array([float(row["latency_s"]) for row in train])

def errors(parameters):
    t0_us, p_tflops, w_gbs = parameters

    theta = {
        "t0": t0_us * 1e-6,      # микросекунды → секунды
        "P": p_tflops * 1e12,    # TFLOP/s → FLOP/s
        "W": w_gbs * 1e9,        # GB/s → байт/с
    }

    predicted = latency(sizes, batches, theta)
    return np.log(predicted / measured)

fit = least_squares(
    errors,
    x0=[180, 10, 300],
    bounds=([0, 0.01, 0.01], [1000, 1000, 100000]),
)

print("t0, мкс:", fit.x[0])
print("P, TFLOP/s:", fit.x[1])
print("W, GB/s:", fit.x[2])

validation = [
    row for row in csv.DictReader(csv_path.open(newline=""))
    if row["status"] == "OK" and row["is_validation"] == "True"
]

theta = {
    "t0": fit.x[0] * 1e-6,
    "P": fit.x[1] * 1e12,
    "W": fit.x[2] * 1e9,
}

s_val = np.array([int(row["S"]) for row in validation])
b_val = np.array([int(row["B"]) for row in validation])
real = np.array([float(row["latency_s"]) for row in validation])

predicted = latency(s_val, b_val, theta)
relative_error = np.abs(predicted - real) / real
worst = np.argmax(relative_error)

print("Проверочных точек:", len(validation))
print("Средняя относительная ошибка:", relative_error.mean() * 100, "%")
print(
    "Наибольшая ошибка при S, B:",
    s_val[worst], b_val[worst],
    "—", relative_error[worst] * 100, "%",
)

measured_energy = np.array([float(row["energy_j"]) for row in train])

def energy_errors(parameters):
    power_w, joules_per_tflop, joules_per_gb = parameters

    theta_energy = {
        **theta,
        "base_power_w": power_w,
        "joules_per_flop": joules_per_tflop / 1e12,
        "joules_per_byte": joules_per_gb / 1e9,
    }

    predicted = energy(sizes, batches, theta_energy)
    return np.log(predicted / measured_energy)

energy_fit = least_squares(
    energy_errors,
    x0=[35, 10, 1],
    bounds=([0, 0, 0], [1000, 1000, 1000]),
)

print("P0, Вт:", energy_fit.x[0])
print("e_F, Дж/TFLOP:", energy_fit.x[1])
print("e_D, Дж/ГБ:", energy_fit.x[2])

theta_energy = {
    **theta,
    "base_power_w": energy_fit.x[0],
    "joules_per_flop": energy_fit.x[1] / 1e12,
    "joules_per_byte": energy_fit.x[2] / 1e9,
}

real_energy = np.array([float(row["energy_j"]) for row in validation])
predicted_energy = energy(s_val, b_val, theta_energy)
energy_error = np.abs(predicted_energy - real_energy) / real_energy
worst_energy = np.argmax(energy_error)

print("Средняя ошибка энергии:", energy_error.mean() * 100, "%")
print(
    "Наибольшая ошибка энергии при S, B:",
    s_val[worst_energy], b_val[worst_energy],
    "—", energy_error[worst_energy] * 100, "%",
)

theta_path = csv_path.parent / "theta.json"
theta_path.write_text(
    json.dumps(
        {"latency": theta, "energy": theta_energy},
        indent=2,
    ),
    encoding="utf-8",
)
print("Параметры сохранены:", theta_path)
