import numpy as np


def flops(image_size, batch):
    """Число FLOPs одного прямого прохода."""
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)

    result = b * (17714 * s**2 + 313700)
    return result.item() if result.ndim == 0 else result

def memory(image_size, batch):
    """Прогноз пиковой памяти прямого прохода, в байтах."""
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)

    result = 4_161_296 + 52 * b * s**2
    return result.item() if result.ndim == 0 else result

def bytes_moved(image_size, batch):
    """Расчётный объем чтения и записи за проход, в байтах."""
    s = np.asarray(image_size, dtype=np.float64)
    b = np.asarray(batch, dtype=np.float64)

    result = 4 * (91 * b * s**2 + 2148 * b + 1_040_324)
    return result.item() if result.ndim == 0 else result

def latency(image_size, batch, theta):
    """Прогноз времени одного прохода, в секундах."""
    t0 = theta["t0"]
    P = theta["P"]
    W = theta["W"]

    result = t0 + np.maximum(
        flops(image_size, batch) / P,
        bytes_moved(image_size, batch) / W,
    )
    return result.item() if np.ndim(result) == 0 else result

def energy(image_size, batch, theta_energy):
    """Прогноз энергии одного прохода всей GPU, в джоулях."""
    result = (
        theta_energy["base_power_w"] * latency(image_size, batch, theta_energy)
        + theta_energy["joules_per_flop"] * flops(image_size, batch)
        + theta_energy["joules_per_byte"] * bytes_moved(image_size, batch)
    )
    return result.item() if np.ndim(result) == 0 else result
