from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim import Optimizer


@dataclass(frozen=True)
class SchedulePoint:
    lr: float
    momentum: float
    phase: str


def _linear_inclusive(start: float, end: float, index: int, count: int) -> float:
    if count <= 1:
        return end
    fraction = index / (count - 1)
    return start + (end - start) * fraction


def _linear_to_boundary(start: float, end: float, index: int, count: int) -> float:
    """Линейное движение к границе, которая станет первым шагом следующей фазы."""

    fraction = index / count
    return start + (end - start) * fraction


class InverseSchedule:
    """Обратно-степенное расписание из стандартного Caffe LeNet solver.

    lr(step) = base_lr * (1 + gamma * step) ** (-power)
    """

    def __init__(
        self,
        base_lr: float = 0.01,
        momentum: float = 0.9,
        gamma: float = 0.0001,
        power: float = 0.75,
    ) -> None:
        self.base_lr = float(base_lr)
        self.momentum = float(momentum)
        self.gamma = float(gamma)
        self.power = float(power)

    def __call__(self, step: int) -> SchedulePoint:
        lr = self.base_lr * (1.0 + self.gamma * step) ** (-self.power)
        return SchedulePoint(lr=lr, momentum=self.momentum, phase="inverse")


class OneCycleSchedule:
    """Линейный 1cycle с отдельным завершающим снижением скорости."""

    def __init__(
        self,
        up_steps: int,
        down_steps: int,
        final_steps: int,
        min_lr: float,
        max_lr: float,
        final_lr: float,
        min_momentum: float,
        max_momentum: float,
    ) -> None:
        if min(up_steps, down_steps, final_steps) <= 0:
            raise ValueError("Каждый этап 1cycle должен содержать хотя бы один шаг")
        if not 0 < final_lr < min_lr < max_lr:
            raise ValueError("Требуется final_lr < min_lr < max_lr")
        if not 0 <= min_momentum < max_momentum < 1:
            raise ValueError("Требуется 0 <= min_momentum < max_momentum < 1")

        self.up_steps = int(up_steps)
        self.down_steps = int(down_steps)
        self.final_steps = int(final_steps)
        self.total_steps = self.up_steps + self.down_steps + self.final_steps
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.final_lr = float(final_lr)
        self.min_momentum = float(min_momentum)
        self.max_momentum = float(max_momentum)

    def __call__(self, step: int) -> SchedulePoint:
        if step < 0 or step >= self.total_steps:
            raise IndexError(f"Шаг {step} находится вне диапазона 0..{self.total_steps - 1}")

        if step < self.up_steps:
            lr = _linear_to_boundary(self.min_lr, self.max_lr, step, self.up_steps)
            momentum = _linear_to_boundary(
                self.max_momentum, self.min_momentum, step, self.up_steps
            )
            return SchedulePoint(lr=lr, momentum=momentum, phase="up")

        if step < self.up_steps + self.down_steps:
            local_step = step - self.up_steps
            lr = _linear_to_boundary(self.max_lr, self.min_lr, local_step, self.down_steps)
            momentum = _linear_to_boundary(
                self.min_momentum, self.max_momentum, local_step, self.down_steps
            )
            return SchedulePoint(lr=lr, momentum=momentum, phase="down")

        local_step = step - self.up_steps - self.down_steps
        lr = _linear_inclusive(self.min_lr, self.final_lr, local_step, self.final_steps)
        return SchedulePoint(lr=lr, momentum=self.max_momentum, phase="final")


def apply_schedule(optimizer: "Optimizer", point: SchedulePoint) -> None:
    """Применяет базовую LR и Momentum, сохраняя Caffe lr_mult групп."""

    for group in optimizer.param_groups:
        lr_scale = float(group.get("lr_scale", 1.0))
        group["lr"] = point.lr * lr_scale
        group["momentum"] = point.momentum
