"""Частичное воспроизведение статьи о сверхсходимости."""

from .model import CaffeLeNet
from .schedules import InverseSchedule, OneCycleSchedule, SchedulePoint, StepSchedule

__all__ = [
    "CaffeLeNet",
    "InverseSchedule",
    "StepSchedule",
    "OneCycleSchedule",
    "SchedulePoint",
]
