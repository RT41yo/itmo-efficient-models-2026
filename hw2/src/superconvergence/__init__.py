"""Частичное воспроизведение статьи о сверхсходимости."""

from .model import CaffeLeNet
from .schedules import InverseSchedule, OneCycleSchedule, SchedulePoint

__all__ = ["CaffeLeNet", "InverseSchedule", "OneCycleSchedule", "SchedulePoint"]

