from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Проверенная конфигурация одного запуска."""

    raw: dict[str, Any]
    source_path: Path
    repo_root: Path

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Раздел конфигурации {name!r} отсутствует или имеет неверный формат")
        return value

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path


def load_config(path: str | Path, repo_root: str | Path) -> ExperimentConfig:
    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        raise ValueError("Корень конфигурации должен быть объектом JSON")
    for required in ("name", "data", "optimizer", "scheduler", "training"):
        if required not in raw:
            raise ValueError(f"В конфигурации отсутствует обязательный раздел {required!r}")

    training = raw["training"]
    if int(training["epochs"]) <= 0:
        raise ValueError("Число эпох должно быть положительным")

    scheduler = raw["scheduler"]
    if scheduler["kind"] not in {"inverse", "step", "onecycle"}:
        raise ValueError(f"Неизвестное расписание: {scheduler['kind']!r}")
    if scheduler["kind"] == "step":
        if int(scheduler["step_size"]) <= 0:
            raise ValueError("Размер ступени должен быть положительным")
        if not 0 < float(scheduler["gamma"]) < 1:
            raise ValueError("Коэффициент step должен находиться между 0 и 1")
    if scheduler["kind"] == "onecycle":
        phase_epochs = (
            int(scheduler["up_epochs"])
            + int(scheduler["down_epochs"])
            + int(scheduler["final_epochs"])
        )
        if phase_epochs != int(training["epochs"]):
            raise ValueError(
                "Для 1cycle сумма up_epochs, down_epochs и final_epochs "
                "должна совпадать с training.epochs"
            )

    paper = raw.get("paper")
    if paper is not None:
        for required in ("order", "label", "schedule", "target_accuracy", "target_std", "target_epochs"):
            if required not in paper:
                raise ValueError(f"В разделе 'paper' отсутствует поле {required!r}")

    return ExperimentConfig(raw=raw, source_path=source_path, repo_root=Path(repo_root).resolve())
