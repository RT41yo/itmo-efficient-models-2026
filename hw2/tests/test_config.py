from __future__ import annotations

import json
from pathlib import Path

import pytest

from superconvergence.config import load_config


HW2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HW2_ROOT.parent


def test_project_configs_are_valid() -> None:
    baseline = load_config(HW2_ROOT / "configs" / "baseline.json", REPO_ROOT)
    onecycle = load_config(HW2_ROOT / "configs" / "onecycle.json", REPO_ROOT)
    assert baseline.section("training")["epochs"] == 85
    assert baseline.section("training")["max_steps"] == 10_000
    assert onecycle.section("training")["epochs"] == 12
    assert onecycle.section("scheduler")["up_epochs"] == 5
    assert onecycle.section("scheduler")["down_epochs"] == 5


def test_onecycle_phase_sum_is_checked(tmp_path: Path) -> None:
    source = json.loads((HW2_ROOT / "configs" / "onecycle.json").read_text(encoding="utf-8"))
    source["scheduler"]["final_epochs"] = 3
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="сумма"):
        load_config(invalid, REPO_ROOT)
