from __future__ import annotations

import pytest

from superconvergence.schedules import InverseSchedule, OneCycleSchedule


def test_inverse_schedule_matches_caffe_formula() -> None:
    schedule = InverseSchedule(base_lr=0.01, momentum=0.9, gamma=0.0001, power=0.75)
    assert schedule(0).lr == pytest.approx(0.01)
    assert schedule(10_000).lr == pytest.approx(0.01 * 2 ** (-0.75))
    assert schedule(123).momentum == pytest.approx(0.9)


def test_onecycle_reaches_all_declared_boundaries() -> None:
    schedule = OneCycleSchedule(
        up_steps=3,
        down_steps=3,
        final_steps=3,
        min_lr=0.01,
        max_lr=0.1,
        final_lr=1e-5,
        min_momentum=0.8,
        max_momentum=0.95,
    )
    assert schedule(0).lr == pytest.approx(0.01)
    assert schedule(0).momentum == pytest.approx(0.95)
    assert schedule(3).lr == pytest.approx(0.1)
    assert schedule(3).momentum == pytest.approx(0.8)
    assert schedule(6).lr == pytest.approx(0.01)
    assert schedule(6).momentum == pytest.approx(0.95)
    assert schedule(8).lr == pytest.approx(1e-5)
    assert schedule(8).momentum == pytest.approx(0.95)


def test_onecycle_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        OneCycleSchedule(
            up_steps=0,
            down_steps=3,
            final_steps=3,
            min_lr=0.01,
            max_lr=0.1,
            final_lr=1e-5,
            min_momentum=0.8,
            max_momentum=0.95,
        )


def test_onecycle_rejects_out_of_range_step() -> None:
    schedule = OneCycleSchedule(2, 2, 2, 0.01, 0.1, 1e-5, 0.8, 0.95)
    with pytest.raises(IndexError):
        schedule(6)
