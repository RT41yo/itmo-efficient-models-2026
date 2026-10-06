from __future__ import annotations

import pytest

from superconvergence.suite import aggregate_suite_rows


def test_suite_aggregation_uses_final_accuracy() -> None:
    rows = [
        {
            "order": 1,
            "experiment": "example",
            "label": "пример",
            "paper_schedule": "0.01/inv",
            "seed": 42,
            "epochs": 85,
            "total_steps": 10_000,
            "final_test_accuracy": 99.0,
            "best_test_accuracy": 99.2,
            "paper_accuracy": 99.1,
            "paper_std": 0.04,
            "accuracy_difference": -0.1,
            "total_seconds": 10.0,
            "peak_gpu_memory_mib": 200.0,
        },
        {
            "order": 1,
            "experiment": "example",
            "label": "пример",
            "paper_schedule": "0.01/inv",
            "seed": 43,
            "epochs": 85,
            "total_steps": 10_000,
            "final_test_accuracy": 99.2,
            "best_test_accuracy": 99.3,
            "paper_accuracy": 99.1,
            "paper_std": 0.04,
            "accuracy_difference": 0.1,
            "total_seconds": 12.0,
            "peak_gpu_memory_mib": 202.0,
        },
    ]
    aggregate = aggregate_suite_rows(rows)
    assert len(aggregate) == 1
    assert aggregate[0]["runs"] == 2
    assert aggregate[0]["our_mean_accuracy"] == pytest.approx(99.1)
    assert aggregate[0]["our_std_accuracy"] == pytest.approx(2**0.5 / 10)
    assert aggregate[0]["mean_total_seconds"] == pytest.approx(11.0)
