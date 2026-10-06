from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from superconvergence.data import ToCaffeTensor
from superconvergence.model import CaffeLeNet, build_caffe_parameter_groups, count_parameters
from superconvergence.schedules import SchedulePoint, apply_schedule


def test_lenet_output_shape_and_parameter_count() -> None:
    model = CaffeLeNet()
    output = model(torch.zeros(4, 1, 28, 28))
    assert output.shape == (4, 10)
    assert count_parameters(model) == 431_080


def test_caffe_pixel_scale() -> None:
    pixels = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    tensor = ToCaffeTensor()(Image.fromarray(pixels, mode="L"))
    assert tensor.shape == (1, 2, 2)
    assert tensor[0, 1, 0].item() == pytest.approx(255 / 256)
    assert tensor[0, 0, 1].item() == pytest.approx(128 / 256)


def test_caffe_bias_learning_rate_multiplier() -> None:
    model = CaffeLeNet()
    groups = build_caffe_parameter_groups(model, base_lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.SGD(groups, momentum=0.9)
    apply_schedule(optimizer, SchedulePoint(lr=0.03, momentum=0.8, phase="test"))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.03)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.06)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.8)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(5e-4)


def test_one_training_step_changes_parameters() -> None:
    torch.manual_seed(1)
    model = CaffeLeNet()
    groups = build_caffe_parameter_groups(model, base_lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.SGD(groups, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.rand(8, 1, 28, 28)
    targets = torch.arange(8) % 10
    before = model.conv1.weight.detach().clone()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.conv1.weight.detach())

