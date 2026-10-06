from __future__ import annotations

import math

import torch
from torch import nn


class CaffeLeNet(nn.Module):
    """LeNet из примера BVLC Caffe для изображений MNIST 28×28.

    Последовательность слоёв:
    Conv(1→20, 5×5) → MaxPool(2×2) → Conv(20→50, 5×5)
    → MaxPool(2×2) → Linear(800→500) → ReLU → Linear(500→10).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, 10)
        self.reset_caffe_parameters()

    def reset_caffe_parameters(self) -> None:
        """Приближение стандартного Xavier filler из Caffe.

        В исходном prototxt для всех обучаемых слоёв указан filler xavier.
        Стандартный Caffe Xavier filler использует FAN_IN и равномерное
        распределение в границах ±sqrt(3 / fan_in). Смещения равны нулю.
        """

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = math.sqrt(3.0 / fan_in)
                nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.pool1(self.conv1(inputs))
        outputs = self.pool2(self.conv2(outputs))
        outputs = torch.flatten(outputs, 1)
        outputs = self.relu(self.fc1(outputs))
        return self.fc2(outputs)


def build_caffe_parameter_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Создаёт группы параметров с lr_mult=1 для весов и lr_mult=2 для смещений.

    Именно такие множители записаны в исходном Caffe prototxt. Весовой спад
    сохраняется для обеих групп, поскольку decay_mult там отдельно не изменён.
    """

    weights: list[nn.Parameter] = []
    biases: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (biases if name.endswith("bias") else weights).append(parameter)

    return [
        {
            "params": weights,
            "lr": base_lr,
            "lr_scale": 1.0,
            "weight_decay": weight_decay,
        },
        {
            "params": biases,
            "lr": base_lr * 2.0,
            "lr_scale": 2.0,
            "weight_decay": weight_decay,
        },
    ]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

