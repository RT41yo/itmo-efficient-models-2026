from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import functional as transform_functional


class ToCaffeTensor:
    """Преобразует пиксели MNIST так же, как scale: 1/256 в Caffe."""

    def __call__(self, image) -> torch.Tensor:
        return transform_functional.pil_to_tensor(image).to(torch.float32).div_(256.0)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    test: DataLoader


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_mnist_loaders(
    root: Path,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    download: bool,
    drop_last: bool,
    seed: int,
    caffe_pixel_scale: bool = True,
) -> DataLoaders:
    if not caffe_pixel_scale:
        raise ValueError(
            "В этом воспроизведении ожидается масштаб Caffe: значение пикселя / 256"
        )

    transform = ToCaffeTensor()
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=download)

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return DataLoaders(train=train_loader, test=test_loader)

