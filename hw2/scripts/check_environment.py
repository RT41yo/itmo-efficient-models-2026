#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {platform.python_version()}")
    try:
        import torch
    except Exception as error:
        raise SystemExit(f"Не удалось импортировать PyTorch: {error}") from error

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA в сборке PyTorch: {torch.version.cuda}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Видеокарта: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        tensor = torch.tensor([1.0], device="cuda") * 2
        print(f"Проверка вычисления на GPU: {tensor.item() == 2.0}")

    try:
        import torchvision

        print(f"torchvision: {torchvision.__version__}")
        from torchvision import datasets  # noqa: F401

        print("Импорт torchvision.datasets: успешно")
    except Exception as error:
        print(f"Ошибка torchvision: {error}")
        if str(torch.__version__).startswith("2.13.0+cu129"):
            print("Для этого окружения установите torchvision командой:")
            print(
                "python -m pip install --no-deps "
                "'torchvision==0.28.0+cu129' "
                "--index-url https://download.pytorch.org/whl/cu129"
            )
        raise SystemExit(1) from error

    try:
        import matplotlib

        print(f"Matplotlib: {matplotlib.__version__}")
    except Exception as error:
        print(f"Ошибка Matplotlib: {error}")
        raise SystemExit(1) from error

    print(f"Исполняемый файл Python: {sys.executable}")
    print("Окружение готово к запуску hw2")


if __name__ == "__main__":
    main()
