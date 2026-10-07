# Домашнее задание №2: воспроизведение экспериментов статьи про Super-Convergence

В проекте воспроизводятся все шесть экспериментов MNIST–LeNet из таблицы 2 статьи Лесли Смита и Николая Топина «Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates».

Код сравнивает два обычных расписания скорости обучения и четыре варианта политики `1cycle`. Описание результатов вынесено в отдельный файл [report.md](report.md). Данный файл содержит только инструкции по подготовке окружения и запуску экспериментов.

## Структура проекта

```text
hw2/
├── configs/                    # конфигурации шести экспериментов
├── data/                       # MNIST, загружается автоматически
├── docs/
│   └── EXPERIMENT_PROTOCOL.md  # протокол и допущения переноса
├── report_assets/              # иллюстрации для отчета
├── results/                    # результаты запусков, не сохраняются в Git
├── scripts/
│   ├── check_environment.py    # проверка программного окружения и GPU
│   ├── lr_range_test.py        # LR Range Test
│   ├── plot_mnist_suite.py     # построение общей сводки
│   ├── run_all.py              # короткая команда полного запуска
│   ├── run_mnist_suite.py      # запуск набора из шести опытов
│   └── train.py                # запуск одного опыта
├── src/superconvergence/       # модель, расписания и цикл обучения
├── tests/                      # модульные тесты
├── pytest.ini
├── requirements.txt
├── report.md
└── README.md
```

## Окружение

Эксперименты выполнялись в следующем окружении:

| Компонент | Версия или модель |
|---|---|
| Операционная система | Manjaro Linux |
| Python | 3.14.7 |
| PyTorch | 2.13.0+cu129 |
| torchvision | 0.28.0+cu129 |
| CUDA в сборке PyTorch | 12.9 |
| cuDNN | 92000 |
| Видеокарта | NVIDIA GeForce RTX 5070 Ti, 16 ГБ |

## Подготовка виртуального окружения

Все команды выполняются из корня репозитория `itmo-efficient-models-2026`.

Если виртуальное окружение уже создано:

```bash
source .venv/bin/activate
```

Если окружения еще нет - создать:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Установить общие зависимости:

```bash
python -m pip install -r hw2/requirements.txt
```

PyTorch и torchvision намеренно не включены в `requirements.txt`, чтобы установка не заменила уже работающую сборку с CUDA.

Для нового окружения установить согласованные сборки:

```bash
python -m pip install \
  'torch==2.13.0+cu129' \
  'torchvision==0.28.0+cu129' \
  --index-url https://download.pytorch.org/whl/cu129
```

Если PyTorch `2.13.0+cu129` уже установлен, а отсутствует только torchvision:

```bash
python -m pip install \
  --no-deps \
  'torchvision==0.28.0+cu129' \
  --index-url https://download.pytorch.org/whl/cu129
```

Проверить окружение и доступность видеокарты:

```bash
python hw2/scripts/check_environment.py
```

## Проверка кода

Запустить модульные тесты:

```bash
python -m pytest -c hw2/pytest.ini hw2/tests -q
```

Тесты проверяют архитектуру LeNet, число параметров, подготовку изображений, один шаг обучения, множитель скорости для смещений, три вида расписания и сбор итоговой таблицы.

## Короткая проверка конвейера

Перед полным обучением можно выполнить несколько пакетов данных:

```bash
python hw2/scripts/train.py \
  --config hw2/configs/baseline.json \
  --epochs 1 \
  --limit-train-batches 5 \
  --limit-test-batches 2 \
  --output-dir hw2/results/smoke
```

При первом запуске набор MNIST автоматически загружается в `hw2/data`.

## Проверка диапазона скорости обучения

Для выполнения `LR Range Test` запустить:

```bash
python hw2/scripts/lr_range_test.py
```

Тест постепенно повышает `Learning Rate` от `0,00001` до `0,3` и сохраняет таблицу и график в каталоге `hw2/results/lr_range`.

## Запуск всех шести экспериментов

Полный запуск с исходным случайным числом 42:

```bash
python hw2/scripts/run_all.py
```

Если часть опытов уже выполнена, готовые каталоги можно пропустить:

```bash
python hw2/scripts/run_all.py --skip-existing
```

Чтобы выполнить по три повтора каждого режима и получить среднее значение со стандартным отклонением:

```bash
python hw2/scripts/run_all.py --skip-existing --seeds 42 43 44
```

Чтобы заново построить только сводные файлы без обучения моделей:

```bash
python hw2/scripts/run_all.py --only-summary
```

## Запуск отдельного эксперимента

| Конфигурация | Режим |
|---|---|
| `baseline.json` | обратно-степенное расписание, 85 эпох |
| `baseline_step.json` | ступенчатое расписание, 85 эпох |
| `onecycle.json` | `1cycle`, 12 эпох |
| `onecycle_25.json` | `1cycle`, 25 эпох |
| `onecycle_50.json` | `1cycle`, 50 эпох |
| `onecycle_85.json` | `1cycle`, 85 эпох |

Пример отдельного запуска:

```bash
python hw2/scripts/train.py --config hw2/configs/onecycle_25.json
```

Другую случайную инициализацию и отдельный каталог можно указать параметрами командной строки:

```bash
python hw2/scripts/train.py \
  --config hw2/configs/onecycle.json \
  --seed 43 \
  --output-dir hw2/results/onecycle_seed43
```

## Выходные файлы

Каждый эксперимент создает:

| Файл | Назначение |
|---|---|
| `config.json` | фактически использованная конфигурация |
| `environment.json` | сведения о Python, PyTorch, CUDA и GPU |
| `history.csv` | потери, точность и время по эпохам |
| `schedule.csv` | скорость обучения и инерция на каждом шаге |
| `summary.json` | основные итоговые показатели |
| `model.pt` | веса модели и конфигурация |
| `training_curves.png` | графики одного запуска |

После полного набора создаются:

```text
hw2/results/mnist_runs.csv
hw2/results/mnist_summary.csv
hw2/results/mnist_comparison.png
```

Каталоги `data` и `results` исключены из Git. В репозитории сохраняются только небольшие иллюстрации, используемые в отчете.

## Дополнительная инфо

Точные параметры шести режимов, принятые допущения и соответствие исходной реализации Caffe описаны в [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).
