# Video Genre Classifier

Классификатор жанров YouTube-видео на основе визуальных признаков. Модель обучена на датасете [YouTube-8M](https://research.google.com/youtube8m/) и определяет жанр из 12 категорий по первой минуте видео.

**Жанры:** Animals, Animation, Beauty, Dance, Film, Food, Gaming, Music, Performance, Sports, Tech, Vehicles

## Результаты

| Модель     | Параметры | Test Accuracy | Macro F1 |
|------------|-----------|---------------|----------|
| FlattenMLP | 35.9M     | 0.743         | 0.740    |
| FrameRNN   | 1.09M     | **0.785**     | **0.784**|
| GatedRNN   | 1.09M     | 0.779         | 0.778    |

Лучшая модель — **FrameRNN** (двунаправленный GRU поверх признаков Inception-v3).

## Как запустить

### Способ 1 — Веб-интерфейс через Docker

Требуется: [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
cd deploy
docker compose build
docker compose up web
```

Открыть в браузере: `http://localhost:7860`

### Способ 2 — CLI через Python

Требуется: Python 3.12+, ffmpeg

```bash
pip install -e .
python scripts/predict.py "https://www.youtube.com/shorts/..."
```

В обоих случаях первый запуск дольше (~2 мин): скачивается Inception-v3 (~100 MB) и кешируется на будущее.

## Структура проекта

```
pipeline/
  01_parse_frame_level.ipynb   # парсинг TFRecord → numpy
  02_eda.ipynb                 # разведочный анализ
  03_preprocessing.ipynb       # нормализация, балансировка
  04_models.ipynb              # определение архитектур
  05_training.ipynb            # обучение и отбор модели
  06_inference.ipynb           # оценка на тесте

models/
  best_rnn.pt                  # веса FrameRNN
  norm_stats.npz               # статистики нормализации
  config.json                  # конфиг модели

scripts/
  predict.py                   # CLI-инференс
  app.py                       # Gradio веб-приложение
  feature_extractor.py         # извлечение признаков (Inception-v3 + PCA)

deploy/
  Dockerfile
  docker-compose.yml

docs/
  report.pdf                   # финальный отчёт ClearML
```

## Технический стек

- **Признаки:** Inception-v3 (1024D RGB) + VGGish (128D Audio) из YouTube-8M
- **Модель:** двунаправленный GRU, 1.09M параметров
- **Фреймворки:** PyTorch, TensorFlow (feature extractor), Gradio
- **Трекинг экспериментов:** ClearML
