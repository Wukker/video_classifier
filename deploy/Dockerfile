FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    nodejs ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    tensorflow \
    opencv-python-headless \
    yt-dlp \
    numpy \
    six \
    gradio

COPY scripts/ ./scripts/
COPY models/   ./models/

ENTRYPOINT ["python", "scripts/predict.py"]
