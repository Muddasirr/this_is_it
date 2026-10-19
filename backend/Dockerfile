FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY wheels /wheels

# 1) CPU torch from local wheels (do not hit PyPI for CUDA torch)
# 2) Rest of deps prefer /wheels; ultralytics installed --no-deps then its non-torch deps via requirements
RUN pip install --no-cache-dir --default-timeout=1000 \
      --find-links=/wheels --no-index \
      torch torchvision \
    && pip install --no-cache-dir --default-timeout=1000 \
      --find-links=/wheels \
      -r requirements.txt \
    && pip install --no-cache-dir --default-timeout=1000 \
      --find-links=/wheels --no-deps \
      ultralytics-opencv-headless \
    && rm -rf /wheels

COPY . .

ENV PYTHONPATH=/app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
EXPOSE 8000
