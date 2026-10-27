# PadelLens production image: API + web app; analysis runs on the Kaggle GPU (PADEL_BACKEND=kaggle).
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --legacy-peer-deps --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PADEL_BACKEND=kaggle PADEL_DATA_DIR=/data PADEL_KAGGLE_CFG=/data/.kaggle_cfg
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY padel_analyzer/ padel_analyzer/
COPY kaggle/ kaggle/
COPY scripts/ scripts/
COPY --from=web /web/dist web/dist
VOLUME /data
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8010/api/health || exit 1
CMD ["uvicorn", "padel_analyzer.api.main:app", "--host", "0.0.0.0", "--port", "8010", "--proxy-headers"]
