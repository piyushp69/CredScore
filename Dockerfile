# 1. Build the React dashboard
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# 2. Serve the API and the built dashboard from one process
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY credscore ./credscore
COPY backend ./backend
COPY --from=web /web/dist ./web/dist

ENV CREDSCORE_DATA_DIR=/app/dataset CREDSCORE_MODEL_DIR=/app/models
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
