FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY credscore ./credscore
COPY backend ./backend
COPY dashboard ./dashboard
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit

ENV CREDSCORE_DATA_DIR=/app/dataset CREDSCORE_MODEL_DIR=/app/models
EXPOSE 8000 8501
# Default: the dashboard. docker-compose also runs the API from this image.
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
