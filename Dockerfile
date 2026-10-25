
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PRODUCTION_MODEL_PATH=/app/models/production

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/production/ ./models/production/

EXPOSE 8000

CMD ["uvicorn", "src.mlops_platform.api:app", "--host", "0.0.0.0", "--port", "8000"]
