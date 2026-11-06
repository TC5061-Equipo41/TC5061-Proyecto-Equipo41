FROM python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN grep -v '^-e' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY mibici/ mibici/

COPY models/p3_model/ /app/model/
ENV MODEL_URI=/app/model
ARG MODEL_VERSION=champion
ENV MODEL_VERSION=$MODEL_VERSION

EXPOSE 8000
CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]