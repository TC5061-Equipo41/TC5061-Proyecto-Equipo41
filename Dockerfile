FROM python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN grep -v '^-e' requirements.txt > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY mibici/ mibici/

# Modelos exportados con `python -m mibici.api.scripts.export_model`; cada carpeta trae su VERSION,
# así el contenedor no necesita el servidor de MLflow.
COPY models/p3_model/ /app/models/p3_model/
COPY models/p1_current/ /app/models/p1_current/
COPY models/p1_challenger/ /app/models/p1_challenger/

ENV P3_MODEL_NAME=mibici-equipo41-p3-grupos \
    P3_MODEL_ALIAS=champion \
    P3_MODEL_URI=/app/models/p3_model \
    P1_CURRENT_MODEL=mibici-equipo41-p1-demanda \
    P1_CURRENT_ALIAS=champion \
    P1_CURRENT_URI=/app/models/p1_current \
    P1_CHALLENGER_MODEL=mibici-equipo41-p1-demanda \
    P1_CHALLENGER_ALIAS=challenger \
    P1_CHALLENGER_URI=/app/models/p1_challenger

EXPOSE 8000
CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
