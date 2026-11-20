FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml README.md ./
COPY mibici ./mibici

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Modelos exportados del registry con `python -m mibici.api.scripts.export_model`; cada carpeta
# trae su archivo VERSION, así la API no depende del servidor de MLflow para predecir.
# Se copia la carpeta completa (y no cada modelo) para que el build también funcione en la CI de
# GitHub Actions, donde models/ solo trae .gitkeep: ahí se construye la imagen pero no se levanta
# la API. Para correr la API hay que exportar los modelos antes del build.
COPY models/ /app/models/

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
