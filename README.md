# mibici

Proyecto de MLOps para el análisis de datos abiertos del sistema MiBici.

El proyecto contiene tres preguntas de negocio:

- **P1:** estimación de demanda.
- **P2:** clasificación de riesgo futuro en estaciones.
- **P3:** agrupamiento de estaciones por patrones de uso.

La infraestructura utiliza **MLflow Projects** para ejecutar la ingesta y los entrenamientos de forma reproducible, además de MLflow Tracking y Model Registry para registrar experimentos, artefactos y modelos.

---

## Project Organization

```text
├── LICENSE
├── Makefile
├── README.md
├── MLproject               <- Entry points reproducibles de MLflow
├── python_env.yaml         <- Entorno Python utilizado por MLflow Project
├── requirements.txt        <- Dependencias con versiones fijas
│
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
│
├── docs
│
├── models
│
├── notebooks
│
├── pyproject.toml
│
├── references
│
├── reports
│   └── figures
│
└── mibici
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── features.py
    ├── plots.py
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── train.py
    │   ├── p1
    │   ├── p2
    │   └── p3
    │
    ├── p1
    ├── p2
    └── P3
```

---

## Setup inicial

### 1. Crear el entorno virtual

Desde la raíz del repositorio:

```powershell
python -m venv .venv
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

En Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Instalar las dependencias

```powershell
pip install -r requirements.txt
```

El proyecto utiliza:

```text
Python 3.14.7
```

El entorno reproducible también se encuentra declarado en:

```text
python_env.yaml
```

---

## Preparación de los datos

Los datos abiertos deben descargarse desde el portal oficial de MiBici:

https://mibici.net/es/datos-abiertos/

Los archivos de viajes mensuales y la nomenclatura de estaciones deben colocarse en:

```text
data/raw/
```

La selección de archivos utilizada por el proyecto se encuentra centralizada en:

```text
mibici/config.py
```

Actualmente el rango configurado es:

```text
2025_10 a 2026_09
```

---

# Ejecución reproducible con MLflow Projects

El archivo:

```text
MLproject
```

define dos entry points principales:

```text
ingest
train
```

Esto permite ejecutar la preparación de datos y los entrenamientos desde MLflow sin invocar manualmente los scripts internos.

---

## 1. Iniciar el servidor local de MLflow

Desde una terminal independiente:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

La interfaz de MLflow estará disponible en:

```text
http://127.0.0.1:5000
```

En otra terminal, con el entorno virtual activo:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

---

## 2. Ejecutar la ingesta

Desde la raíz del repositorio:

```powershell
mlflow run . -e ingest --env-manager local
```

El entry point ejecuta:

```text
python -m mibici.dataset
```

La ingesta limpia y une los viajes con los metadatos de las estaciones.

Como resultado se generan, entre otros archivos:

```text
data/interim/joined_trips_stations-2025_10-2026_09.parquet
data/interim/data_manifest-2025_10-2026_09.json
```

El manifiesto registra información reproducible del dataset, incluyendo:

- rango temporal;
- archivos fuente;
- tamaño de los archivos;
- hashes SHA-256;
- número de filas;
- número y nombres de columnas;
- hash del dataset procesado;
- versión reproducible de los datos.

La versión de datos utilizada en los experimentos se obtiene directamente de este manifiesto.

---

# Entrenamiento

El entry point `train` recibe los parámetros:

```text
question
model
```

La ejecución general tiene la siguiente forma:

```powershell
mlflow run . -e train -P question=<pregunta> -P model=<modelo> --env-manager local
```

---

## P1

Para ejecutar P1:

```powershell
mlflow run