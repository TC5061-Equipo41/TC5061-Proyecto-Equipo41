import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
DEPRECATED_STATION_IDS_REPLACEMENTS = {116: 248, 138: 249}
MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CURRENT_MONTH_DATA = ['datos_abiertos_2026_09.csv']
THREE_MONTH_DATA = ['datos_abiertos_2026_08.csv', 'datos_abiertos_2026_07.csv', 'datos_abiertos_2026_06.csv']
YEARLY_DATA = [
        'datos_abiertos_2026_09.csv', 
        'datos_abiertos_2026_08.csv', 
        'datos_abiertos_2026_07.csv', 
        'datos_abiertos_2026_06.csv',
        'datos_abiertos_2026_05.csv', 
        'datos_abiertos_2026_04.csv', 
        'datos_abiertos_2026_03.csv', 
        'datos_abiertos_2026_02.csv',
        'datos_abiertos_2026_01.csv',
        'datos_abiertos_2025_12.csv', 
        'datos_abiertos_2025_11.csv',
        'datos_abiertos_2025_10.csv',
    ]
STATIONS_DATA = 'nomenclatura_2026_09.csv'

DATA_METADATA = {
    'monthly': YEARLY_DATA,
    'stations': STATIONS_DATA,
    'data_range': {'from': '2025_10', 'to': '2026_09'}
}

# ---------------------------------------------------------------------------
# Entrega 2 · Infraestructura MLflow (equipo, tracking, versión de datos, defaults)
# ---------------------------------------------------------------------------
TEAM_NAME = "equipo41"
EXPERIMENT_NAME = f"mibici-{TEAM_NAME}"
SEED = 42

# Servidor de tracking local respaldado por SQLite (lo requiere el Model Registry):
#   mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
# Prioridad: MLFLOW_TRACKING_URI (estándar de MLflow; `mlflow run` la pasa al proceso hijo),
# luego MLFLOW_URI (nombre usado antes en el .env del equipo) y por último el servidor local.
MLFLOW_TRACKING_URI = (
    os.getenv("MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_URI") or "http://127.0.0.1:5000"
)
# Compatibilidad: el código que lee MLFLOW_URI (p. ej. modeling/p3/train.py) usa el mismo valor
os.environ.setdefault("MLFLOW_URI", MLFLOW_TRACKING_URI)

# Manifiesto de datos (único archivo de data/ que se versiona en Git)
MANIFEST_PATH = DATA_DIR / "manifest.json"
DATA_PIPELINE_VERSION = "1.0"  # súbelo si cambia la limpieza en dataset.py

# Valores por defecto del punto de entrada `train` del MLproject. En el MLproject todo vale
# "default" y se resuelve aquí, para no repetir valores. Cada train_pX_model puede leerlos.
TRAIN_DEFAULTS = {
    "p1": {
        "model": "ridge",
        "params": {
            "alpha": 0.1,
            "fit_intercept": True,
            "solver": "lsqr",
            "tol": 1e-4,
            "max_iter": 10_000,
        },
    },
    "p2": {
        "model": "logistic",
        "params": {"class_weight": "balanced", "max_iter": 1000, "random_state": SEED},
    },
    "p3": {
        "model": "kmeans",
        "params": {"min_k": 2, "max_k": 8, "random_state": SEED},
    },
}
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
