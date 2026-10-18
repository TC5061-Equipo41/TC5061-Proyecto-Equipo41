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
        'datos_abiertos_2025_09.csv', 
    ]
STATIONS_DATA = 'nomenclatura_2026_09.csv'

DATA_METADATA = {
    'monthly': YEARLY_DATA,
    'stations': STATIONS_DATA,
    'data_range': {'from': '2025_09', 'to': '2026_09'}
}
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
