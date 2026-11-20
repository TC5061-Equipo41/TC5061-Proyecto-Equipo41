import csv

from mibici.api.models.request.p1_prediction_request import P1PredictionRequest
from mibici.api.models.response.p1_prediction_response import P1PredictionResponse
from mibici.config import PROJ_ROOT

LOG_PATH = PROJ_ROOT / "logs" / "p1_predictions.csv"


def log_p1_prediction(request: P1PredictionRequest, response: P1PredictionResponse ) -> None:
    """Agrega una fila al CSV: atributos de la petición, predicción y modelo que la hizo."""
    row = {**request.model_dump(), **response.model_dump()}
    LOG_PATH.parent.mkdir(exist_ok=True)
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)
