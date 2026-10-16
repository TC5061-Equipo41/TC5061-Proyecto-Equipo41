

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def log_metrics(real, pred, prefix):
    metrics = {
                f"{prefix}_mae": mean_absolute_error(real, pred),
                f"{prefix}_rmse": np.sqrt(
                    mean_squared_error(real, pred)
                ),
                f"{prefix}_r2": r2_score(real, pred),
            }
    mlflow.log_metrics(metrics)
