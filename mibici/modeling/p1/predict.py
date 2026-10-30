import os

from dotenv import load_dotenv
import mlflow
import numpy as np
import pandas as pd

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

model = mlflow.sklearn.load_model("models:/mibici-equipo41-p1-demanda@champion")

X = pd.DataFrame([{"Origen_Id": 12, "Hora": 8, "Dia": "Monday"}])
print(np.maximum(0, model.predict(X)))