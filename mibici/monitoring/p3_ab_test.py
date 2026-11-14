import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import silhouette_score, davies_bouldin_score
from mibici.monitoring.drift import rescale_counts

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

name = "mibici-equipo41-p3-grupos"

# Cargar las dos versiones registradas

import argparse

parser = argparse.ArgumentParser(
    description="A/B testing offline del modelo P3"
)
parser.add_argument("--champion-version", type=int, default=2)
parser.add_argument("--challenger-version", type=int, default=3)
args = parser.parse_args()

model_a = mlflow.sklearn.load_model(
    f"models:/{name}/{args.champion_version}"
)
model_b = mlflow.sklearn.load_model(
    f"models:/{name}/{args.challenger_version}"
)


# Mismo conjunto de evaluacion: octubre 2026
df = pd.read_csv(
    "data/interim/drift/p3_station_features_actual.csv"
)

features = [
    "viajes_salida",
    "viajes_llegada",
    "duracion_promedio"
]

assert list(model_a.feature_names_in_) == features
assert list(model_b.feature_names_in_) == features

# Ajustar conteos a la escala de cada entrenamiento
X_a = rescale_counts(df, 365 / 31)[features]
X_b = rescale_counts(df, 30 / 31)[features]

assert not X_a.isna().any().any()
assert not X_b.isna().any().any()

# Predicciones de los modelos ya entrenados
labels_a = model_a.predict(X_a)
labels_b = model_b.predict(X_b)

# Espacio comun para evaluar ambos agrupamientos
X_eval = model_a.named_steps["scaler"].transform(X_a)

print("\n=== A/B TESTING OFFLINE P3 ===")
print("Periodo: Octubre 2026")
print("Estaciones evaluadas:", len(df))

for version, labels in [
    (args.champion_version, labels_a),
    (args.challenger_version, labels_b)
]:
    print(f"\n--- Modelo version {version} ---")
    print("Silhouette:", silhouette_score(X_eval, labels))
    print("Davies-Bouldin:", davies_bouldin_score(X_eval, labels))
    print("Distribucion de clusters:")
    print(pd.Series(labels).value_counts().sort_index())

print("\nEvaluacion finalizada.")
