import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# Rutas del proyecto
ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "processed" / "p3_station_features_current.csv"
DB_PATH = ROOT / "mlflow.db"

# Configuración
MODEL_NAME = "mibici-equipo41-p3-grupos"
EXPERIMENT_NAME = "mibici-equipo41"

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_registry_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment(EXPERIMENT_NAME)

# Cargar datos
if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

features = [
    "viajes_salida",
    "viajes_llegada",
    "duracion_promedio"
]

X = df[features].copy()
X = X.fillna(X.median())

# Entrenar modelo alternativo: KMeans con 3 clústeres
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    ))
])

pipeline.fit(X)
labels = pipeline.predict(X)

X_scaled = pipeline.named_steps["scaler"].transform(X)
score = silhouette_score(X_scaled, labels)

# Registrar ejecución y modelo
with mlflow.start_run(run_name="p3_kmeans_3_clusters_current") as run:

    mlflow.log_param("algorithm", "KMeans")
    mlflow.log_param("n_clusters", 3)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("feature_set", "counts_duration")
    mlflow.log_param("features", ",".join(features))
    mlflow.log_param("data_period", "2026_09")

    mlflow.log_metric("silhouette_score", score)

    signature = infer_signature(X, labels)

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="candidate_model",
        signature=signature,
        input_example=X.head(5)
    )

    print("Run ID:", run.info.run_id)
    print("Silhouette score:", score)

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_NAME
    )

    print("Nueva versión registrada:", registered.version)
    print("Estado:", registered.status)

# Verificar que no se modificó champion
client = MlflowClient()
model = client.get_registered_model(MODEL_NAME)

print("Alias champion:", model.aliases.get("champion"))
print("Entrenamiento alternativo finalizado.")
