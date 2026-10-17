from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJ_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJ_ROOT
    / "data"
    / "processed"
    / "p2_station_hour.csv"
)

FIGURES_DIR = PROJ_ROOT / "reports" / "figures"

EXPERIMENT_NAME = "mibici-equipo41"
RISK_THRESHOLD = 0.25
DATA_VERSION = "mibici_2026_06_to_2026_09_v1"

FEATURES = [
    "departures",
    "arrivals",
    "capacity",
    "net_flow",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "departures_lag1",
    "arrivals_lag1",
    "net_flow_lag1",
]

TARGET = "target_risk"


def load_data():
    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["fecha_hora"],
    )

    return df.dropna(
        subset=[
            "future_pressure",
            "departures_lag1",
            "arrivals_lag1",
            "net_flow_lag1",
        ]
    )


def temporal_split(df):
    train = df[df["fecha_hora"] < "2026-09-01"]

    test = df[
        (df["fecha_hora"] >= "2026-09-01")
        & (df["fecha_hora"] < "2026-10-01")
    ]

    return (
        train[FEATURES],
        test[FEATURES],
        train[TARGET],
        test[TARGET],
    )


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def evaluate(y_true, predictions):
    return {
        "test_accuracy": accuracy_score(y_true, predictions),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "test_precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "test_recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "test_f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


def run_experiment(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    parameters,
):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = evaluate(
        y_test,
        predictions,
    )

    print(f"\n{name}")
    print("-" * 50)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("\nMatriz de confusión:")
    print(matrix)

    # Artefacto
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    slug = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    figure_path = (
        FIGURES_DIR
        / f"p2_{slug}_confusion_matrix.png"
    )

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Normal", "Riesgo"],
    ).plot()

    plt.title(f"P2 - {name}")
    plt.tight_layout()
    plt.savefig(
        figure_path,
        dpi=150,
    )
    plt.close()

    # MLflow
    with mlflow.start_run(
        run_name=f"P2 - {name}"
    ):
        mlflow.set_tags(
            {
                "question": "P2",
                "task": "classification",
                "team": "41",
                "git_commit": get_git_commit(),
                "data_version": DATA_VERSION,
            }
        )

        mlflow.log_params(
               {
        "risk_threshold": RISK_THRESHOLD,
        "train_period": "2026-06_to_2026-08",
        "test_period": "2026-09",
        "data_version": DATA_VERSION,
        **parameters,
    }
        )

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            str(figure_path),
            artifact_path="confusion_matrix",
        )

    return {
        "model": name,
        **metrics,
    }


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = temporal_split(df)

    print(f"Train: {len(X_train):,}")
    print(f"Test:  {len(X_test):,}")

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    dummy_results = run_experiment(
        "DummyClassifier",
        dummy,
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "model": "DummyClassifier",
            "strategy": "most_frequent",
        },
    )

    logistic_results = run_experiment(
        "Logistic Regression",
        logistic,
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
    )

    results = pd.DataFrame(
        [
            dummy_results,
            logistic_results,
        ]
    )

    print("\nComparación final:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()