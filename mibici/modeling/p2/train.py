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

from mibici.config import FIGURES_DIR
from mibici.p2.dataset import (
    RISK_THRESHOLD,
    create_p2_dataset,
)


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


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def get_train_test_split(
    dataset: pd.DataFrame,
    date_range: dict,
):
    """
    Usa el último mes disponible como conjunto de prueba.
    En el rango actual:
    train = sep 2025 a ago 2026
    test = sep 2026
    """

    dataset = dataset.dropna(
        subset=[
            "future_pressure",
            "departures_lag1",
            "arrivals_lag1",
            "net_flow_lag1",
        ]
    ).copy()

    test_start = pd.to_datetime(
        date_range["to"],
        format="%Y_%m",
    )

    test_end = (
        test_start
        + pd.offsets.MonthBegin(1)
    )

    train = dataset[
        dataset["fecha_hora"] < test_start
    ]

    test = dataset[
        (dataset["fecha_hora"] >= test_start)
        & (dataset["fecha_hora"] < test_end)
    ]

    return (
        train[FEATURES],
        train[TARGET],
        test[FEATURES],
        test[TARGET],
        test_start,
        test_end,
    )


def evaluate(y_true, predictions):
    return {
        "test_accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "test_precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "test_recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "test_f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
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
    data_version,
    date_range,
    test_start,
    test_end,
):
    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    metrics = evaluate(
        y_test,
        predictions,
    )

    print(f"\n{name}")
    print("-" * 50)

    for metric, value in metrics.items():
        print(
            f"{metric}: "
            f"{value:.4f}"
        )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print("\nMatriz de confusión:")
    print(matrix)

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
        display_labels=[
            "Normal",
            "Riesgo",
        ],
    ).plot(
        values_format="d"
    )

    plt.title(
        f"P2 - {name}"
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=150,
    )

    plt.close()

    train_period = (
        f"{date_range['from']}"
        f"_to_"
        f"{(test_start - pd.offsets.MonthBegin(1)).strftime('%Y_%m')}"
    )

    test_period = (
        test_start.strftime(
            "%Y_%m"
        )
    )

    with mlflow.start_run(
        run_name=f"P2 - {name}"
    ):
        mlflow.set_tags(
            {
                "question": "P2",
                "task": "classification",
                "team": "41",
                "git_commit": get_git_commit(),
                "data_version": data_version,
                "date_range": (
                    f"{date_range['from']}"
                    f"_to_"
                    f"{date_range['to']}"
                ),
            }
        )

        mlflow.log_params(
            {
                "risk_threshold": RISK_THRESHOLD,
                "train_period": train_period,
                "test_period": test_period,
                "data_version": data_version,
                **parameters,
            }
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_artifact(
            str(figure_path),
            artifact_path="confusion_matrix",
        )

    return {
        "model": name,
        **metrics,
    }


def train_p2_model(
    original_data: pd.DataFrame,
    data_version: str,
    date_range: dict,
):
    """
    Entrena y registra en MLflow los modelos de P2.
    """

    print("\nConstruyendo dataset P2...")

    dataset = create_p2_dataset(
        original_data
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        test_start,
        test_end,
    ) = get_train_test_split(
        dataset,
        date_range,
    )

    print(
        f"Train: {len(X_train):,}"
    )

    print(
        f"Test:  {len(X_test):,}"
    )

    print(
        "\nDistribución target train:"
    )
    print(
        y_train.value_counts(
            normalize=True
        )
    )

    print(
        "\nDistribución target test:"
    )
    print(
        y_test.value_counts(
            normalize=True
        )
    )

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    logistic = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
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
        name="DummyClassifier",
        model=dummy,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        parameters={
            "model": "DummyClassifier",
            "strategy": "most_frequent",
        },
        data_version=data_version,
        date_range=date_range,
        test_start=test_start,
        test_end=test_end,
    )

    logistic_results = run_experiment(
        name="Logistic Regression",
        model=logistic,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        parameters={
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
        data_version=data_version,
        date_range=date_range,
        test_start=test_start,
        test_end=test_end,
    )

    results = pd.DataFrame(
        [
            dummy_results,
            logistic_results,
        ]
    )

    print(
        "\nComparación final P2:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    return results