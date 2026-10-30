import json
import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
)
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

REGISTERED_MODEL_NAME = "mibici-equipo41-p2-riesgo"

LOGISTIC_C_VALUES = [
    0.01,
    0.1,
    10.0,
]

SELECTION_METRIC = "test_average_precision"


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
    train = meses anteriores
    test = último mes disponible
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


def build_logistic_model(c_value=1.0):
    """
    Crea el pipeline de regresión logística.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def get_scores(
    model,
    X_test,
    predictions,
):
    """
    Obtiene un score continuo para la curva
    Precision-Recall y Average Precision.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(
            X_test
        )[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(
            X_test
        )

    return predictions


def evaluate(
    y_true,
    predictions,
    scores,
):
    """
    Métricas de evaluación para P2.
    """

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
        "test_average_precision": average_precision_score(
            y_true,
            scores,
        ),
    }


def save_confusion_matrix(
    name,
    y_test,
    predictions,
):
    """
    Guarda la matriz de confusión.
    """

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    slug = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )

    figure_path = (
        FIGURES_DIR
        / f"p2_{slug}_confusion_matrix.png"
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Normal",
            "Riesgo",
        ],
    ).plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        f"P2 - {name}"
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=150,
    )

    plt.close(fig)

    return figure_path


def save_precision_recall_curve(
    name,
    y_test,
    scores,
):
    """
    Guarda la curva Precision-Recall.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    slug = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )

    figure_path = (
        FIGURES_DIR
        / f"p2_{slug}_precision_recall.png"
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    PrecisionRecallDisplay.from_predictions(
        y_test,
        scores,
        name=name,
        ax=ax,
    )

    ax.set_title(
        f"P2 - Precision-Recall - {name}"
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=150,
    )

    plt.close(fig)

    return figure_path


def log_run_config(
    parameters,
    data_version,
    date_range,
):
    """
    Guarda la configuración de la corrida
    como artefacto de MLflow.
    """

    config = {
        "question": "P2",
        "data_version": data_version,
        "date_range": date_range,
        "parameters": parameters,
        "git_commit": get_git_commit(),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = (
            f"{temp_dir}/p2_run_config.json"
        )

        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                config,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        mlflow.log_artifact(
            config_path,
            artifact_path="config",
        )


def get_project_parent_run_id():
    """
    Obtiene el ID de la corrida padre creada por MLflow Project.
    """

    return os.getenv(
        "MIBICI_PROJECT_PARENT_RUN_ID"
    )


def run_experiment(
    name,
    model,
    model_family,
    stage_of_work,
    X_train,
    y_train,
    X_test,
    y_test,
    parameters,
    data_version,
    date_range,
    test_start,
    test_end,
    nested=False,
):
    """
    Entrena, evalúa y registra una corrida de P2.
    """

    project_parent_run_id = (
        get_project_parent_run_id()
    )

    start_run_kwargs = {
        "run_name": f"P2 - {name}",
    }

    if nested:
        start_run_kwargs["nested"] = True

    elif project_parent_run_id:
        start_run_kwargs[
            "parent_run_id"
        ] = project_parent_run_id

    with mlflow.start_run(
        **start_run_kwargs
    ) as run:

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        scores = get_scores(
            model,
            X_test,
            predictions,
        )

        metrics = evaluate(
            y_test,
            predictions,
            scores,
        )

        print(
            f"\n{name}"
        )

        print(
            "-" * 50
        )

        for metric, value in metrics.items():
            print(
                f"{metric}: "
                f"{value:.4f}"
            )

        matrix = confusion_matrix(
            y_test,
            predictions,
        )

        print(
            "\nMatriz de confusión:"
        )

        print(
            matrix
        )

        confusion_path = (
            save_confusion_matrix(
                name,
                y_test,
                predictions,
            )
        )

        precision_recall_path = (
            save_precision_recall_curve(
                name,
                y_test,
                scores,
            )
        )

        train_period = (
            f"{date_range['from']}"
            f"_to_"
            f"{(
                test_start
                - pd.offsets.MonthBegin(1)
            ).strftime('%Y_%m')}"
        )

        test_period = (
            test_start.strftime(
                "%Y_%m"
            )
        )

        mlflow.set_tags(
            {
                "question": "P2",
                "task": "classification",
                "team": "41",
                "model_family": model_family,
                "stage_of_work": stage_of_work,
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

        # -------------------------------------------------
        # Registrar manifiesto de versión de datos
        # -------------------------------------------------

        manifest_path = (
            INTERIM_DATA_DIR
            / (
                f"data_manifest-"
                f"{date_range['from']}-"
                f"{date_range['to']}.json"
            )
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                "No se encontró el manifiesto de datos: "
                f"{manifest_path}"
            )

        mlflow.log_artifact(
            str(manifest_path),
            artifact_path="data",
        )

        # -------------------------------------------------
        # Registrar gráficas
        # -------------------------------------------------

        mlflow.log_artifact(
            str(confusion_path),
            artifact_path="plots",
        )

        mlflow.log_artifact(
            str(precision_recall_path),
            artifact_path="plots",
        )

        # -------------------------------------------------
        # Registrar configuración
        # -------------------------------------------------

        log_run_config(
            parameters=parameters,
            data_version=data_version,
            date_range=date_range,
        )

        # -------------------------------------------------
        # Registrar metadata del dataset de entrenamiento
        # -------------------------------------------------

        training_dataset = (
            mlflow.data.from_pandas(
                X_train,
                name="p2_training_features",
            )
        )

        mlflow.log_input(
            training_dataset,
            context="training",
        )

        # -------------------------------------------------
        # Signature e input example
        # -------------------------------------------------

        input_example = (
            X_train.head(5).copy()
        )

        example_prediction = (
            model.predict(
                input_example
            )
        )

        signature = infer_signature(
            input_example,
            example_prediction,
        )

        # -------------------------------------------------
        # Guardar modelo como artefacto MLflow
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        run_id = run.info.run_id

    return {
        "model": name,
        "model_family": model_family,
        "stage_of_work": stage_of_work,
        "run_id": run_id,
        **metrics,
    }

def run_logistic_c_search(
    X_train,
    y_train,
    X_test,
    y_test,
    data_version,
    date_range,
    test_start,
    test_end,
):
    """
    Búsqueda organizada de C para LogisticRegression.

    La corrida padre representa la búsqueda.
    Cada valor de C se registra como una corrida hija.
    """

    search_results = []

    project_parent_run_id = (
        get_project_parent_run_id()
    )

    search_run_kwargs = {
        "run_name": "P2 - Logistic C Search",
    }

    if project_parent_run_id:
        search_run_kwargs[
            "parent_run_id"
        ] = project_parent_run_id

    with mlflow.start_run(
        **search_run_kwargs
    ) as parent_run:

        mlflow.set_tags(
            {
                "question": "P2",
                "task": "classification",
                "team": "41",
                "model_family": "LogisticRegression",
                "stage_of_work": "search",
                "git_commit": get_git_commit(),
                "data_version": data_version,
            }
        )

        mlflow.log_param(
            "search_parameter",
            "C",
        )

        mlflow.log_param(
            "C_values",
            ",".join(
                str(value)
                for value
                in LOGISTIC_C_VALUES
            ),
        )

        for c_value in LOGISTIC_C_VALUES:

            logistic = build_logistic_model(
                c_value=c_value
            )

            result = run_experiment(
                name=(
                    "Logistic Regression "
                    f"C={c_value}"
                ),
                model=logistic,
                model_family="LogisticRegression",
                stage_of_work="search_candidate",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                parameters={
                    "model": "LogisticRegression",
                    "C": c_value,
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "random_state": 42,
                },
                data_version=data_version,
                date_range=date_range,
                test_start=test_start,
                test_end=test_end,
                nested=True,
            )

            result["C"] = c_value
            result[
                "search_parent_run_id"
            ] = parent_run.info.run_id

            search_results.append(
                result
            )

        best_search_result = max(
            search_results,
            key=lambda result: result[
                SELECTION_METRIC
            ],
        )

        mlflow.log_metric(
            "best_test_average_precision",
            best_search_result[
                SELECTION_METRIC
            ],
        )

        mlflow.log_param(
            "best_C",
            best_search_result["C"],
        )

        search_dataframe = (
            pd.DataFrame(
                search_results
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            search_path = (
                f"{temp_dir}/"
                "p2_logistic_c_search.csv"
            )

            search_dataframe.to_csv(
                search_path,
                index=False,
            )

            mlflow.log_artifact(
                search_path,
                artifact_path="search",
            )

    return search_results

def register_champion(
    best_result,
    data_version,
):
    """
    Registra el mejor modelo P2 y asigna
    el alias champion.
    """

    run_id = best_result[
        "run_id"
    ]

    model_uri = (
        f"runs:/{run_id}/model"
    )

    print(
        "\nRegistrando mejor modelo P2..."
    )

    registered_model = (
        mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )
    )

    client = MlflowClient()

    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description=(
            "Modelo de clasificación de riesgo "
            "para la pregunta P2 del Equipo 41. "
            "Predice riesgo de presión futura "
            "en estaciones MiBici."
        ),
    )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        description=(
            f"Modelo seleccionado usando "
            f"{SELECTION_METRIC}. "
            f"Run de origen: {run_id}."
        ),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="question",
        value="P2",
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="data_version",
        value=data_version,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="git_commit",
        value=get_git_commit(),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="source_run_id",
        value=run_id,
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    champion_uri = (
        f"models:/{REGISTERED_MODEL_NAME}"
        "@champion"
    )

    # Verificación requerida por la actividad.
    mlflow.sklearn.load_model(
        champion_uri
    )

    print(
        f"\nModelo registrado: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Versión: "
        f"{registered_model.version}"
    )

    print(
        "Alias: champion"
    )

    print(
        f"URI verificada: "
        f"{champion_uri}"
    )

    return registered_model.version


def train_p2_model(
    original_data: pd.DataFrame,
    data_version: str,
    date_range: dict,
    model: str = "all",
):
    """
    Ejecuta los experimentos de P2.

    model permite ejecutar:
    - all
    - dummy
    - logistic
    - logistic_search
    - random_forest
    - hist_gradient_boosting
    """

    valid_models = {
        "all",
        "dummy",
        "logistic",
        "logistic_search",
        "random_forest",
        "hist_gradient_boosting",
    }

    if model not in valid_models:
        raise ValueError(
            f"Modelo P2 no válido: {model}. "
            f"Opciones: {sorted(valid_models)}"
        )

    print(
        "\nConstruyendo dataset P2..."
    )

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

    all_results = []

    # -------------------------------------------------
    # BASELINE 1: DummyClassifier
    # -------------------------------------------------

    if model in {
        "all",
        "dummy",
    }:

        dummy = DummyClassifier(
            strategy="most_frequent"
        )

        dummy_result = run_experiment(
            name="DummyClassifier",
            model=dummy,
            model_family="DummyClassifier",
            stage_of_work="baseline",
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

        all_results.append(
            dummy_result
        )

    # -------------------------------------------------
    # BASELINE 2: Logistic Regression
    # -------------------------------------------------

    if model in {
        "all",
        "logistic",
    }:

        baseline_logistic = (
            build_logistic_model(
                c_value=1.0
            )
        )

        baseline_logistic_result = (
            run_experiment(
                name=(
                    "Logistic Regression "
                    "Baseline"
                ),
                model=baseline_logistic,
                model_family=(
                    "LogisticRegression"
                ),
                stage_of_work="baseline",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                parameters={
                    "model": (
                        "LogisticRegression"
                    ),
                    "C": 1.0,
                    "class_weight": (
                        "balanced"
                    ),
                    "max_iter": 1000,
                    "random_state": 42,
                },
                data_version=data_version,
                date_range=date_range,
                test_start=test_start,
                test_end=test_end,
            )
        )

        all_results.append(
            baseline_logistic_result
        )

    # -------------------------------------------------
    # HYPERPARAMETER SEARCH
    # -------------------------------------------------

    if model in {
        "all",
        "logistic_search",
    }:

        logistic_search_results = (
            run_logistic_c_search(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                data_version=data_version,
                date_range=date_range,
                test_start=test_start,
                test_end=test_end,
            )
        )

        all_results.extend(
            logistic_search_results
        )

    # -------------------------------------------------
    # RANDOM FOREST
    # -------------------------------------------------

    if model in {
        "all",
        "random_forest",
    }:

        random_forest = (
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        )

        random_forest_result = (
            run_experiment(
                name="Random Forest",
                model=random_forest,
                model_family=(
                    "RandomForestClassifier"
                ),
                stage_of_work="candidate",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                parameters={
                    "model": (
                        "RandomForestClassifier"
                    ),
                    "n_estimators": 100,
                    "max_depth": 12,
                    "min_samples_leaf": 5,
                    "class_weight": (
                        "balanced"
                    ),
                    "random_state": 42,
                },
                data_version=data_version,
                date_range=date_range,
                test_start=test_start,
                test_end=test_end,
            )
        )

        all_results.append(
            random_forest_result
        )

    # -------------------------------------------------
    # HIST GRADIENT BOOSTING
    # -------------------------------------------------

    if model in {
        "all",
        "hist_gradient_boosting",
    }:

        hist_gradient_boosting = (
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=100,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=42,
            )
        )

        hist_gradient_result = (
            run_experiment(
                name=(
                    "Hist Gradient Boosting"
                ),
                model=hist_gradient_boosting,
                model_family=(
                    "HistGradientBoostingClassifier"
                ),
                stage_of_work="candidate",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                parameters={
                    "model": (
                        "HistGradientBoostingClassifier"
                    ),
                    "learning_rate": 0.1,
                    "max_iter": 100,
                    "max_leaf_nodes": 31,
                    "l2_regularization": 1.0,
                    "class_weight": (
                        "balanced"
                    ),
                    "random_state": 42,
                },
                data_version=data_version,
                date_range=date_range,
                test_start=test_start,
                test_end=test_end,
            )
        )

        all_results.append(
            hist_gradient_result
        )

    # -------------------------------------------------
    # Resultados
    # -------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    if results.empty:
        raise RuntimeError(
            "No se generaron resultados para P2."
        )

    results[
        "is_champion"
    ] = False

    # -------------------------------------------------
    # Flujo completo: comparar candidatos
    # -------------------------------------------------

    if model == "all":

        candidate_results = [
            result
            for result in all_results
            if result["stage_of_work"]
            in {
                "candidate",
                "search_candidate",
            }
        ]

        best_result = max(
            candidate_results,
            key=lambda result: result[
                SELECTION_METRIC
            ],
        )

        results[
            "is_champion"
        ] = (
            results["run_id"]
            == best_result["run_id"]
        )

        print(
            "\nComparación final P2:"
        )

        display_columns = [
            "model",
            "stage_of_work",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_average_precision",
            "is_champion",
        ]

        print(
            results[
                display_columns
            ].to_string(
                index=False
            )
        )

        print(
            "\nMejor candidato P2:"
        )

        print(
            best_result["model"]
        )

        print(
            f"{SELECTION_METRIC}: "
            f"{best_result[SELECTION_METRIC]:.4f}"
        )

        champion_version = (
            register_champion(
                best_result=best_result,
                data_version=data_version,
            )
        )

        print(
            f"\nChampion P2 versión: "
            f"{champion_version}"
        )

    # -------------------------------------------------
    # Reproducción directa del modelo ganador
    # -------------------------------------------------

    elif model == "hist_gradient_boosting":

        best_result = all_results[0]

        results[
            "is_champion"
        ] = True

        print(
            "\nReproduciendo modelo ganador P2:"
        )

        print(
            best_result["model"]
        )

        print(
            f"{SELECTION_METRIC}: "
            f"{best_result[SELECTION_METRIC]:.4f}"
        )

        champion_version = (
            register_champion(
                best_result=best_result,
                data_version=data_version,
            )
        )

        print(
            f"\nChampion P2 versión: "
            f"{champion_version}"
        )

    else:

        print(
            "\nResultado P2:"
        )

        display_columns = [
            "model",
            "stage_of_work",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_average_precision",
        ]

        print(
            results[
                display_columns
            ].to_string(
                index=False
            )
        )

    return results