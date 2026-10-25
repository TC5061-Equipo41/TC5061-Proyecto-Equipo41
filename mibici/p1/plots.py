from pathlib import Path

import matplotlib.pyplot as plt
import mlflow


def plot_prediction_vs_real_data(y_real, y_pred):
    x = range(len(y_real))

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(x, y_real.to_numpy(), color="blue", label="Viajes reales", s=15)
    ax.scatter(x, y_pred, color="orange", label="Viajes predichos", s=15, alpha=0.6)
    ax.set_xlabel("Observación del conjunto de prueba")
    ax.set_ylabel("Viajes")
    ax.set_title("Valores reales y predichos")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "test_real_vs_predicted.png"

    try:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    mlflow.log_artifact(str(plot_path), artifact_path="evaluation")