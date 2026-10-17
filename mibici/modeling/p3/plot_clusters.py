
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer
from loguru import logger

from mibici.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    predictions_path: Path = PROCESSED_DATA_DIR / "p3_station_clusters.csv",
    figure_path: Path = FIGURES_DIR / "p3_station_map.png",
):
    """Plot MiBici stations geographically by P3 cluster."""
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}"
        )

    df = pd.read_csv(predictions_path)
    df = df.dropna(subset=["Latitud", "Longitud", "cluster"]).copy()

    if df.empty:
        raise ValueError("No stations have valid geographic coordinates.")

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cluster in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster]
        ax.scatter(
            subset["Longitud"],
            subset["Latitud"],
            s=22,
            alpha=0.75,
            label=f"Cluster {cluster} ({len(subset)} stations)",
        )

    ax.set_title("MiBici station clusters — P3")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    logger.success(f"Station map saved to {figure_path}")
    logger.info(f"Stations plotted: {len(df)}")


if __name__ == "__main__":
    app()
