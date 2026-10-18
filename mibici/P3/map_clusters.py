
from pathlib import Path

import folium
import pandas as pd

from mibici.config import PROCESSED_DATA_DIR, REPORTS_DIR

INPUT_PATH = PROCESSED_DATA_DIR / "p3_station_clusters.csv"
OUTPUT_PATH = REPORTS_DIR / "figures" / "p3_station_clusters_map.html"


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(subset=["Latitud", "Longitud", "cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)

    center = [df["Latitud"].mean(), df["Longitud"].mean()]
    mapa = folium.Map(
    location=center,
    zoom_start=13,
    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attr="© OpenTopoMap contributors",
    )
    

    colors = {0: "blue", 1: "orange"}

    for _, row in df.iterrows():
        cluster = int(row["cluster"])
        nombre = str(row["Nombre"]) if pd.notna(row["Nombre"]) else "Sin nombre"

        folium.CircleMarker(
            location=[row["Latitud"], row["Longitud"]],
            radius=5,
            color=colors.get(cluster, "gray"),
            fill=True,
            fill_color=colors.get(cluster, "gray"),
            fill_opacity=0.75,
            tooltip=f"Estación {row['Estacion_Id']} - Cluster {cluster}",
            popup=(
                f"<b>{nombre}</b><br>"
                f"Cluster: {cluster}<br>"
                f"Viajes de salida: {row['viajes_salida']:,.0f}<br>"
                f"Viajes de llegada: {row['viajes_llegada']:,.0f}"
            ),
        ).add_to(mapa)

    leyenda = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 9999;
        background: white; padding: 10px 14px; border: 1px solid gray;
        border-radius: 5px;">
        <b>Clusters P3 — periodo anual</b><br>
        <span style="color: blue;">●</span> Cluster 0: mayor volumen<br>
        <span style="color: orange;">●</span> Cluster 1: menor volumen
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(leyenda))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapa.save(str(OUTPUT_PATH))

    print(f"Estaciones representadas: {len(df)}")
    print(f"Mapa guardado en: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
