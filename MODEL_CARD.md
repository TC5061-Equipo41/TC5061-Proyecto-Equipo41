# Model Card de P1 · Demanda por estación y hora

**Proyecto MiBici Guadalajara · Equipo 41 · TC5061**

## Qué hace el modelo

P1 estima cuántos viajes salen de una estación en una hora determinada. Cada fila representa una estación durante una hora, incluidas las horas con cero salidas. La estimación sirve para analizar patrones de demanda y apoyar la planeación operativa.

| Variable | Significado |
|---|---|
| `Origen_Id` | Estación de salida |
| `Hora` | Hora del día |
| `Dia` | Día de la semana |
| `Viajes` | Número observado de salidas en esa estación y hora. Es el valor que el modelo aprende a estimar. |

## Versión y trazabilidad

| Elemento | Valor registrado en MLflow |
|---|---|
| Modelo | `mibici-equipo41-p1-demanda` |
| Versión y alias en el registro local consultado | `1`, `champion` |
| Algoritmo | Regresión Ridge, `alpha=0.1`, `solver=lsqr`, con intercepto |
| Corrida | `1796edc3b1bf4ecaa87902455d402b88` |
| Versión de datos | `v1.0-cec7eeaf7d56` |
| Periodo general de los datos de la corrida | Octubre de 2025 a septiembre de 2026 |
| Commit de referencia | `5f6dcbc9a97e62633da49b1b4d5a1ebbfd6313a3` |

MLflow registró `git_dirty=true`: había cambios locales al ejecutar el entrenamiento. Para identificar el modelo evaluado se debe consultar la corrida y su artefacto registrado, además del commit.

## Resultados de evaluación

| Métrica | Entrenamiento | Prueba |
|---|---:|---:|
| Error absoluto medio (MAE) | 0.9414 | 0.9513 |
| Raíz del error cuadrático medio (RMSE) | 1.7715 | 1.7690 |
| Coeficiente de determinación (R²) | 0.4034 | 0.4069 |

En prueba, la diferencia absoluta entre la demanda estimada y la observada fue, en promedio, de 0.95 viajes por estación y hora. La etiqueta `date_range` describe la cobertura general de la corrida. El periodo exacto de prueba queda pendiente de verificar.

## Monitoreo de octubre de 2026

El análisis con Evidently comparó octubre de 2025 a agosto de 2026 como referencia frente a octubre de 2026. Se usó el corte de referencia definido para el champion en el diseño A/B del proyecto. Se compararon las tres entradas y, por separado, la demanda observada.

| Variable | Valor de drift | Umbral | ¿Superó el umbral? |
|---|---:|---:|---|
| Estación (`Origen_Id`) | 0.0209 | 0.1 | No |
| Hora (`Hora`) | 0.0040 | 0.1 | No |
| Día (`Dia`) | 0.0373 | 0.1 | No |
| Demanda observada (`Viajes`) | 0.3205 | 0.1 | Sí |

En octubre cambió la distribución del número de salidas por estación y hora. Para saber cómo afectó ese cambio a la precisión, se requiere comparar predicciones con viajes observados en octubre. El valor de drift por sí solo tampoco indica si la demanda subió o bajó.

## Uso y limitaciones

- La API expone `POST /predict/p1/abtest` y devuelve la estimación junto con el alias de la variante que respondió. El registro local consultado contiene únicamente la versión 1. La comparación con el challenger debe sustentarse con la evidencia del entorno donde se ejecutó el A/B.
- El modelo aprende de viajes realizados. Una estación sin bicicletas puede registrar pocas salidas aunque haya personas que quieran usarlas.
- La métrica promedio combina estaciones y horarios. Para tomar decisiones de redistribución, conviene revisar el error en las estaciones y horas de interés, junto con la disponibilidad de bicicletas.

## Evidencia

[Preparación de los datos de P1](mibici/p1/dataset.py) · [Entrenamiento](mibici/modeling/p1/train.py) · [Script de monitoreo](mibici/monitoring/p1_drift.py) · [Resumen del drift](reports/drift/p1_drift_summary.json) · [Reporte de Evidently](reports/drift/p1_drift.html)
