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
| Versión y alias en el registro de la demo | `1`, `champion` |
| Algoritmo | Random Forest para regresión |
| Parámetros | `max_features=0.5`, `max_samples=0.3`, `min_samples_leaf=50`, `n_jobs=-1` |
| Run ID | `d6ac04bfac4b4b9498857f1a3cc61f93` |
| Versión de datos | `v1.0-cec7eeaf7d56` |
| Periodo de entrenamiento | Octubre de 2025 a agosto de 2026 |
| Periodo de prueba | Octubre de 2026 |

El registro de esta corrida indica `git_dirty=true`. Para identificar el modelo evaluado, consulta la corrida y el artefacto asociado a la versión 1 en el registro de la demo.

## Resultados de evaluación

| Métrica | Entrenamiento | Prueba de octubre de 2026 |
|---|---:|---:|
| Error absoluto medio (MAE) | 0.7566 | 1.1967 |
| Raíz del error cuadrático medio (RMSE) | 1.5312 | 2.4936 |
| Coeficiente de determinación (R²) | 0.5529 | 0.4396 |

En octubre, la diferencia absoluta entre la estimación y los viajes observados fue de 1.20 viajes por estación y hora en promedio.

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

- La ruta `POST /predict/p1` usa el modelo con alias `champion` y devuelve la predicción, el nombre registrado, el alias y la versión. La ruta `POST /predict/p1/abtest` registra qué variante respondió a cada solicitud.
- El modelo aprende de viajes realizados. Una estación sin bicicletas puede registrar pocas salidas aunque haya personas que quieran usarlas.
- La métrica promedio combina estaciones y horarios. Para tomar decisiones de redistribución, conviene revisar el error en las estaciones y horas de interés, junto con la disponibilidad de bicicletas.

## Evidencia

[Preparación de los datos de P1](mibici/p1/dataset.py) · [Entrenamiento](mibici/modeling/p1/train.py) · [Script de monitoreo](mibici/monitoring/p1_drift.py) · [Resumen del drift](reports/drift/p1_drift_summary.json) · [Reporte de Evidently](reports/drift/p1_drift.html)
