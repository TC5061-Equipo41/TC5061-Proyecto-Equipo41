# 5. Análisis FinOps: costo por inferencia de MiBici

Para estimar cuánto cuesta servir una predicción, usamos la prueba escalonada de Locust. En 360 segundos, la API completó 13,142 predicciones: 9,180 en `/predict/p3` y 3,962 en `/predict/p1/abtest`. Ambos endpoints registraron cero fallas. La latencia p95 fue de 39 ms, por debajo del objetivo de 500 ms del equipo.

## Cálculo del costo por inferencia

Tomamos como referencia una instancia de Google Cloud Compute Engine `e2-standard-2`, con 2 vCPU, 8 GiB de RAM y una tarifa bajo demanda de USD 0.06701142 por hora. Suponemos que la API y MLflow comparten la instancia. El rendimiento se midió en el entorno local del equipo, por lo que el costo en la nube es una estimación de cómputo. El almacenamiento y la transferencia de datos quedan fuera del cálculo.

| Concepto | Operación | Resultado |
|---|---|---:|
| Costo de los seis minutos de prueba | 0.06701142 × (360 / 3,600) | USD 0.00670114 |
| Costo por inferencia exitosa | 0.00670114 / 13,142 | USD 0.000000510 |
| Costo equivalente por millón de inferencias | (0.00670114 / 13,142) × 1,000,000 | USD 0.51 |

Estimamos **USD 0.51 por millón de inferencias** si se mantiene la carga promedio de la prueba. Si la instancia permanece encendida en horarios con pocas solicitudes, el costo por predicción aumenta porque el mismo gasto por hora se reparte entre menos respuestas.

## Uso de recursos y opciones de optimización

- **Ajustar la memoria a la carga habitual.** Con 100 usuarios, la API atendió 79.1 peticiones por segundo con un p95 de 37 ms. En ese tramo, la API alcanzó 272.4 MiB y MLflow 2,447.4 MiB. La suma de ambos máximos fue de aproximadamente 2.66 GiB. Proponemos repetir la prueba con 2 vCPU y 4 GiB, medir la memoria disponible para el sistema operativo y comparar el costo por millón antes de elegir la configuración.
- **Escalar la API cuando la latencia lo justifique.** Al pasar de 220 a 270 usuarios, el rendimiento apenas subió de 146.1 a 148.2 peticiones por segundo, mientras que el p95 pasó de 440 a 780 ms y rebasó el objetivo. Con 220 usuarios, la API ya consumía en promedio el equivalente a 1.95 núcleos y MLflow registraba 0.8 % de CPU. Proponemos probar una segunda réplica de la API con CPU adicional y comparar ambas configuraciones con la misma carga. El criterio de decisión será el costo por millón de predicciones exitosas manteniendo el p95 en 500 ms o menos y los errores en 1 % o menos.

## Fuentes

- [Resultados de la prueba escalonada](./escalones_resumen.json)
- [Resultados de la prueba de estrés](./estres_resumen.json)
- [Tarifas de instancias de propósito general de Google Cloud](https://cloud.google.com/products/compute/pricing/general-purpose?hl=en)