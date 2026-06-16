# Glosario

Términos de dominio y técnicos que usa el proyecto, para que la documentación se sostenga
sola.

- **Calibración.** Ajuste para que la probabilidad predicha refleje la frecuencia real:
  que entre los casos con proba "0.8" ocurra el evento ~80% de las veces. Métodos aquí:
  isotónica y Platt.
- **Isotónica (regresión isotónica).** Calibración no paramétrica que aprende una función
  monótona creciente de proba cruda → proba calibrada. Flexible; necesita datos suficientes.
- **Platt scaling (sigmoide).** Calibración paramétrica: una regresión logística sobre el
  score crudo. Más estable con pocos datos, asume forma sigmoide.
- **Brier score.** Error cuadrático medio entre la probabilidad y el resultado real (0/1).
  Menor es mejor. Mide **calidad de la probabilidad**, no solo el ranking.
- **PR-AUC (Average Precision).** Área bajo la curva precisión-recall. Métrica honesta con
  **desbalance** fuerte; el azar vale ≈ la prevalencia (aquí ~0.006), no 0.5.
- **ROC-AUC.** Área bajo la curva ROC; mide capacidad de **ranking**. Puede verse alta aun
  con clases muy desbalanceadas, por eso se acompaña con PR-AUC.
- **Desbalance.** Cuando una clase es rarísima (fraude ≈ 0.6%). Hace que accuracy engañe y
  exige métricas y manejo específicos (`is_unbalance`, PR-AUC, calibración).
- **Umbral (de bloqueo/intervención).** Punto de corte sobre la probabilidad para decidir
  (bloquear/no). Aquí se **deriva de una precisión objetivo**, no se fija a ojo.
- **Split temporal.** Partir train/calibración/test por orden de tiempo (no aleatorio):
  entrenar en el pasado y evaluar en el futuro. Evita fuga temporal.
- **Drift (data drift).** Que la distribución de producción se aleje de la de
  entrenamiento; el modelo degrada en silencio. Es detección de señales aplicada al modelo.
- **PSI (Population Stability Index).** Mide la **magnitud** del corrimiento de una
  distribución entre referencia y producción. Regla: <0.1 estable, 0.1–0.2 leve, >0.2 relevante.
- **KS (Kolmogorov–Smirnov).** Test estadístico de que dos muestras vienen de la misma
  distribución. p-valor bajo (<0.05) ⇒ distribuciones distintas.
- **Haversine.** Distancia sobre la esfera entre dos coordenadas (lat/long). Aquí: distancia
  cardholder↔comercio, una señal fuerte de fraude.
- **Pandera.** Librería de validación de DataFrames: define un esquema (tipos, rangos,
  categorías) y falla si los datos no lo cumplen.
- **MLflow.** Tracking de experimentos: registra hiperparámetros, métricas y artefactos
  para reproducibilidad.
- **LightGBM.** Implementación eficiente de gradient boosting sobre árboles. Fuerte en
  tabular; maneja categóricas y desbalance.
- **PySpark.** API de Python para Apache Spark; procesamiento distribuido para ETL a escala.
- **Núcleo agnóstico al dominio.** Código que no asume el caso de uso; se parametriza con un
  `config` + esquema por dominio (banca/salud/educación).
