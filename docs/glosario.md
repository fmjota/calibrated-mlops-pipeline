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
- **Intervalo creíble (IC).** En estadística bayesiana: rango de valores que contiene el
  parámetro de interés con probabilidad `level` (ej. IC 90% contiene el verdadero valor de
  riesgo en el 90% de los casos). Más interpretable que el intervalo de confianza frecuentista.
- **NUTS (No-U-Turn Sampler).** Algoritmo de MCMC usado por PyMC para muestrear la posterior.
  Variante de HMC que se auto-calibra, produciendo muestras de alta calidad con bajo rechazo.
- **Posterior predictiva.** Distribución de predicciones para nuevas observaciones, integrando
  la incertidumbre sobre los parámetros del modelo. La media es la probabilidad calibrada;
  los cuantiles son los extremos del intervalo creíble.
- **IC Coverage (cobertura del intervalo).** Fracción de filas cuya probabilidad predicha cae
  dentro del intervalo creíble. Una cobertura cercana al nivel nominal (ej. 90%) indica que
  los intervalos están bien calibrados.
- **Subsample estratificado.** Para escalar NUTS a datasets medianos: se toma una muestra
  manteniendo la proporción de positivos y negativos del original.
- **Dropout (deserción estudiantil).** Target del dominio educación: estudiante que abandona
  la carrera sin graduarse. El modelo prioriza a quiénes intervenir.
- **Readmission (reingreso hospitalario).** Target del dominio salud: readmisión en <30 días
  (umbral clínico estándar, costosa y potencialmente prevenible).
