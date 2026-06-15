"""Fábrica de SparkSession con descubrimiento de un JDK compatible.

Spark 4.x soporta Java 17/21, no Java 25. En esta máquina el Java por defecto es 25, así
que apuntamos `JAVA_HOME` a un JDK 17/21 dedicado **solo para el proceso de Spark**, sin
tocar el Java del sistema. Buscamos primero en el entorno y luego en un JDK instalado a
nivel de usuario en `~/.local/share/jvm`.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from pyspark.sql import SparkSession

_COMPATIBLE_MAJORS = (21, 17)


def _java_major(java_home: str) -> int | None:
    """Lee la versión mayor de Java desde el archivo `release` del JDK."""
    release = Path(java_home) / "release"
    if not release.exists():
        return None
    for line in release.read_text().splitlines():
        if line.startswith("JAVA_VERSION="):
            ver = line.split("=", 1)[1].strip().strip('"')
            parts = ver.split(".")
            # "21.0.11" -> 21 ; "1.8.0_xx" -> 8
            return int(parts[1]) if parts[0] == "1" else int(parts[0])
    return None


def find_compatible_java() -> str | None:
    """Devuelve la ruta de un JDK 17/21 compatible con Spark, o None."""
    env = os.environ.get("JAVA_HOME")
    if env and _java_major(env) in _COMPATIBLE_MAJORS:
        return env
    candidates: list[str] = []
    for major in _COMPATIBLE_MAJORS:
        candidates += glob.glob(str(Path.home() / f".local/share/jvm/jdk-{major}*"))
    for path in sorted(candidates, reverse=True):
        if _java_major(path) in _COMPATIBLE_MAJORS:
            return path
    return None


def ensure_java_home() -> str:
    """Fija JAVA_HOME a un JDK compatible; falla temprano y claro si no hay ninguno."""
    java_home = find_compatible_java()
    if java_home is None:
        raise RuntimeError(
            "No se encontró un JDK 17/21 compatible con Spark. Instala uno "
            "(p.ej. Temurin 21 en ~/.local/share/jvm) o exporta JAVA_HOME a un JDK 17/21."
        )
    os.environ["JAVA_HOME"] = java_home
    return java_home


def get_spark(app_name: str = "mlops-core", shuffle_partitions: int = 8) -> SparkSession:
    """Crea (o reutiliza) una SparkSession local con JAVA_HOME garantizado."""
    ensure_java_home()
    return (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
