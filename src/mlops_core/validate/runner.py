"""Runner de validación: aplica un esquema Pandera y, si falla, levanta un error trazable.

"Fallar temprano y trazable" significa que cuando los datos rompen el contrato no
seguimos hacia el modelo: cortamos con un reporte que dice qué columna, qué chequeo y
cuántas filas fallaron, con ejemplos. Eso es justo lo que un entorno auditado exige.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera.errors as pa_errors

from mlops_core.schemas import get_schema


@dataclass
class ValidationReport:
    domain: str
    n_rows: int
    ok: bool
    n_failures: int = 0
    failure_cases: pd.DataFrame | None = None

    def summary(self, max_rows: int = 10) -> str:
        if self.ok:
            return f"[{self.domain}] OK — {self.n_rows} filas validadas."
        lines = [
            f"[{self.domain}] FALLA de validación — "
            f"{self.n_failures} casos en {self.n_rows} filas.",
        ]
        if self.failure_cases is not None and not self.failure_cases.empty:
            grouped = (
                self.failure_cases.groupby(["column", "check"], dropna=False)
                .size()
                .reset_index(name="n")
                .sort_values("n", ascending=False)
            )
            for _, row in grouped.head(max_rows).iterrows():
                lines.append(
                    f"  - columna={row['column']!r} chequeo={row['check']!r}: {row['n']} fila(s)"
                )
        return "\n".join(lines)


class DataValidationError(Exception):
    """Se levanta cuando los datos no cumplen el contrato del dominio."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.summary())


def validate_dataframe(df: pd.DataFrame, domain: str, *, lazy: bool = True) -> pd.DataFrame:
    """Valida `df` contra el esquema de `domain`. Devuelve el df coercido o levanta error.

    `lazy=True` junta todos los fallos en un solo reporte en vez de cortar en el primero.
    """
    schema = get_schema(domain)
    try:
        return schema.validate(df, lazy=lazy)
    except (pa_errors.SchemaError, pa_errors.SchemaErrors) as exc:
        cases = getattr(exc, "failure_cases", None)
        if isinstance(cases, pd.DataFrame):
            n_failures = len(cases)
        else:
            cases, n_failures = None, 1
        report = ValidationReport(
            domain=domain,
            n_rows=len(df),
            ok=False,
            n_failures=n_failures,
            failure_cases=cases,
        )
        raise DataValidationError(report) from exc
