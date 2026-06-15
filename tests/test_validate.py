"""El runner produce reportes trazables y maneja dominios desconocidos."""

from __future__ import annotations

import pytest

from mlops_core.schemas import get_schema
from mlops_core.validate import DataValidationError, validate_dataframe


def test_unknown_domain_raises():
    with pytest.raises(KeyError):
        get_schema("inexistente")


def test_report_is_traceable(valid_fraud_df):
    valid_fraud_df.loc[0, "amt"] = -10.0
    try:
        validate_dataframe(valid_fraud_df, "fraud")
    except DataValidationError as exc:
        summary = exc.report.summary()
        assert "FALLA" in summary
        assert "amt" in summary
        assert exc.report.failure_cases is not None
    else:
        pytest.fail("Se esperaba DataValidationError")


def test_ok_report_summary(valid_fraud_df):
    validate_dataframe(valid_fraud_df, "fraud")  # no levanta
