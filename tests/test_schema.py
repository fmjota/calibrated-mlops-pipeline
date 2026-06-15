"""La validación acepta datos buenos y rechaza datos que rompen el contrato."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_core.validate import DataValidationError, validate_dataframe


def test_valid_data_passes(valid_fraud_df):
    out = validate_dataframe(valid_fraud_df, "fraud")
    assert len(out) == len(valid_fraud_df)


def test_negative_amount_rejected(valid_fraud_df):
    valid_fraud_df.loc[0, "amt"] = -5.0
    with pytest.raises(DataValidationError) as exc:
        validate_dataframe(valid_fraud_df, "fraud")
    assert "amt" in exc.value.report.summary()


def test_unknown_category_rejected(valid_fraud_df):
    valid_fraud_df.loc[0, "category"] = "crypto"
    with pytest.raises(DataValidationError) as exc:
        validate_dataframe(valid_fraud_df, "fraud")
    assert "category" in exc.value.report.summary()


def test_null_state_rejected(valid_fraud_df):
    valid_fraud_df.loc[0, "state"] = None
    with pytest.raises(DataValidationError):
        validate_dataframe(valid_fraud_df, "fraud")


def test_out_of_domain_label_rejected(valid_fraud_df):
    valid_fraud_df.loc[0, "is_fraud"] = 5
    with pytest.raises(DataValidationError) as exc:
        validate_dataframe(valid_fraud_df, "fraud")
    assert "is_fraud" in exc.value.report.summary()


def test_coordinate_out_of_range_rejected(valid_fraud_df):
    valid_fraud_df.loc[0, "lat"] = 200.0
    with pytest.raises(DataValidationError):
        validate_dataframe(valid_fraud_df, "fraud")


def test_multiple_failures_collected(valid_fraud_df):
    valid_fraud_df.loc[0, "amt"] = -1.0
    valid_fraud_df.loc[1, "category"] = "crypto"
    valid_fraud_df.loc[np.arange(3), "is_fraud"] = 9
    with pytest.raises(DataValidationError) as exc:
        validate_dataframe(valid_fraud_df, "fraud")
    # lazy=True junta todos los fallos en un solo reporte
    assert exc.value.report.n_failures >= 3
