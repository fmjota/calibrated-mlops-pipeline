from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valid_fraud_df() -> pd.DataFrame:
    """DataFrame pequeño y válido según el esquema de fraude."""
    n = 50
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.date_range("2020-01-01", periods=n, freq="h"),
            "amt": np.linspace(1.0, 500.0, n),
            "category": ["grocery_pos"] * n,
            "gender": ["M", "F"] * (n // 2),
            "state": ["CA"] * n,
            "city_pop": np.arange(n) * 1000 + 100,
            "lat": np.linspace(30.0, 45.0, n),
            "long": np.linspace(-120.0, -80.0, n),
            "merch_lat": np.linspace(30.0, 45.0, n),
            "merch_long": np.linspace(-120.0, -80.0, n),
            "is_fraud": [0, 1] * (n // 2),
        }
    )


@pytest.fixture(scope="session")
def spark():
    """SparkSession compartida para toda la sesión de tests (arranca el JVM una vez)."""
    from mlops_core.spark import get_spark

    session = get_spark(app_name="tests", shuffle_partitions=2)
    yield session
    session.stop()
