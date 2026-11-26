"""Tests for causal rolling-median / MAD statistical anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_faults.anomaly_detection import detect_statistical_anomalies
from wind_faults.config import AnomalyConfig, ConfigurationError


def _baseline_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-15 00:00:00", periods=n, freq="1min"),
            "temperature_c": 50.0 + rng.normal(0.0, 0.3, size=n),
            "vibration_mm_s": 2.5 + rng.normal(0.0, 0.08, size=n),
            "rotor_speed_rpm": 14.0 + rng.normal(0.0, 0.15, size=n),
            "power_kw": 900.0 + rng.normal(0.0, 10.0, size=n),
        }
    )


def test_statistical_anomaly_detects_spike_without_lookahead() -> None:
    df = _baseline_df(40)
    # Inject spike at index 30
    df.loc[30, "vibration_mm_s"] = 9.5

    cfg = AnomalyConfig(rolling_window=20, min_periods=10, z_threshold=3.5)
    out, summary = detect_statistical_anomalies(df, cfg)

    assert bool(out.loc[30, "is_statistical_anomaly"]) is True
    assert out.loc[30, "anomaly_score_vibration_mm_s"] >= 3.5
    assert summary.total_anomalous_rows >= 1

    # Verify causality: changing future values (indices > 25) must NOT alter score at index 25
    df_future_modified = df.copy()
    df_future_modified.loc[35, "vibration_mm_s"] = 45.0
    out_mod, _ = detect_statistical_anomalies(df_future_modified, cfg)
    assert out.loc[25, "anomaly_score_vibration_mm_s"] == out_mod.loc[25, "anomaly_score_vibration_mm_s"]


def test_statistical_anomaly_handles_nans_and_constant_series() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-15 00:00:00", periods=25, freq="1min"),
            "temperature_c": [50.0] * 25,
            "vibration_mm_s": [2.0] * 25,
            "rotor_speed_rpm": [12.0] * 25,
            "power_kw": [0.0] * 25,
        }
    )
    df.loc[15, "temperature_c"] = np.nan

    cfg = AnomalyConfig(rolling_window=15, min_periods=5, z_threshold=3.5)
    out, summary = detect_statistical_anomalies(df, cfg)

    assert np.isnan(out.loc[15, "anomaly_score_temperature_c"])
    assert summary.total_anomalous_rows == 0
    assert summary.max_observed_score == 0.0


def test_invalid_rolling_window_configuration_raises() -> None:
    with pytest.raises(ConfigurationError, match="rolling_window"):
        AnomalyConfig(rolling_window=1, min_periods=1)

    with pytest.raises(ConfigurationError, match="min_periods"):
        AnomalyConfig(rolling_window=10, min_periods=15)
