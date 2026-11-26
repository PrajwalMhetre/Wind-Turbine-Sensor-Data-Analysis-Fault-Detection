"""Tests for data quality assessment and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_faults.config import PreprocessingConfig, default_sensor_limits
from wind_faults.data_quality import assess_data_quality
from wind_faults.preprocessing import preprocess_sensor_data


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-15 00:00:00",
                "2026-01-15 00:01:00",
                "2026-01-15 00:02:00",
            ],
            "wind_speed_mps": [8.0, 8.5, 9.0],
            "rotor_speed_rpm": [13.0, 13.5, 14.0],
            "generator_speed_rpm": [1040.0, 1080.0, 1120.0],
            "temperature_c": [52.0, 53.0, 54.0],
            "vibration_mm_s": [2.2, 2.3, 2.4],
            "power_kw": [500.0, 600.0, 700.0],
            "torque_nm": [4500.0, 5000.0, 5500.0],
            "pitch_angle_deg": [1.0, 1.0, 1.0],
            "status": ["OPERATING", "OPERATING", "OPERATING"],
        }
    )


def test_invalid_data_types_coerced_and_tracked() -> None:
    df = _base_df()
    df["temperature_c"] = df["temperature_c"].astype(object)
    df.loc[1, "temperature_c"] = "SENSOR_ERR"

    assessed, report = assess_data_quality(df, default_sensor_limits())
    assert report.non_numeric_coercions_per_column["temperature_c"] == 1
    assert np.isnan(assessed.loc[1, "temperature_c"])
    assert bool(assessed.loc[1, "dq_has_missing_sensor"]) is True


def test_missing_sensor_values_retained_by_default() -> None:
    df = _base_df()
    df.loc[1, "vibration_mm_s"] = np.nan

    assessed, report = assess_data_quality(df, default_sensor_limits())
    assert report.missing_values_per_column["vibration_mm_s"] == 1
    assert report.total_missing_values == 1

    prep = preprocess_sensor_data(assessed, PreprocessingConfig(missing_value_strategy="retain"))
    assert prep.values_imputed == 0
    assert np.isnan(prep.df.loc[1, "vibration_mm_s"])


def test_duplicate_timestamps_detected_and_handled() -> None:
    df = _base_df()
    df.loc[2, "timestamp"] = "2026-01-15 00:01:00"

    assessed, report = assess_data_quality(df, default_sensor_limits())
    assert report.duplicate_timestamp_count == 1
    assert report.duplicate_timestamps == ["2026-01-15T00:01:00"]
    assert bool(assessed.loc[1, "dq_is_duplicate_ts"]) is True
    assert bool(assessed.loc[2, "dq_is_duplicate_ts"]) is True

    prep = preprocess_sensor_data(
        assessed, PreprocessingConfig(duplicate_timestamp_strategy="keep_first")
    )
    assert prep.duplicates_removed == 1
    assert prep.rows_after == 2
    assert prep.df.loc[1, "power_kw"] == 600.0


def test_out_of_range_values_flagged_without_overwriting() -> None:
    df = _base_df()
    df.loc[0, "temperature_c"] = 250.0
    df.loc[2, "vibration_mm_s"] = -4.0

    assessed, report = assess_data_quality(df, default_sensor_limits())
    assert report.total_out_of_range_readings == 2
    assert report.out_of_range_counts_per_column["temperature_c"] == 1
    assert report.out_of_range_counts_per_column["vibration_mm_s"] == 1
    assert bool(assessed.loc[0, "dq_out_of_range"]) is True
    assert assessed.loc[0, "temperature_c"] == 250.0
