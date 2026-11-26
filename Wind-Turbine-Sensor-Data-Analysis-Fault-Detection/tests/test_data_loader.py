"""Tests for CSV loading and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wind_faults.data_loader import (
    REQUIRED_COLUMNS,
    DataLoadError,
    SchemaValidationError,
    load_sensor_csv,
)


def _sample_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-15 00:00:00", "2026-01-15 00:01:00"],
            "wind_speed_mps": [8.5, 9.0],
            "rotor_speed_rpm": [14.0, 14.5],
            "generator_speed_rpm": [1120.0, 1160.0],
            "temperature_c": [55.0, 56.0],
            "vibration_mm_s": [2.5, 2.6],
            "power_kw": [650.0, 720.0],
            "torque_nm": [5500.0, 5900.0],
            "pitch_angle_deg": [1.0, 1.1],
            "status": ["OPERATING", "OPERATING"],
        }
    )


def test_load_valid_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "valid.csv"
    _sample_valid_df().to_csv(csv_path, index=False)

    loaded = load_sensor_csv(csv_path)
    assert len(loaded) == 2
    assert list(loaded.columns) == list(REQUIRED_COLUMNS)
    assert loaded.loc[0, "power_kw"] == 650.0


def test_missing_required_columns_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_cols.csv"
    df = _sample_valid_df().drop(columns=["vibration_mm_s", "torque_nm"])
    df.to_csv(csv_path, index=False)

    with pytest.raises(SchemaValidationError, match="vibration_mm_s"):
        load_sensor_csv(csv_path)


def test_empty_csv_file_raises(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(DataLoadError, match="empty"):
        load_sensor_csv(empty_file)


def test_header_only_csv_raises(tmp_path: Path) -> None:
    header_only = tmp_path / "header_only.csv"
    header_only.write_text(",".join(REQUIRED_COLUMNS) + "\n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="0 data rows"):
        load_sensor_csv(header_only)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="not found"):
        load_sensor_csv(tmp_path / "does_not_exist.csv")
