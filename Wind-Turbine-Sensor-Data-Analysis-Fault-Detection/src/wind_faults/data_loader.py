"""CSV data loading and initial schema validation for wind turbine sensor data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "wind_speed_mps",
    "rotor_speed_rpm",
    "generator_speed_rpm",
    "temperature_c",
    "vibration_mm_s",
    "power_kw",
    "torque_nm",
    "pitch_angle_deg",
    "status",
)

NUMERIC_SENSOR_COLUMNS: tuple[str, ...] = (
    "wind_speed_mps",
    "rotor_speed_rpm",
    "generator_speed_rpm",
    "temperature_c",
    "vibration_mm_s",
    "power_kw",
    "torque_nm",
    "pitch_angle_deg",
)


class DataLoadError(Exception):
    """Raised when a sensor CSV file cannot be read or is empty/malformed."""


class SchemaValidationError(ValueError):
    """Raised when required columns are missing from the loaded dataset."""


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure all required sensor columns are present in the DataFrame."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Input dataset is missing required column(s): {', '.join(missing)}"
        )


def load_sensor_csv(file_path: Path | str) -> pd.DataFrame:
    """Load wind turbine sensor data from a CSV file with graceful error handling.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        Raw DataFrame with original string/numeric values preserved.

    Raises:
        DataLoadError: If the file does not exist, is empty, or is malformed.
        SchemaValidationError: If required columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataLoadError(f"Sensor CSV file not found: {path}")
    if not path.is_file():
        raise DataLoadError(f"Expected a file path, got directory or special file: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Sensor CSV file is empty: {path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DataLoadError(f"Failed to parse sensor CSV file {path}: {exc}") from exc

    if df.empty and len(df.columns) == 0:
        raise DataLoadError(f"Sensor CSV file contains no columns or data: {path}")

    validate_required_columns(df)

    if df.empty:
        raise DataLoadError(f"Sensor CSV file has headers but 0 data rows: {path}")

    logger.info("Loaded %d rows and %d columns from %s", len(df), len(df.columns), path)
    return df
