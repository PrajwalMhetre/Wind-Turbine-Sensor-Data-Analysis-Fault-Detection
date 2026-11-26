"""Data-quality assessment for wind turbine sensor datasets."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from wind_faults.config import SensorRange
from wind_faults.data_loader import NUMERIC_SENSOR_COLUMNS, validate_required_columns

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Structured summary of data-quality checks on a sensor DataFrame."""

    total_rows: int
    valid_timestamp_rows: int
    invalid_timestamp_rows: int
    duplicate_timestamp_count: int
    duplicate_timestamps: list[str] = field(default_factory=list)
    missing_values_per_column: dict[str, int] = field(default_factory=dict)
    total_missing_values: int = 0
    non_numeric_coercions_per_column: dict[str, int] = field(default_factory=dict)
    out_of_range_counts_per_column: dict[str, int] = field(default_factory=dict)
    total_out_of_range_readings: int = 0
    is_valid_schema: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return asdict(self)


def assess_data_quality(
    df: pd.DataFrame,
    sensor_limits: dict[str, SensorRange],
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate types, parse timestamps, and flag duplicates, missingness, and range violations.

    Preserves original raw DataFrame values; adds `dq_*` indicator columns and
    coerces numeric sensor columns cleanly while tracking any non-numeric entries.
    """
    validate_required_columns(df)
    assessed = df.copy()
    notes: list[str] = []

    # 1. Timestamp parsing
    parsed_ts = pd.to_datetime(assessed["timestamp"], errors="coerce")
    invalid_ts_mask = parsed_ts.isna()
    invalid_ts_count = int(invalid_ts_mask.sum())
    assessed["timestamp"] = parsed_ts
    assessed["dq_invalid_timestamp"] = invalid_ts_mask

    if invalid_ts_count > 0:
        notes.append(f"Detected {invalid_ts_count} row(s) with unparseable timestamps.")

    # 2. Duplicate timestamps (among valid timestamps)
    valid_ts_mask = ~invalid_ts_mask
    dup_mask = pd.Series(False, index=assessed.index)
    dup_mask.loc[valid_ts_mask] = assessed.loc[valid_ts_mask, "timestamp"].duplicated(keep=False)
    dup_extra_count = int(
        assessed.loc[valid_ts_mask, "timestamp"].duplicated(keep="first").sum()
    )
    dup_ts_strings = sorted(
        {ts.isoformat() for ts in assessed.loc[dup_mask, "timestamp"].dropna()}
    )
    assessed["dq_is_duplicate_ts"] = dup_mask

    if dup_extra_count > 0:
        notes.append(
            f"Detected {dup_extra_count} duplicate timestamp record(s) across "
            f"{len(dup_ts_strings)} unique timestamp(s)."
        )

    # 3. Numeric sensor type validation and coercion
    non_numeric_coercions: dict[str, int] = {}
    for col in NUMERIC_SENSOR_COLUMNS:
        orig_non_null = assessed[col].notna()
        coerced = pd.to_numeric(assessed[col], errors="coerce")
        coerced_nans = int((orig_non_null & coerced.isna()).sum())
        non_numeric_coercions[col] = coerced_nans
        assessed[col] = coerced
        if coerced_nans > 0:
            notes.append(
                f"Column '{col}' contained {coerced_nans} non-numeric value(s) coerced to NaN."
            )

    # 4. Missing values
    missing_per_col: dict[str, int] = {
        col: int(assessed[col].isna().sum()) for col in assessed.columns if not col.startswith("dq_")
    }
    total_missing = int(sum(missing_per_col.values()))
    sensor_missing_mask = assessed[list(NUMERIC_SENSOR_COLUMNS)].isna().any(axis=1)
    assessed["dq_has_missing_sensor"] = sensor_missing_mask

    # 5. Physical sensor range checks
    out_of_range_counts: dict[str, int] = {}
    out_of_range_cols_per_row: list[list[str]] = [[] for _ in range(len(assessed))]

    for col, limits in sensor_limits.items():
        if col not in assessed.columns:
            continue
        series = assessed[col]
        invalid_range = series.notna() & ((series < limits.min_val) | (series > limits.max_val))
        count = int(invalid_range.sum())
        out_of_range_counts[col] = count
        if count > 0:
            notes.append(
                f"Column '{col}' has {count} reading(s) outside configured physical limits "
                f"[{limits.min_val}, {limits.max_val}] {limits.unit}."
            )
            for idx_pos in invalid_range[invalid_range].index:
                row_pos = assessed.index.get_loc(idx_pos)
                out_of_range_cols_per_row[row_pos].append(col)

    total_out_of_range = int(sum(out_of_range_counts.values()))
    assessed["dq_out_of_range"] = [len(cols) > 0 for cols in out_of_range_cols_per_row]
    assessed["dq_out_of_range_cols"] = [
        "; ".join(cols) if cols else "NONE" for cols in out_of_range_cols_per_row
    ]

    # Normalize status strings cleanly without dropping missing
    assessed["status"] = assessed["status"].astype(str).str.strip().str.upper()

    report = DataQualityReport(
        total_rows=len(assessed),
        valid_timestamp_rows=int(valid_ts_mask.sum()),
        invalid_timestamp_rows=invalid_ts_count,
        duplicate_timestamp_count=dup_extra_count,
        duplicate_timestamps=dup_ts_strings,
        missing_values_per_column=missing_per_col,
        total_missing_values=total_missing,
        non_numeric_coercions_per_column=non_numeric_coercions,
        out_of_range_counts_per_column=out_of_range_counts,
        total_out_of_range_readings=total_out_of_range,
        is_valid_schema=True,
        notes=notes,
    )
    logger.info(
        "Data quality assessment complete: %d rows, %d missing values, %d duplicate timestamps, %d out-of-range readings.",
        report.total_rows,
        report.total_missing_values,
        report.duplicate_timestamp_count,
        report.total_out_of_range_readings,
    )
    return assessed, report
