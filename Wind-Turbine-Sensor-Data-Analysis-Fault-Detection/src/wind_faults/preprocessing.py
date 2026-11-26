"""Configurable preprocessing for wind turbine sensor data."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from wind_faults.config import PreprocessingConfig
from wind_faults.data_loader import NUMERIC_SENSOR_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Summary of preprocessing actions applied to the sensor DataFrame."""

    df: pd.DataFrame
    rows_before: int
    rows_after: int
    invalid_timestamps_dropped: int
    duplicates_removed: int
    values_imputed: int
    actions_taken: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return JSON-serializable summary without the DataFrame."""
        data = asdict(self)
        data.pop("df", None)
        return data


def preprocess_sensor_data(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> PreprocessingResult:
    """Sort chronologically and apply configurable duplicate and missing-value policies.

    Preserves original sensor values by default (`missing_value_strategy='retain'`)
    so missingness remains transparent for engineering review.
    """
    processed = df.copy()
    rows_before = len(processed)
    actions: list[str] = []

    # Drop rows only if timestamp itself is unparseable (cannot place on time axis)
    invalid_ts_count = int(processed["timestamp"].isna().sum())
    if invalid_ts_count > 0:
        processed = processed.loc[processed["timestamp"].notna()].copy()
        actions.append(
            f"Dropped {invalid_ts_count} row(s) with invalid/unparseable timestamps."
        )

    # Sort chronologically while preserving original relative order for ties
    processed = processed.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    actions.append("Sorted records chronologically by timestamp.")

    # Duplicate timestamp handling
    duplicates_removed = 0
    if config.duplicate_timestamp_strategy in {"keep_first", "keep_last"}:
        keep_arg = "first" if config.duplicate_timestamp_strategy == "keep_first" else "last"
        dup_mask = processed["timestamp"].duplicated(keep=keep_arg)
        duplicates_removed = int(dup_mask.sum())
        if duplicates_removed > 0:
            processed = processed.loc[~dup_mask].reset_index(drop=True)
            actions.append(
                f"Removed {duplicates_removed} duplicate timestamp row(s) using "
                f"strategy '{config.duplicate_timestamp_strategy}'."
            )
    else:
        actions.append(
            "Retained duplicate timestamp rows with 'dq_is_duplicate_ts' flag ('flag_only')."
        )

    # Missing value handling (never blindly interpolate)
    values_imputed = 0
    if config.missing_value_strategy == "retain":
        actions.append(
            "Retained missing sensor values as NaN ('retain' policy) to avoid masking sensor dropouts."
        )
    elif config.missing_value_strategy == "ffill_limit" and config.ffill_limit > 0:
        for col in NUMERIC_SENSOR_COLUMNS:
            before_na = int(processed[col].isna().sum())
            processed[col] = processed[col].ffill(limit=config.ffill_limit)
            after_na = int(processed[col].isna().sum())
            values_imputed += before_na - after_na
        actions.append(
            f"Forward-filled {values_imputed} missing sensor value(s) with limit={config.ffill_limit}."
        )
    else:
        actions.append("Retained missing sensor values as NaN (ffill_limit=0).")

    rows_after = len(processed)
    logger.info(
        "Preprocessing complete: %d -> %d rows (duplicates removed: %d, imputed: %d).",
        rows_before,
        rows_after,
        duplicates_removed,
        values_imputed,
    )
    return PreprocessingResult(
        df=processed,
        rows_before=rows_before,
        rows_after=rows_after,
        invalid_timestamps_dropped=invalid_ts_count,
        duplicates_removed=duplicates_removed,
        values_imputed=values_imputed,
        actions_taken=actions,
    )
