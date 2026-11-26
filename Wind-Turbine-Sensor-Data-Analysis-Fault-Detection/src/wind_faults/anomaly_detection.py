"""Causal rolling-median and MAD statistical anomaly detection."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from wind_faults.config import AnomalyConfig, SensorRange

logger = logging.getLogger(__name__)

# Consistency constant for normal distribution: MAD * 1.4826 ≈ standard deviation
MAD_SCALE_FACTOR = 1.4826


@dataclass
class AnomalySummary:
    """Summary statistics from statistical anomaly detection."""

    rolling_window: int
    min_periods: int
    z_threshold: float
    monitored_columns: list[str]
    total_anomalous_rows: int
    anomalous_rows_per_sensor: dict[str, int]
    max_observed_score: float
    note: str = (
        "Statistical anomalies indicate unusual behavior relative to recent causal "
        "rolling baselines and are NOT confirmed mechanical failures."
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


def _rolling_mad_zscore(
    series: pd.Series,
    window: int,
    min_periods: int,
    epsilon: float,
) -> pd.Series:
    """Compute causal (past-only) rolling modified Z-score using median and MAD.

    No future observations are used at any index t (center=False), preventing
    look-ahead leakage. Handles NaNs via np.nanmedian and avoids division-by-zero.
    """
    rolling_med = series.rolling(window=window, min_periods=min_periods, center=False).median()

    def _mad_fn(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < min_periods:
            return float("nan")
        med = float(np.median(valid))
        return float(np.median(np.abs(valid - med)))

    rolling_mad = series.rolling(window=window, min_periods=min_periods, center=False).apply(
        _mad_fn, raw=True
    )

    rolling_std = series.rolling(window=window, min_periods=min_periods, center=False).std()
    diff = (series - rolling_med).abs()
    scaled_mad = MAD_SCALE_FACTOR * rolling_mad
    # Fall back to rolling_std (then epsilon) when MAD is zero to prevent division-by-zero spikes
    effective_scale = np.where(
        scaled_mad > 0.0,
        scaled_mad,
        np.maximum(rolling_std.fillna(0.0), epsilon),
    )

    scores = pd.Series(np.nan, index=series.index, dtype=float)
    valid_mask = series.notna() & rolling_med.notna() & rolling_mad.notna()

    diff_valid = diff.loc[valid_mask]
    denom_valid = pd.Series(effective_scale, index=series.index).loc[valid_mask].clip(lower=epsilon)
    scores.loc[valid_mask] = np.where(diff_valid == 0.0, 0.0, diff_valid / denom_valid)
    return scores


def detect_statistical_anomalies(
    df: pd.DataFrame,
    config: AnomalyConfig,
    sensor_limits: dict[str, SensorRange] | None = None,
) -> tuple[pd.DataFrame, AnomalySummary]:
    """Compute causal rolling-MAD anomaly scores for monitored sensor columns.

    Args:
        df: Chronologically sorted DataFrame.
        config: Validated AnomalyConfig.
        sensor_limits: Optional physical limits; if `exclude_out_of_range_from_baseline`
            is True, physically impossible values are masked before computing the
            rolling baseline so single-point glitches do not corrupt subsequent windows.
    """
    if config.rolling_window < 2:
        raise ValueError(f"rolling_window must be >= 2, got {config.rolling_window}.")
    if config.min_periods < 1 or config.min_periods > config.rolling_window:
        raise ValueError(
            f"min_periods ({config.min_periods}) must be in [1, {config.rolling_window}]."
        )

    result = df.copy()
    score_cols: list[str] = []
    anomalous_per_sensor: dict[str, int] = {}

    for col in config.monitored_columns:
        if col not in result.columns:
            raise ValueError(f"Monitored column '{col}' not found in DataFrame.")

        work_series = result[col].astype(float).copy()
        if config.exclude_out_of_range_from_baseline and sensor_limits and col in sensor_limits:
            sr = sensor_limits[col]
            out_mask = work_series.notna() & (
                (work_series < sr.min_val) | (work_series > sr.max_val)
            )
            work_series = work_series.mask(out_mask)

        score_col = f"anomaly_score_{col}"
        scores = _rolling_mad_zscore(
            work_series,
            window=config.rolling_window,
            min_periods=config.min_periods,
            epsilon=config.epsilon,
        )
        result[score_col] = scores.round(4)
        score_cols.append(score_col)
        anomalous_per_sensor[col] = int((scores >= config.z_threshold).sum())

    score_matrix = result[score_cols]
    max_scores = score_matrix.max(axis=1, skipna=True)
    result["max_anomaly_score"] = max_scores.round(4)
    result["is_statistical_anomaly"] = max_scores.notna() & (max_scores >= config.z_threshold)

    # Human-readable anomaly explanation per row
    explanations: list[str] = []
    for idx in result.index:
        if not bool(result.loc[idx, "is_statistical_anomaly"]):
            explanations.append("Within normal statistical baseline.")
            continue
        triggered: list[str] = []
        for col in config.monitored_columns:
            sc = result.loc[idx, f"anomaly_score_{col}"]
            if pd.notna(sc) and float(sc) >= config.z_threshold:
                triggered.append(f"{col} (score={float(sc):.2f})")
        explanations.append(
            "Suspected statistical anomaly (not a confirmed mechanical failure) in: "
            + ", ".join(triggered)
        )
    result["anomaly_explanation"] = explanations

    max_obs = float(max_scores.max()) if max_scores.notna().any() else 0.0
    summary = AnomalySummary(
        rolling_window=config.rolling_window,
        min_periods=config.min_periods,
        z_threshold=config.z_threshold,
        monitored_columns=list(config.monitored_columns),
        total_anomalous_rows=int(result["is_statistical_anomaly"].sum()),
        anomalous_rows_per_sensor=anomalous_per_sensor,
        max_observed_score=round(max_obs, 4),
    )
    logger.info(
        "Statistical anomaly detection complete: %d suspected anomalous timestamp(s) (threshold=%.2f).",
        summary.total_anomalous_rows,
        config.z_threshold,
    )
    return result, summary
