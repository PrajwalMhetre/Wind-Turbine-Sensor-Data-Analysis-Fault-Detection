"""Rule-based fault detection engine with configurable engineering thresholds."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from wind_faults.config import PipelineConfig

logger = logging.getLogger(__name__)

SEVERITY_RANK: dict[str, int] = {
    "NONE": 0,
    "WARNING": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}


@dataclass(frozen=True)
class FaultEvent:
    """Structured record of a single rule-based fault detection event."""

    timestamp: str
    fault_category: str
    sensor_name: str
    sensor_value: float
    threshold_condition: str
    severity: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert fault event to dictionary."""
        return asdict(self)


def _max_severity(severities: list[str]) -> str:
    if not severities:
        return "NONE"
    return max(severities, key=lambda s: SEVERITY_RANK.get(s, 0))


def detect_rule_based_faults(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, list[FaultEvent]]:
    """Apply configurable rule-based fault checks to preprocessed sensor data.

    Supports multiple simultaneous fault conditions per timestamp and avoids
    flagging low power during normal low-wind or non-operating states.
    """
    result = df.copy()
    rules = config.rules
    limits = config.sensor_limits
    operating_set = {s.upper() for s in rules.operating_statuses}

    # 1. High vibration
    vib_mask = result["vibration_mm_s"].notna() & (
        result["vibration_mm_s"] > rules.high_vibration_mm_s
    )

    # 2. High temperature
    temp_mask = result["temperature_c"].notna() & (
        result["temperature_c"] > rules.high_temperature_c
    )

    # 3. Low power output under suitable wind & active operating state
    is_operating = result["status"].astype(str).str.upper().isin(operating_set)
    low_power_mask = (
        is_operating
        & result["wind_speed_mps"].notna()
        & result["power_kw"].notna()
        & (result["wind_speed_mps"] >= rules.low_power_wind_speed_min_mps)
        & (result["power_kw"] < rules.low_power_output_max_kw)
    )

    # 4. Abnormal rotor speed (underspeed while operating, or overspeed anytime)
    rotor_valid = result["rotor_speed_rpm"].notna()
    rotor_overspeed = rotor_valid & (
        result["rotor_speed_rpm"] > rules.rotor_speed_operating_max_rpm
    )
    rotor_underspeed = (
        is_operating
        & rotor_valid
        & (result["rotor_speed_rpm"] < rules.rotor_speed_operating_min_rpm)
    )
    rotor_mask = rotor_overspeed | rotor_underspeed

    # 5. Sensor range violations (computed directly against config.sensor_limits)
    range_violation_cols: list[list[str]] = [[] for _ in range(len(result))]
    for col, sr in limits.items():
        if col not in result.columns:
            continue
        bad = result[col].notna() & ((result[col] < sr.min_val) | (result[col] > sr.max_val))
        for idx_pos in bad[bad].index:
            pos = result.index.get_loc(idx_pos)
            range_violation_cols[pos].append(col)
    range_mask = pd.Series([len(c) > 0 for c in range_violation_cols], index=result.index)

    result["fault_high_vibration"] = vib_mask
    result["fault_high_temperature"] = temp_mask
    result["fault_low_power"] = low_power_mask
    result["fault_abnormal_rotor_speed"] = rotor_mask
    result["fault_sensor_range"] = range_mask
    result["rule_fault_detected"] = (
        vib_mask | temp_mask | low_power_mask | rotor_mask | range_mask
    )

    events: list[FaultEvent] = []
    categories_col: list[str] = []
    severity_col: list[str] = []
    explanation_col: list[str] = []

    for pos, idx in enumerate(result.index):
        row = result.loc[idx]
        ts_val = row["timestamp"]
        ts_str = ts_val.isoformat() if pd.notna(ts_val) else "UNKNOWN"
        row_cats: list[str] = []
        row_sevs: list[str] = []
        row_exps: list[str] = []

        if bool(vib_mask.loc[idx]):
            val = float(row["vibration_mm_s"])
            sev = "CRITICAL" if val >= rules.high_vibration_mm_s * 1.25 else "WARNING"
            cond = f"vibration_mm_s > {rules.high_vibration_mm_s} mm/s"
            exp = (
                f"Vibration reached {val:.2f} mm/s, exceeding threshold "
                f"{rules.high_vibration_mm_s:.2f} mm/s."
            )
            row_cats.append("HIGH_VIBRATION")
            row_sevs.append(sev)
            row_exps.append(exp)
            events.append(
                FaultEvent(ts_str, "HIGH_VIBRATION", "vibration_mm_s", val, cond, sev, exp)
            )

        if bool(temp_mask.loc[idx]):
            val = float(row["temperature_c"])
            sev = "CRITICAL" if val >= rules.high_temperature_c + 10.0 else "WARNING"
            cond = f"temperature_c > {rules.high_temperature_c} C"
            exp = (
                f"Operating temperature reached {val:.2f} C, exceeding threshold "
                f"{rules.high_temperature_c:.2f} C."
            )
            row_cats.append("HIGH_TEMPERATURE")
            row_sevs.append(sev)
            row_exps.append(exp)
            events.append(
                FaultEvent(ts_str, "HIGH_TEMPERATURE", "temperature_c", val, cond, sev, exp)
            )

        if bool(low_power_mask.loc[idx]):
            pwr = float(row["power_kw"])
            wnd = float(row["wind_speed_mps"])
            sev = "WARNING"
            cond = (
                f"status in {list(operating_set)} AND wind_speed_mps >= "
                f"{rules.low_power_wind_speed_min_mps} m/s AND power_kw < {rules.low_power_output_max_kw} kW"
            )
            exp = (
                f"Power output was {pwr:.2f} kW at wind speed {wnd:.2f} m/s "
                f"(expected >= {rules.low_power_output_max_kw:.2f} kW during operation)."
            )
            row_cats.append("LOW_POWER_OUTPUT")
            row_sevs.append(sev)
            row_exps.append(exp)
            events.append(
                FaultEvent(ts_str, "LOW_POWER_OUTPUT", "power_kw", pwr, cond, sev, exp)
            )

        if bool(rotor_mask.loc[idx]):
            rpm = float(row["rotor_speed_rpm"])
            is_over = rpm > rules.rotor_speed_operating_max_rpm
            sev = "CRITICAL" if is_over else "WARNING"
            cond = (
                f"rotor_speed_rpm outside [{rules.rotor_speed_operating_min_rpm}, "
                f"{rules.rotor_speed_operating_max_rpm}] RPM"
            )
            exp = (
                f"Rotor speed was {rpm:.2f} RPM "
                f"({'above max' if is_over else 'below min'} operating limit)."
            )
            row_cats.append("ABNORMAL_ROTOR_SPEED")
            row_sevs.append(sev)
            row_exps.append(exp)
            events.append(
                FaultEvent(
                    ts_str, "ABNORMAL_ROTOR_SPEED", "rotor_speed_rpm", rpm, cond, sev, exp
                )
            )

        if range_violation_cols[pos]:
            row_cats.append("SENSOR_RANGE_VIOLATION")
            row_sevs.append("HIGH")
            for col in range_violation_cols[pos]:
                val = float(row[col])
                sr = limits[col]
                cond = f"{col} outside [{sr.min_val}, {sr.max_val}] {sr.unit}"
                exp = (
                    f"Sensor '{col}' reading {val:.2f} {sr.unit} is outside physical limits "
                    f"[{sr.min_val}, {sr.max_val}] {sr.unit}."
                )
                row_exps.append(exp)
                events.append(
                    FaultEvent(ts_str, "SENSOR_RANGE_VIOLATION", col, val, cond, "HIGH", exp)
                )

        categories_col.append("; ".join(row_cats) if row_cats else "NONE")
        severity_col.append(_max_severity(row_sevs))
        explanation_col.append(" | ".join(row_exps) if row_exps else "Normal operating condition.")

    result["fault_categories"] = categories_col
    result["fault_severity"] = severity_col
    result["fault_explanation"] = explanation_col

    logger.info(
        "Rule-based detection complete: %d faulty timestamp(s), %d total fault event(s).",
        int(result["rule_fault_detected"].sum()),
        len(events),
    )
    return result, events
