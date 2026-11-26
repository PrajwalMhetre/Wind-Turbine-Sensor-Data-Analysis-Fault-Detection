"""Tests for rule-based fault detection engine."""

from __future__ import annotations

import pandas as pd

from wind_faults.config import PipelineConfig
from wind_faults.data_quality import assess_data_quality
from wind_faults.fault_detection import detect_rule_based_faults


def _make_clean_df(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-15 00:00:00", periods=rows, freq="1min"),
            "wind_speed_mps": [9.5] * rows,
            "rotor_speed_rpm": [14.0] * rows,
            "generator_speed_rpm": [1120.0] * rows,
            "temperature_c": [55.0] * rows,
            "vibration_mm_s": [2.5] * rows,
            "power_kw": [950.0] * rows,
            "torque_nm": [6500.0] * rows,
            "pitch_angle_deg": [1.0] * rows,
            "status": ["OPERATING"] * rows,
        }
    )


def test_high_vibration_detection() -> None:
    cfg = PipelineConfig()
    df = _make_clean_df()
    df.loc[1, "vibration_mm_s"] = 13.5

    assessed, _ = assess_data_quality(df, cfg.sensor_limits)
    out, events = detect_rule_based_faults(assessed, cfg)

    assert bool(out.loc[1, "fault_high_vibration"]) is True
    assert out.loc[1, "fault_severity"] == "CRITICAL"
    assert any(e.fault_category == "HIGH_VIBRATION" for e in events)


def test_high_temperature_detection() -> None:
    cfg = PipelineConfig()
    df = _make_clean_df()
    df.loc[0, "temperature_c"] = 95.0

    assessed, _ = assess_data_quality(df, cfg.sensor_limits)
    out, events = detect_rule_based_faults(assessed, cfg)

    assert bool(out.loc[0, "fault_high_temperature"]) is True
    assert out.loc[0, "fault_severity"] == "WARNING"
    assert any(e.fault_category == "HIGH_TEMPERATURE" for e in events)


def test_low_power_detection_and_idle_exclusion() -> None:
    cfg = PipelineConfig()
    df = _make_clean_df(rows=3)
    # Row 0: normal operating -> no fault
    # Row 1: operating with high wind but low power -> LOW_POWER_OUTPUT
    df.loc[1, "wind_speed_mps"] = 10.0
    df.loc[1, "power_kw"] = 100.0
    # Row 2: IDLE with low wind & 0 power -> NO low power fault
    df.loc[2, "status"] = "IDLE"
    df.loc[2, "wind_speed_mps"] = 2.0
    df.loc[2, "rotor_speed_rpm"] = 0.5
    df.loc[2, "power_kw"] = 0.0

    assessed, _ = assess_data_quality(df, cfg.sensor_limits)
    out, _ = detect_rule_based_faults(assessed, cfg)

    assert bool(out.loc[0, "fault_low_power"]) is False
    assert bool(out.loc[1, "fault_low_power"]) is True
    assert bool(out.loc[2, "fault_low_power"]) is False


def test_sensor_range_violation_detection() -> None:
    cfg = PipelineConfig()
    df = _make_clean_df()
    df.loc[2, "wind_speed_mps"] = 85.0

    assessed, _ = assess_data_quality(df, cfg.sensor_limits)
    out, events = detect_rule_based_faults(assessed, cfg)

    assert bool(out.loc[2, "fault_sensor_range"]) is True
    assert any(e.fault_category == "SENSOR_RANGE_VIOLATION" for e in events)


def test_multiple_simultaneous_faults() -> None:
    cfg = PipelineConfig()
    df = _make_clean_df()
    df.loc[1, "vibration_mm_s"] = 14.0
    df.loc[1, "temperature_c"] = 102.0
    df.loc[1, "rotor_speed_rpm"] = 25.0

    assessed, _ = assess_data_quality(df, cfg.sensor_limits)
    out, events = detect_rule_based_faults(assessed, cfg)

    assert bool(out.loc[1, "fault_high_vibration"]) is True
    assert bool(out.loc[1, "fault_high_temperature"]) is True
    assert bool(out.loc[1, "fault_abnormal_rotor_speed"]) is True
    assert "HIGH_VIBRATION" in out.loc[1, "fault_categories"]
    assert "HIGH_TEMPERATURE" in out.loc[1, "fault_categories"]
    assert "ABNORMAL_ROTOR_SPEED" in out.loc[1, "fault_categories"]
    assert len(events) == 3
