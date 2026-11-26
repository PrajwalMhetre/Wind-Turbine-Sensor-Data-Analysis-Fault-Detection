"""Configuration dataclasses and JSON loader for wind turbine fault detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when configuration values are missing or invalid."""


@dataclass(frozen=True)
class SensorRange:
    """Physical operating or measurement limits for a single sensor."""

    min_val: float
    max_val: float
    unit: str = ""

    def __post_init__(self) -> None:
        if self.min_val >= self.max_val:
            raise ConfigurationError(
                f"SensorRange min_val ({self.min_val}) must be less than max_val ({self.max_val})."
            )


@dataclass(frozen=True)
class RuleThresholds:
    """Configurable engineering thresholds for rule-based fault detection.

    Note: Default values are illustrative engineering assumptions for
    demonstration and should be calibrated to specific turbine OEM specs
    before industrial use.
    """

    high_vibration_mm_s: float = 10.0
    high_temperature_c: float = 90.0
    low_power_wind_speed_min_mps: float = 8.0
    low_power_output_max_kw: float = 200.0
    rotor_speed_operating_min_rpm: float = 4.0
    rotor_speed_operating_max_rpm: float = 22.0
    operating_statuses: tuple[str, ...] = ("OPERATING",)

    def __post_init__(self) -> None:
        if self.high_vibration_mm_s <= 0:
            raise ConfigurationError("high_vibration_mm_s must be positive.")
        if self.low_power_wind_speed_min_mps < 0:
            raise ConfigurationError("low_power_wind_speed_min_mps must be non-negative.")
        if self.rotor_speed_operating_min_rpm < 0:
            raise ConfigurationError("rotor_speed_operating_min_rpm must be non-negative.")
        if self.rotor_speed_operating_min_rpm >= self.rotor_speed_operating_max_rpm:
            raise ConfigurationError(
                "rotor_speed_operating_min_rpm must be less than rotor_speed_operating_max_rpm."
            )


@dataclass(frozen=True)
class AnomalyConfig:
    """Configuration for rolling-median / MAD statistical anomaly detection."""

    rolling_window: int = 30
    min_periods: int = 10
    z_threshold: float = 3.5
    monitored_columns: tuple[str, ...] = (
        "temperature_c",
        "vibration_mm_s",
        "rotor_speed_rpm",
        "power_kw",
    )
    epsilon: float = 0.5
    exclude_out_of_range_from_baseline: bool = True

    def __post_init__(self) -> None:
        if self.rolling_window < 2:
            raise ConfigurationError(
                f"rolling_window must be >= 2, got {self.rolling_window}."
            )
        if self.min_periods < 1 or self.min_periods > self.rolling_window:
            raise ConfigurationError(
                f"min_periods ({self.min_periods}) must be between 1 and rolling_window ({self.rolling_window})."
            )
        if self.z_threshold <= 0:
            raise ConfigurationError(f"z_threshold must be > 0, got {self.z_threshold}.")
        if self.epsilon <= 0:
            raise ConfigurationError(f"epsilon must be > 0, got {self.epsilon}.")
        if not self.monitored_columns:
            raise ConfigurationError("monitored_columns must not be empty.")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for data cleaning and preprocessing behavior."""

    duplicate_timestamp_strategy: str = "keep_first"
    missing_value_strategy: str = "retain"
    ffill_limit: int = 0

    def __post_init__(self) -> None:
        valid_dup = {"keep_first", "keep_last", "flag_only"}
        if self.duplicate_timestamp_strategy not in valid_dup:
            raise ConfigurationError(
                f"duplicate_timestamp_strategy must be one of {sorted(valid_dup)}, "
                f"got '{self.duplicate_timestamp_strategy}'."
            )
        valid_missing = {"retain", "ffill_limit"}
        if self.missing_value_strategy not in valid_missing:
            raise ConfigurationError(
                f"missing_value_strategy must be one of {sorted(valid_missing)}, "
                f"got '{self.missing_value_strategy}'."
            )
        if self.ffill_limit < 0:
            raise ConfigurationError("ffill_limit must be >= 0.")


def default_sensor_limits() -> dict[str, SensorRange]:
    """Return default illustrative physical sensor limits."""
    return {
        "wind_speed_mps": SensorRange(0.0, 50.0, "m/s"),
        "rotor_speed_rpm": SensorRange(0.0, 35.0, "RPM"),
        "generator_speed_rpm": SensorRange(0.0, 2500.0, "RPM"),
        "temperature_c": SensorRange(-30.0, 150.0, "C"),
        "vibration_mm_s": SensorRange(0.0, 50.0, "mm/s"),
        "power_kw": SensorRange(-50.0, 3500.0, "kW"),
        "torque_nm": SensorRange(-500.0, 25000.0, "Nm"),
        "pitch_angle_deg": SensorRange(-5.0, 95.0, "deg"),
    }


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration container for the fault detection pipeline."""

    rules: RuleThresholds = field(default_factory=RuleThresholds)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sensor_limits: dict[str, SensorRange] = field(default_factory=default_sensor_limits)


def _parse_sensor_limits(raw_limits: dict[str, Any]) -> dict[str, SensorRange]:
    limits: dict[str, SensorRange] = {}
    for col, spec in raw_limits.items():
        if not isinstance(spec, dict) or "min" not in spec or "max" not in spec:
            raise ConfigurationError(f"Invalid sensor_limits entry for '{col}'.")
        limits[col] = SensorRange(
            min_val=float(spec["min"]),
            max_val=float(spec["max"]),
            unit=str(spec.get("unit", "")),
        )
    return limits


def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    """Load pipeline configuration from a JSON file, or return defaults if None."""
    if config_path is None:
        return PipelineConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Malformed JSON in configuration file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a JSON object.")

    rules_raw = raw.get("rules", {})
    rules = RuleThresholds(
        high_vibration_mm_s=float(rules_raw.get("high_vibration_mm_s", 10.0)),
        high_temperature_c=float(rules_raw.get("high_temperature_c", 90.0)),
        low_power_wind_speed_min_mps=float(rules_raw.get("low_power_wind_speed_min_mps", 8.0)),
        low_power_output_max_kw=float(rules_raw.get("low_power_output_max_kw", 200.0)),
        rotor_speed_operating_min_rpm=float(rules_raw.get("rotor_speed_operating_min_rpm", 4.0)),
        rotor_speed_operating_max_rpm=float(rules_raw.get("rotor_speed_operating_max_rpm", 22.0)),
        operating_statuses=tuple(rules_raw.get("operating_statuses", ["OPERATING"])),
    )

    anom_raw = raw.get("anomaly_detection", {})
    anomaly = AnomalyConfig(
        rolling_window=int(anom_raw.get("rolling_window", 30)),
        min_periods=int(anom_raw.get("min_periods", 10)),
        z_threshold=float(anom_raw.get("z_threshold", 3.5)),
        monitored_columns=tuple(
            anom_raw.get(
                "monitored_columns",
                ["temperature_c", "vibration_mm_s", "rotor_speed_rpm", "power_kw"],
            )
        ),
        epsilon=float(anom_raw.get("epsilon", 0.5)),
        exclude_out_of_range_from_baseline=bool(
            anom_raw.get("exclude_out_of_range_from_baseline", True)
        ),
    )

    prep_raw = raw.get("preprocessing", {})
    preprocessing = PreprocessingConfig(
        duplicate_timestamp_strategy=str(
            prep_raw.get("duplicate_timestamp_strategy", "keep_first")
        ),
        missing_value_strategy=str(prep_raw.get("missing_value_strategy", "retain")),
        ffill_limit=int(prep_raw.get("ffill_limit", 0)),
    )

    limits_raw = raw.get("sensor_limits")
    sensor_limits = (
        _parse_sensor_limits(limits_raw) if limits_raw is not None else default_sensor_limits()
    )

    return PipelineConfig(
        rules=rules,
        anomaly=anomaly,
        preprocessing=preprocessing,
        sensor_limits=sensor_limits,
    )
