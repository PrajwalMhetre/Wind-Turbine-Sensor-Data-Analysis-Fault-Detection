"""Deterministic synthetic wind turbine sensor dataset generator.

Generates 24 hours (1,440 1-minute records + 2 documented duplicate timestamps = 1,442 rows)
with realistic operating physics and explicitly documented synthetic fault and
data-quality scenarios. This data is strictly for proof-of-concept demonstration
and does NOT represent real industrial SCADA measurements.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticScenario:
    """Documents an injected synthetic fault or data-quality scenario."""

    name: str
    category: str
    start_index: int
    end_index: int
    description: str


SYNTHETIC_SCENARIOS: tuple[SyntheticScenario, ...] = (
    SyntheticScenario(
        name="Normal Low-Wind Calm Period",
        category="NORMAL_IDLE",
        start_index=120,
        end_index=164,
        description="02:00-02:44: Wind drops below 2.8 m/s; turbine enters IDLE with ~0 kW and ~0 RPM (tests no false low-power alerts).",
    ),
    SyntheticScenario(
        name="Excessive Vibration Episode",
        category="HIGH_VIBRATION",
        start_index=300,
        end_index=344,
        description="05:00-05:44: Vibration ramps to 11.5-15.2 mm/s (exceeding 10.0 mm/s threshold).",
    ),
    SyntheticScenario(
        name="High Operating Temperature Episode",
        category="HIGH_TEMPERATURE",
        start_index=540,
        end_index=584,
        description="09:00-09:44: Operating temperature rises to 92.5-104.0 C (exceeding 90.0 C threshold).",
    ),
    SyntheticScenario(
        name="Unexpectedly Low Power Output",
        category="LOW_POWER_OUTPUT",
        start_index=720,
        end_index=759,
        description="12:00-12:39: Wind speed is 9.0-11.2 m/s while OPERATING, but power drops to 85-145 kW.",
    ),
    SyntheticScenario(
        name="Abnormal Rotor Overspeed",
        category="ABNORMAL_ROTOR_SPEED",
        start_index=900,
        end_index=924,
        description="15:00-15:24: Rotor speed spikes to 24.0-27.5 RPM (exceeding 22.0 RPM operating limit).",
    ),
    SyntheticScenario(
        name="Simultaneous High Vibration & High Temperature",
        category="SIMULTANEOUS_FAULTS",
        start_index=1080,
        end_index=1104,
        description="18:00-18:24: Vibration (12.0-14.5 mm/s) and temperature (93.0-98.5 C) exceed thresholds simultaneously.",
    ),
    SyntheticScenario(
        name="Missing Sensor Values",
        category="DATA_QUALITY_MISSING",
        start_index=400,
        end_index=1320,
        description="12 scattered NaN readings injected across vibration_mm_s, temperature_c, power_kw, and wind_speed_mps.",
    ),
    SyntheticScenario(
        name="Physical Sensor Limit Violations",
        category="SENSOR_RANGE_VIOLATION",
        start_index=1250,
        end_index=1253,
        description="4 records with physically impossible readings (temperature=220 C, vibration=-5 mm/s, wind=85 m/s, power=5200 kW).",
    ),
    SyntheticScenario(
        name="Duplicate Timestamps",
        category="DATA_QUALITY_DUPLICATE",
        start_index=200,
        end_index=600,
        description="2 exact duplicate timestamp rows appended to test duplicate detection and deduplication.",
    ),
)


def get_scenario_manifest() -> list[dict[str, Any]]:
    """Return documented synthetic scenarios as JSON-serializable dicts."""
    return [asdict(s) for s in SYNTHETIC_SCENARIOS]


def generate_synthetic_turbine_data(
    num_minutes: int = 1440,
    seed: int = 42,
    start_time: str = "2026-01-15 00:00:00",
) -> pd.DataFrame:
    """Generate a deterministic synthetic wind turbine SCADA-like dataset."""
    if num_minutes < 1440:
        raise ValueError("num_minutes must be at least 1440 to represent one full day.")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_time, periods=num_minutes, freq="1min")
    t = np.arange(num_minutes, dtype=float)

    # Smooth diurnal wind profile + correlated turbulence
    base_wind = 8.5 + 2.8 * np.sin(2 * np.pi * (t - 180) / num_minutes) + 1.2 * np.cos(
        2 * np.pi * t / 360.0
    )
    noise = rng.normal(0.0, 0.35, size=num_minutes)
    wind_speed = np.clip(base_wind + noise, 3.8, 16.5)

    # Normal low-wind calm period (indices 120..164)
    wind_speed[120:165] = np.clip(2.1 + rng.normal(0.0, 0.25, size=45), 1.2, 2.8)

    status = np.where(wind_speed >= 3.2, "OPERATING", "IDLE")

    # Rotor speed (RPM): 0.5 RPM when idle, 8..18 RPM when operating
    rotor_speed = np.where(
        status == "OPERATING",
        np.clip(4.5 + 1.15 * wind_speed + rng.normal(0.0, 0.2, size=num_minutes), 6.0, 19.5),
        np.clip(0.4 + rng.normal(0.0, 0.1, size=num_minutes), 0.0, 1.0),
    )

    # Gearbox ratio ~80:1
    generator_speed = np.where(
        status == "OPERATING",
        rotor_speed * 80.0 + rng.normal(0.0, 6.0, size=num_minutes),
        rotor_speed * 80.0,
    )

    # Pitch angle (deg): ~1 deg below rated wind (12 m/s), pitches up above 12 m/s, 85 deg when idle
    pitch_angle = np.where(
        status == "OPERATING",
        np.where(
            wind_speed <= 12.0,
            np.clip(1.0 + rng.normal(0.0, 0.25, size=num_minutes), 0.0, 3.0),
            np.clip(1.0 + (wind_speed - 12.0) * 3.5 + rng.normal(0.0, 0.3, size=num_minutes), 1.0, 25.0),
        ),
        85.0 + rng.normal(0.0, 0.3, size=num_minutes),
    )

    # Power output (kW): cubic curve up to 2000 kW rated
    raw_power = 2000.0 * np.clip((wind_speed - 3.0) / (12.0 - 3.0), 0.0, 1.0) ** 3
    power_kw = np.where(
        status == "OPERATING",
        np.clip(raw_power + rng.normal(0.0, 22.0, size=num_minutes), 20.0, 2050.0),
        0.0,
    )

    # Generator/drivetrain torque (Nm): P = tau * omega => tau = (P * 1000) / (2*pi*RPM/60)
    omega_gen = np.maximum(generator_speed * 2.0 * np.pi / 60.0, 1.0)
    torque_nm = np.where(
        status == "OPERATING",
        np.clip((power_kw * 1000.0) / omega_gen + rng.normal(0.0, 30.0, size=num_minutes), 100.0, 18000.0),
        0.0,
    )

    # Baseline temperature (C) and vibration (mm/s)
    temperature_c = np.clip(
        48.0 + 0.008 * power_kw + 3.0 * np.sin(2 * np.pi * t / num_minutes) + rng.normal(0.0, 0.7, size=num_minutes),
        35.0,
        75.0,
    )
    vibration_mm_s = np.clip(
        2.1 + 0.09 * wind_speed + rng.normal(0.0, 0.22, size=num_minutes),
        0.8,
        5.5,
    )

    # --- Inject controlled fault scenarios ---
    # 1. Excessive vibration (300..344)
    vibration_mm_s[300:345] = np.linspace(11.5, 15.2, 45) + rng.normal(0.0, 0.25, size=45)

    # 2. High operating temperature (540..584)
    temperature_c[540:585] = np.linspace(92.5, 104.0, 45) + rng.normal(0.0, 0.4, size=45)

    # 3. Unexpectedly low power under suitable wind (720..759)
    wind_speed[720:760] = np.clip(9.8 + rng.normal(0.0, 0.35, size=40), 9.0, 11.2)
    status[720:760] = "OPERATING"
    power_kw[720:760] = np.clip(115.0 + rng.normal(0.0, 15.0, size=40), 80.0, 150.0)

    # 4. Abnormal rotor overspeed (900..924)
    status[900:925] = "OPERATING"
    rotor_speed[900:925] = np.linspace(24.0, 27.5, 25) + rng.normal(0.0, 0.2, size=25)
    generator_speed[900:925] = rotor_speed[900:925] * 80.0

    # 5. Simultaneous faults: high vibration + high temperature (1080..1104)
    vibration_mm_s[1080:1105] = np.linspace(12.0, 14.5, 25) + rng.normal(0.0, 0.2, size=25)
    temperature_c[1080:1105] = np.linspace(93.0, 98.5, 25) + rng.normal(0.0, 0.3, size=25)

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "wind_speed_mps": np.round(wind_speed, 3),
            "rotor_speed_rpm": np.round(rotor_speed, 3),
            "generator_speed_rpm": np.round(generator_speed, 2),
            "temperature_c": np.round(temperature_c, 2),
            "vibration_mm_s": np.round(vibration_mm_s, 3),
            "power_kw": np.round(power_kw, 2),
            "torque_nm": np.round(torque_nm, 2),
            "pitch_angle_deg": np.round(pitch_angle, 2),
            "status": status,
        }
    )

    # 6. Inject missing values (12 scattered NaNs)
    df.loc[[400, 401, 450], "vibration_mm_s"] = np.nan
    df.loc[[620, 621, 680], "temperature_c"] = np.nan
    df.loc[[810, 815, 850], "power_kw"] = np.nan
    df.loc[[1300, 1310, 1320], "wind_speed_mps"] = np.nan

    # 7. Inject out-of-range physical sensor violations (1250..1253)
    df.loc[1250, "temperature_c"] = 220.0
    df.loc[1251, "vibration_mm_s"] = -5.0
    df.loc[1252, "wind_speed_mps"] = 85.0
    df.loc[1253, "power_kw"] = 5200.0

    # 8. Inject 2 duplicate timestamp rows
    dup_rows = df.iloc[[200, 600]].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    logger.info(
        "Generated synthetic wind turbine dataset: %d rows (seed=%d).", len(df), seed
    )
    return df


def save_synthetic_dataset(
    output_path: Path | str,
    num_minutes: int = 1440,
    seed: int = 42,
) -> Path:
    """Generate and write the synthetic dataset to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_turbine_data(num_minutes=num_minutes, seed=seed)
    df.to_csv(path, index=False)
    logger.info("Saved synthetic dataset to %s", path)
    return path
