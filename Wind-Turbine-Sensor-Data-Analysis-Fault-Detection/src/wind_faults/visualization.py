"""Headless Matplotlib engineering visualizations for wind turbine condition monitoring."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from wind_faults.config import PipelineConfig

logger = logging.getLogger(__name__)


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, linestyle="--", alpha=0.35)


def _shade_fault_mask(
    ax: plt.Axes,
    timestamps: pd.Series,
    mask: pd.Series,
    color: str,
    label: str,
) -> None:
    """Highlight contiguous True intervals on a time-series plot."""
    if not mask.any():
        return
    labeled = False
    in_span = False
    start_ts = None
    prev_ts = None
    for ts, active in zip(timestamps, mask):
        if active and not in_span:
            in_span = True
            start_ts = ts
        elif not active and in_span:
            in_span = False
            ax.axvspan(
                start_ts,
                prev_ts,
                color=color,
                alpha=0.18,
                label=label if not labeled else None,
            )
            labeled = True
        prev_ts = ts
    if in_span and start_ts is not None and prev_ts is not None:
        ax.axvspan(
            start_ts,
            prev_ts,
            color=color,
            alpha=0.18,
            label=label if not labeled else None,
        )


def generate_visualizations(
    df: pd.DataFrame,
    config: PipelineConfig,
    figures_dir: Path | str,
) -> list[Path]:
    """Generate all 7 required engineering plots and save as PNG files."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    ts = pd.to_datetime(df["timestamp"])
    # Mask out-of-range readings when plotting trends so axis scales remain readable
    valid_range = ~df["dq_out_of_range"]

    # 1. Wind speed vs power output
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    normal_mask = valid_range & (~df["rule_fault_detected"]) & (~df["is_statistical_anomaly"])
    low_pwr_mask = valid_range & df["fault_low_power"]
    other_fault_mask = valid_range & df["rule_fault_detected"] & (~df["fault_low_power"])
    anom_only_mask = valid_range & (~df["rule_fault_detected"]) & df["is_statistical_anomaly"]

    ax.scatter(
        df.loc[normal_mask, "wind_speed_mps"],
        df.loc[normal_mask, "power_kw"],
        s=14,
        alpha=0.55,
        color="#2563eb",
        label="Normal / Idle",
    )
    if anom_only_mask.any():
        ax.scatter(
            df.loc[anom_only_mask, "wind_speed_mps"],
            df.loc[anom_only_mask, "power_kw"],
            s=28,
            color="#f59e0b",
            marker="^",
            label="Statistical Anomaly Only",
        )
    if other_fault_mask.any():
        ax.scatter(
            df.loc[other_fault_mask, "wind_speed_mps"],
            df.loc[other_fault_mask, "power_kw"],
            s=32,
            color="#9333ea",
            marker="s",
            label="Other Rule Fault",
        )
    if low_pwr_mask.any():
        ax.scatter(
            df.loc[low_pwr_mask, "wind_speed_mps"],
            df.loc[low_pwr_mask, "power_kw"],
            s=40,
            color="#dc2626",
            marker="X",
            label="Low Power Fault",
        )
    ax.axvline(
        config.rules.low_power_wind_speed_min_mps,
        color="#64748b",
        linestyle="--",
        linewidth=1.1,
        label=f"Min Wind for Power Check ({config.rules.low_power_wind_speed_min_mps} m/s)",
    )
    ax.axhline(
        config.rules.low_power_output_max_kw,
        color="#dc2626",
        linestyle=":",
        linewidth=1.1,
        label=f"Low Power Threshold ({config.rules.low_power_output_max_kw} kW)",
    )
    ax.set_title("Wind Speed vs. Electrical Power Output (Synthetic Demo Data)")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power Output (kW)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "01_wind_speed_vs_power.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_files.append(p1)

    # 2. Vibration trend over time
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=130)
    vib_plot = df["vibration_mm_s"].where(valid_range)
    ax.plot(ts, vib_plot, color="#0f766e", linewidth=1.2, label="Vibration (mm/s)")
    ax.axhline(
        config.rules.high_vibration_mm_s,
        color="#dc2626",
        linestyle="--",
        linewidth=1.3,
        label=f"High Vibration Threshold ({config.rules.high_vibration_mm_s} mm/s)",
    )
    _shade_fault_mask(ax, ts, df["fault_high_vibration"], "#dc2626", "High Vibration Fault")
    ax.set_title("Vibration Velocity Trend Over Time")
    ax.set_xlabel("Timestamp (HH:MM)")
    ax.set_ylabel("Vibration Velocity (mm/s)")
    _format_time_axis(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p2 = out_dir / "02_vibration_trend.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_files.append(p2)

    # 3. Temperature trend over time
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=130)
    temp_plot = df["temperature_c"].where(valid_range)
    ax.plot(ts, temp_plot, color="#ea580c", linewidth=1.2, label="Operating Temperature (°C)")
    ax.axhline(
        config.rules.high_temperature_c,
        color="#dc2626",
        linestyle="--",
        linewidth=1.3,
        label=f"High Temp Threshold ({config.rules.high_temperature_c} °C)",
    )
    _shade_fault_mask(ax, ts, df["fault_high_temperature"], "#dc2626", "High Temperature Fault")
    ax.set_title("Operating Temperature Trend Over Time")
    ax.set_xlabel("Timestamp (HH:MM)")
    ax.set_ylabel("Temperature (°C)")
    _format_time_axis(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p3 = out_dir / "03_temperature_trend.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_files.append(p3)

    # 4. Rotor speed trend over time
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=130)
    rotor_plot = df["rotor_speed_rpm"].where(valid_range)
    ax.plot(ts, rotor_plot, color="#1d4ed8", linewidth=1.2, label="Rotor Speed (RPM)")
    ax.axhline(
        config.rules.rotor_speed_operating_max_rpm,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"Max Operating Limit ({config.rules.rotor_speed_operating_max_rpm} RPM)",
    )
    ax.axhline(
        config.rules.rotor_speed_operating_min_rpm,
        color="#64748b",
        linestyle=":",
        linewidth=1.2,
        label=f"Min Operating Limit ({config.rules.rotor_speed_operating_min_rpm} RPM)",
    )
    _shade_fault_mask(
        ax, ts, df["fault_abnormal_rotor_speed"], "#dc2626", "Abnormal Rotor Speed Fault"
    )
    ax.set_title("Rotor Rotational Speed Trend Over Time")
    ax.set_xlabel("Timestamp (HH:MM)")
    ax.set_ylabel("Rotor Speed (RPM)")
    _format_time_axis(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p4 = out_dir / "04_rotor_speed_trend.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_files.append(p4)

    # 5. Fault detection timeline
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=130)
    lanes = [
        ("fault_high_vibration", "High Vibration", "#dc2626"),
        ("fault_high_temperature", "High Temperature", "#ea580c"),
        ("fault_low_power", "Low Power Output", "#7c3aed"),
        ("fault_abnormal_rotor_speed", "Abnormal Rotor Speed", "#0284c7"),
        ("fault_sensor_range", "Sensor Range Violation", "#111827"),
        ("is_statistical_anomaly", "Statistical Anomaly (Suspected)", "#f59e0b"),
    ]
    for y_idx, (col, label, color) in enumerate(lanes):
        active_ts = ts[df[col].astype(bool)]
        if len(active_ts) > 0:
            ax.scatter(active_ts, [y_idx] * len(active_ts), color=color, s=26, label=label)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([lbl for _, lbl, _ in lanes], fontsize=9)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_title("Fault & Statistical Anomaly Detection Timeline")
    ax.set_xlabel("Timestamp (HH:MM)")
    _format_time_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    p5 = out_dir / "05_fault_detection_timeline.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_files.append(p5)

    # 6. Sensor overview dashboard (3x2)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9.5), sharex=True, dpi=130)
    panels = [
        ("wind_speed_mps", "Wind Speed (m/s)", "#0284c7", None),
        ("power_kw", "Power Output (kW)", "#16a34a", config.rules.low_power_output_max_kw),
        ("vibration_mm_s", "Vibration (mm/s)", "#0f766e", config.rules.high_vibration_mm_s),
        ("temperature_c", "Temperature (°C)", "#ea580c", config.rules.high_temperature_c),
        ("rotor_speed_rpm", "Rotor Speed (RPM)", "#1d4ed8", config.rules.rotor_speed_operating_max_rpm),
        ("pitch_angle_deg", "Pitch Angle (deg)", "#6d28d9", None),
    ]
    for ax, (col, ylabel, color, thresh) in zip(axes.flat, panels):
        series_plot = df[col].where(valid_range)
        ax.plot(ts, series_plot, color=color, linewidth=1.0, label=ylabel)
        if thresh is not None:
            ax.axhline(thresh, color="#dc2626", linestyle="--", linewidth=0.9, label=f"Threshold ({thresh})")
        _shade_fault_mask(ax, ts, df["rule_fault_detected"], "#dc2626", "Rule Fault Period")
        ax.set_ylabel(ylabel, fontsize=9)
        _format_time_axis(ax)
        ax.legend(loc="upper right", fontsize=7)

    axes[2, 0].set_xlabel("Timestamp (HH:MM)")
    axes[2, 1].set_xlabel("Timestamp (HH:MM)")
    fig.suptitle("Wind Turbine Sensor Overview Dashboard (Synthetic Demo Data)", fontsize=13, y=0.995)
    fig.tight_layout()
    p6 = out_dir / "06_sensor_overview_dashboard.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_files.append(p6)

    # 7. Anomaly score trend
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=130)
    palette = ["#2563eb", "#0f766e", "#ea580c", "#7c3aed"]
    for col, color in zip(config.anomaly.monitored_columns, palette):
        sc_col = f"anomaly_score_{col}"
        if sc_col in df.columns:
            ax.plot(ts, df[sc_col], linewidth=1.0, alpha=0.8, color=color, label=col)
    ax.axhline(
        config.anomaly.z_threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.4,
        label=f"Anomaly Threshold (Z = {config.anomaly.z_threshold})",
    )
    ax.set_title("Causal Rolling-MAD Statistical Anomaly Score Trend")
    ax.set_xlabel("Timestamp (HH:MM)")
    ax.set_ylabel("Modified Z-Score (MAD)")
    _format_time_axis(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p7 = out_dir / "07_anomaly_score_trend.png"
    fig.savefig(p7)
    plt.close(fig)
    generated_files.append(p7)

    logger.info("Generated %d visualization PNG file(s) in %s", len(generated_files), out_dir)
    return generated_files
