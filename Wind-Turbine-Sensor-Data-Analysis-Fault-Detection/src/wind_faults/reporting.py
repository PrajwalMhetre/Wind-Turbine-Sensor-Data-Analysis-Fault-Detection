"""Automated JSON summary and Markdown engineering report generation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from wind_faults.anomaly_detection import AnomalySummary
from wind_faults.config import PipelineConfig
from wind_faults.data_quality import DataQualityReport
from wind_faults.fault_detection import FaultEvent
from wind_faults.preprocessing import PreprocessingResult

logger = logging.getLogger(__name__)


def build_summary_dict(
    input_csv: Path,
    df: pd.DataFrame,
    dq_report: DataQualityReport,
    prep_result: PreprocessingResult,
    fault_events: list[FaultEvent],
    anomaly_summary: AnomalySummary,
    config: PipelineConfig,
    figure_paths: list[Path],
) -> dict[str, Any]:
    """Assemble machine-readable pipeline summary dictionary from actual results."""
    category_counts = {
        "HIGH_VIBRATION": int(df["fault_high_vibration"].sum()),
        "HIGH_TEMPERATURE": int(df["fault_high_temperature"].sum()),
        "LOW_POWER_OUTPUT": int(df["fault_low_power"].sum()),
        "ABNORMAL_ROTOR_SPEED": int(df["fault_abnormal_rotor_speed"].sum()),
        "SENSOR_RANGE_VIOLATION": int(df["fault_sensor_range"].sum()),
    }
    severity_counts = {
        sev: int((df["fault_severity"] == sev).sum())
        for sev in ("CRITICAL", "HIGH", "WARNING", "NONE")
    }
    simultaneous_count = int(
        (
            df[
                [
                    "fault_high_vibration",
                    "fault_high_temperature",
                    "fault_low_power",
                    "fault_abnormal_rotor_speed",
                    "fault_sensor_range",
                ]
            ].sum(axis=1)
            >= 2
        ).sum()
    )

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_file": str(input_csv),
            "data_origin": "Synthetic demonstration dataset (unless replaced with real SCADA CSV)",
            "thresholds_disclaimer": "Configured thresholds are illustrative engineering assumptions, not universal OEM specs.",
            "time_range_start": ts_min.isoformat() if pd.notna(ts_min) else None,
            "time_range_end": ts_max.isoformat() if pd.notna(ts_max) else None,
            "records_analyzed": len(df),
        },
        "data_quality": dq_report.to_dict(),
        "preprocessing": prep_result.to_summary_dict(),
        "rule_based_faults": {
            "total_faulty_timestamps": int(df["rule_fault_detected"].sum()),
            "total_fault_events": len(fault_events),
            "simultaneous_fault_timestamps": simultaneous_count,
            "counts_by_category": category_counts,
            "counts_by_severity": severity_counts,
            "configured_thresholds": {
                "high_vibration_mm_s": config.rules.high_vibration_mm_s,
                "high_temperature_c": config.rules.high_temperature_c,
                "low_power_wind_speed_min_mps": config.rules.low_power_wind_speed_min_mps,
                "low_power_output_max_kw": config.rules.low_power_output_max_kw,
                "rotor_speed_operating_min_rpm": config.rules.rotor_speed_operating_min_rpm,
                "rotor_speed_operating_max_rpm": config.rules.rotor_speed_operating_max_rpm,
            },
        },
        "statistical_anomalies": anomaly_summary.to_dict(),
        "generated_figures": [p.name for p in figure_paths],
    }


def write_summary_json(summary: dict[str, Any], output_path: Path | str) -> Path:
    """Write machine-readable summary JSON to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved summary JSON to %s", path)
    return path


def write_analysis_report(
    summary: dict[str, Any],
    df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Generate automated Markdown engineering report from actual pipeline results."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = summary["metadata"]
    dq = summary["data_quality"]
    prep = summary["preprocessing"]
    rf = summary["rule_based_faults"]
    anom = summary["statistical_anomalies"]
    cat_counts = rf["counts_by_category"]
    sev_counts = rf["counts_by_severity"]

    # Compute factual observations from processed df
    valid_df = df.loc[~df["dq_out_of_range"]]
    max_vib = float(valid_df["vibration_mm_s"].max()) if valid_df["vibration_mm_s"].notna().any() else 0.0
    max_temp = float(valid_df["temperature_c"].max()) if valid_df["temperature_c"].notna().any() else 0.0
    max_pwr = float(valid_df["power_kw"].max()) if valid_df["power_kw"].notna().any() else 0.0
    idle_rows = int((df["status"] == "IDLE").sum())
    operating_rows = int((df["status"] == "OPERATING").sum())

    missing_rows_md = "\n".join(
        f"| `{col}` | {cnt} |" for col, cnt in dq["missing_values_per_column"].items()
    )
    cat_rows_md = "\n".join(
        f"| `{cat}` | {cnt} |" for cat, cnt in cat_counts.items()
    )
    anom_sensor_md = "\n".join(
        f"| `{col}` | {cnt} |" for col, cnt in anom["anomalous_rows_per_sensor"].items()
    )
    prep_actions_md = "\n".join(f"- {act}" for act in prep["actions_taken"])
    dq_notes_md = (
        "\n".join(f"- {note}" for note in dq["notes"])
        if dq["notes"]
        else "- No structural or range anomalies detected."
    )
    figs_md = "\n".join(
        f"- `{name}`: `reports/figures/{name}`\n\n  ![{name}](figures/{name})"
        for name in summary["generated_figures"]
    )

    report_text = f"""# Wind Turbine Condition Monitoring & Fault Detection Report

**Generated (UTC):** {meta["generated_at_utc"]}  
**Input Dataset:** `{meta["input_file"]}`  
**Analysis Window:** `{meta["time_range_start"]}` to `{meta["time_range_end"]}`  

> **Scope & Data Origin Notice:** {meta["data_origin"]}. {meta["thresholds_disclaimer"]} Statistical anomaly scores highlight suspected unusual operating behavior relative to recent rolling baselines and do **not** represent confirmed mechanical failures without physical inspection.

---

## 1. Project Overview
This report summarizes the automated execution of the **Wind Turbine Sensor Data Analysis & Fault Detection** pipeline. The pipeline validates raw sensor CSV data, audits data quality, applies transparent preprocessing, evaluates configurable rule-based engineering fault conditions, computes causal rolling-median / MAD statistical anomaly scores, and generates diagnostic visualizations.

## 2. Dataset Description & Records Analyzed
- **Raw Records Loaded:** {dq["total_rows"]}
- **Records Analyzed (Post-Preprocessing):** {meta["records_analyzed"]}
- **Operating State Breakdown:** `OPERATING` = {operating_rows} records, `IDLE` = {idle_rows} records
- **Schema Status:** {"Valid" if dq["is_valid_schema"] else "Invalid"}

## 3. Data-Quality Findings & Preprocessing
- **Invalid Timestamps:** {dq["invalid_timestamp_rows"]}
- **Duplicate Timestamps Detected:** {dq["duplicate_timestamp_count"]}
- **Total Missing Values:** {dq["total_missing_values"]}
- **Out-of-Range Physical Readings:** {dq["total_out_of_range_readings"]}

**Quality Notes:**
{dq_notes_md}

**Preprocessing Actions Applied:**
{prep_actions_md}

### Missing-Value Statistics
| Column | Missing Count |
| --- | ---: |
{missing_rows_md}

## 4. Rule-Based Fault Detection Results
- **Timestamps with at Least One Rule Fault:** {rf["total_faulty_timestamps"]}
- **Total Individual Fault Events:** {rf["total_fault_events"]}
- **Timestamps with Simultaneous Multi-Faults:** {rf["simultaneous_fault_timestamps"]}
- **Severity Breakdown:** CRITICAL = {sev_counts["CRITICAL"]}, HIGH = {sev_counts["HIGH"]}, WARNING = {sev_counts["WARNING"]}, NONE = {sev_counts["NONE"]}

### Fault Categories
| Fault Category | Flagged Timestamps |
| --- | ---: |
{cat_rows_md}

## 5. Statistical Anomaly Detection (Causal Rolling MAD)
- **Rolling Window:** {anom["rolling_window"]} records (`min_periods={anom["min_periods"]}`)
- **Modified Z-Score Threshold:** {anom["z_threshold"]}
- **Total Suspected Anomalous Timestamps:** {anom["total_anomalous_rows"]}
- **Maximum Observed Anomaly Score:** {anom["max_observed_score"]}

| Monitored Sensor | Suspected Anomalous Timestamps |
| --- | ---: |
{anom_sensor_md}

## 6. Key Engineering Observations
- Within valid physical sensor bounds, peak vibration reached **{max_vib:.2f} mm/s**, peak operating temperature reached **{max_temp:.2f} °C**, and peak power output reached **{max_pwr:.2f} kW**.
- Normal low-wind `IDLE` periods ({idle_rows} records) were excluded from low-power fault rules, preventing false-positive curtailment alerts during calm wind conditions.
- Physically impossible sensor readings ({dq["total_out_of_range_readings"]} readings) were flagged as `SENSOR_RANGE_VIOLATION` while preserving raw values and excluding them from rolling MAD baselines.

## 7. Generated Visualizations
{figs_md}

## 8. Limitations
- Demonstrated on a deterministic synthetic dataset rather than live industrial SCADA telemetry.
- Fixed univariate thresholds do not capture complex multivariate aerodynamic or pitch-curve interactions across varying air density or turbulence regimes.
- Causal rolling MAD adapts to local windows; sustained gradual drift longer than the rolling window may be absorbed into the baseline.

## 9. Recommended Next Steps
1. Calibrate thresholds in `config/thresholds.json` against OEM turbine power curves and site-specific SCADA historical baselines.
2. Integrate real 10-minute or 1-minute SCADA CSV exports into `data/raw/` and run `python scripts/run_analysis.py --input <path>`.
3. Add multivariate power-curve residual modeling (e.g., binned wind-speed vs. expected power) once historical normal-operation SCADA data is available.
"""

    path.write_text(report_text, encoding="utf-8")
    logger.info("Saved Markdown analysis report to %s", path)
    return path
