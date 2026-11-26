"""End-to-end orchestration pipeline for wind turbine sensor data analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from wind_faults.anomaly_detection import AnomalySummary, detect_statistical_anomalies
from wind_faults.config import PipelineConfig, load_config
from wind_faults.data_loader import load_sensor_csv
from wind_faults.data_quality import DataQualityReport, assess_data_quality
from wind_faults.fault_detection import FaultEvent, detect_rule_based_faults
from wind_faults.preprocessing import PreprocessingResult, preprocess_sensor_data
from wind_faults.reporting import (
    build_summary_dict,
    write_analysis_report,
    write_summary_json,
)
from wind_faults.visualization import generate_visualizations

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for all outputs produced by a pipeline execution."""

    processed_df: pd.DataFrame
    data_quality_report: DataQualityReport
    preprocessing_result: PreprocessingResult
    fault_events: list[FaultEvent]
    anomaly_summary: AnomalySummary
    summary_dict: dict[str, Any]
    processed_csv_path: Path
    summary_json_path: Path
    report_md_path: Path
    figure_paths: list[Path]


def run_pipeline(
    input_csv: Path | str,
    output_dir: Path | str = ".",
    config_path: Path | str | None = None,
) -> PipelineResult:
    """Execute the full condition-monitoring pipeline from CSV input to reports and plots."""
    input_path = Path(input_csv)
    base_out = Path(output_dir)
    config: PipelineConfig = load_config(config_path)

    logger.info("Starting wind turbine fault detection pipeline on %s", input_path)

    # 1. Load raw CSV (original file is never modified)
    raw_df = load_sensor_csv(input_path)

    # 2. Data quality assessment
    assessed_df, dq_report = assess_data_quality(raw_df, config.sensor_limits)

    # 3. Transparent preprocessing
    prep_result = preprocess_sensor_data(assessed_df, config.preprocessing)

    # 4. Rule-based fault detection
    fault_df, fault_events = detect_rule_based_faults(prep_result.df, config)

    # 5. Statistical anomaly detection (causal rolling MAD)
    final_df, anomaly_summary = detect_statistical_anomalies(
        fault_df, config.anomaly, config.sensor_limits
    )

    # 6. Save processed CSV
    processed_csv_path = base_out / "data" / "processed" / "fault_detection_results.csv"
    processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(processed_csv_path, index=False)
    logger.info("Saved processed results CSV to %s", processed_csv_path)

    # 7. Generate visualizations
    figures_dir = base_out / "reports" / "figures"
    figure_paths = generate_visualizations(final_df, config, figures_dir)

    # 8. Build summary & write JSON + Markdown report
    summary_dict = build_summary_dict(
        input_csv=input_path,
        df=final_df,
        dq_report=dq_report,
        prep_result=prep_result,
        fault_events=fault_events,
        anomaly_summary=anomaly_summary,
        config=config,
        figure_paths=figure_paths,
    )
    summary_json_path = write_summary_json(
        summary_dict, base_out / "reports" / "summary.json"
    )
    report_md_path = write_analysis_report(
        summary_dict, final_df, base_out / "reports" / "analysis_report.md"
    )

    logger.info("Pipeline execution completed successfully.")
    return PipelineResult(
        processed_df=final_df,
        data_quality_report=dq_report,
        preprocessing_result=prep_result,
        fault_events=fault_events,
        anomaly_summary=anomaly_summary,
        summary_dict=summary_dict,
        processed_csv_path=processed_csv_path,
        summary_json_path=summary_json_path,
        report_md_path=report_md_path,
        figure_paths=figure_paths,
    )
