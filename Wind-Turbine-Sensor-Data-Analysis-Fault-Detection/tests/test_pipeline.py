"""End-to-end integration tests for the wind turbine fault detection pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from wind_faults.pipeline import run_pipeline
from wind_faults.synthetic_data import save_synthetic_dataset


def test_end_to_end_pipeline_execution(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw_demo.csv"
    save_synthetic_dataset(raw_csv, num_minutes=1440, seed=42)

    result = run_pipeline(input_csv=raw_csv, output_dir=tmp_path)

    assert result.processed_csv_path.exists()
    assert result.summary_json_path.exists()
    assert result.report_md_path.exists()
    assert len(result.figure_paths) == 7
    for fig_path in result.figure_paths:
        assert fig_path.exists() and fig_path.stat().st_size > 0

    # Verify processed dataset has 1,440 deduplicated rows and expected columns
    assert len(result.processed_df) == 1440
    for required_col in (
        "dq_out_of_range",
        "rule_fault_detected",
        "fault_categories",
        "fault_severity",
        "max_anomaly_score",
        "is_statistical_anomaly",
    ):
        assert required_col in result.processed_df.columns

    # Verify summary JSON matches actual counts
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary["metadata"]["records_analyzed"] == 1440
    assert summary["data_quality"]["duplicate_timestamp_count"] == 2
    assert summary["rule_based_faults"]["counts_by_category"]["HIGH_VIBRATION"] > 0
    assert summary["rule_based_faults"]["counts_by_category"]["HIGH_TEMPERATURE"] > 0
    assert summary["rule_based_faults"]["counts_by_category"]["LOW_POWER_OUTPUT"] > 0
    assert summary["rule_based_faults"]["counts_by_category"]["ABNORMAL_ROTOR_SPEED"] > 0
    assert summary["rule_based_faults"]["counts_by_category"]["SENSOR_RANGE_VIOLATION"] == 4
    assert summary["statistical_anomalies"]["total_anomalous_rows"] > 0

    # Verify Markdown report includes key sections and no hardcoded mismatch
    report_text = result.report_md_path.read_text(encoding="utf-8")
    assert "Wind Turbine Condition Monitoring & Fault Detection Report" in report_text
    assert "1440" in report_text
    assert "06_sensor_overview_dashboard.png" in report_text
