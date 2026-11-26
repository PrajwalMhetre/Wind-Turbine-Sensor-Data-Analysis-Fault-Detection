"""CLI script to run the wind turbine sensor data analysis & fault detection pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wind_faults.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wind turbine sensor data analysis, fault detection, and reporting."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "wind_turbine_sensor_data.csv",
        help="Input sensor CSV path (default: data/raw/wind_turbine_sensor_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT,
        help="Base output directory for processed data, reports, and figures (default: repo root)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "thresholds.json",
        help="Path to thresholds JSON config (default: config/thresholds.json)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    result = run_pipeline(
        input_csv=args.input,
        output_dir=args.output_dir,
        config_path=args.config if args.config.exists() else None,
    )
    logging.info("Processed CSV : %s", result.processed_csv_path)
    logging.info("Summary JSON  : %s", result.summary_json_path)
    logging.info("Report MD     : %s", result.report_md_path)
    logging.info("Figures (%d)  : %s", len(result.figure_paths), result.figure_paths[0].parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
