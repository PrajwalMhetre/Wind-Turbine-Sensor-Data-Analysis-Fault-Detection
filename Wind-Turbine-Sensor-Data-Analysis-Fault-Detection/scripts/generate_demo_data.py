"""CLI script to generate the deterministic synthetic wind turbine sensor dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wind_faults.synthetic_data import get_scenario_manifest, save_synthetic_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic wind turbine SCADA sensor CSV data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "wind_turbine_sensor_data.csv",
        help="Output CSV path (default: data/raw/wind_turbine_sensor_data.csv)",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=1440,
        help="Number of 1-minute records to generate (default: 1440)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    out_path = save_synthetic_dataset(
        output_path=args.output,
        num_minutes=args.minutes,
        seed=args.seed,
    )
    logging.info("Documented synthetic scenarios (%d):", len(get_scenario_manifest()))
    for sc in get_scenario_manifest():
        logging.info("  - [%s] %s", sc["category"], sc["description"])
    logging.info("Dataset ready at: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
