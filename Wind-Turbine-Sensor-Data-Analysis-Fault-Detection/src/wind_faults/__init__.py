"""Wind Turbine Sensor Data Analysis & Fault Detection package."""

from wind_faults.config import PipelineConfig, load_config
from wind_faults.pipeline import PipelineResult, run_pipeline

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "load_config",
    "run_pipeline",
]

__version__ = "0.1.0"
