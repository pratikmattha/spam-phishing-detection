"""
Project paths and settings.
All scripts import paths from here so the project works on any machine.
"""

from pathlib import Path

# Root of the project (the folder this file lives in)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

# Output folders
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"

# Reproducibility
RANDOM_SEED = 42