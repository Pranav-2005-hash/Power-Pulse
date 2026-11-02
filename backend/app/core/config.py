import os
from pathlib import Path

# Resolve base directories safely whether running from repo root or backend/
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = ROOT_DIR.parent if ROOT_DIR.name == "backend" else ROOT_DIR

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR = ARTIFACTS_DIR / "models"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "household_power_sample.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "hourly_consumption.csv"

# Ensure all vital directories exist
for directory in [DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, MODELS_DIR, SCALERS_DIR, 
                  DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "sample"]:
    os.makedirs(directory, exist_ok=True)

# Deep Learning Parameters per prompt specification
SEQUENCE_LENGTH = 24  # 24-hour input window
PREDICTION_STEP = 1   # Predict next 1 hour
TRAIN_SPLIT = 0.70    # 70% chronological
VAL_SPLIT = 0.15      # 15% chronological
TEST_SPLIT = 0.15     # 15% chronological

RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 15

# Default Anomaly Thresholds
DEFAULT_THRESHOLD_PERCENTILE = 95.0
