import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

from app.core.config import (
    ARTIFACTS_DIR, MODELS_DIR, SCALERS_DIR, SAMPLE_DATA_PATH, PROCESSED_DATA_PATH
)
from app.ml.preprocessing import (
    parse_uci_or_standard_csv, resample_to_hourly_mean, prepare_dataset_splits
)

class StateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StateManager, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.state_lock = threading.Lock()
        
        # Dataset state
        self.is_dataset_loaded = False
        self.source_type = "none"
        self.dataset_filename = None
        self.hourly_df: Optional[pd.DataFrame] = None
        self.raw_metadata: Dict[str, Any] = {}
        self.resample_stats: Dict[str, Any] = {}
        self.splits: Optional[Dict[str, np.ndarray]] = None
        self.scaler = None
        self.split_info: Dict[str, Any] = {}

        # Model state
        self.models_trained = False
        self.keras_models = {}
        self.metrics = {}
        self.best_model_name: Optional[str] = None
        self.selection_reason: str = "Awaiting training completion"
        self.validation_thresholds: Dict[float, float] = {}

        # Training task tracking
        self.train_status = {
            "status": "idle",
            "progress_percentage": 0.0,
            "current_model": None,
            "current_epoch": None,
            "total_epochs": None,
            "error_message": None,
            "started_at": None,
            "completed_at": None
        }

        # Anomaly reviews storage
        self.reviews_file = os.path.join(ARTIFACTS_DIR, "anomaly_reviews.json")
        self.anomaly_reviews = self._load_reviews()

        # Check for previously saved state
        self._try_restore_saved_state()

    def _load_reviews(self) -> Dict[str, bool]:
        if os.path.exists(self.reviews_file):
            try:
                with open(self.reviews_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_review(self, anomaly_id: str, reviewed: bool):
        with self.state_lock:
            self.anomaly_reviews[anomaly_id] = reviewed
            try:
                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                with open(self.reviews_file, 'w') as f:
                    json.dump(self.anomaly_reviews, f, indent=2)
            except Exception as e:
                print(f"Error saving review: {e}")

    def load_dataset_from_file(
        self,
        file_path: str,
        source_type: str = "uploaded",
        timestamp_col: Optional[str] = None,
        consumption_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Loads CSV, runs cleaning & hourly resampling, splits data and fits scaler.
        Invalidates any previously trained models from older datasets.
        """
        with self.state_lock:
            cleaned_df, raw_meta = parse_uci_or_standard_csv(
                file_path, timestamp_col, consumption_col
            )
            hourly_df, resample_stats = resample_to_hourly_mean(cleaned_df)
            
            # Save processed hourly CSV
            os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
            hourly_df.to_csv(PROCESSED_DATA_PATH, index=False)

            # Prepare chronological splits and fit scaler
            splits, scaler, split_info = prepare_dataset_splits(hourly_df)

            self.hourly_df = hourly_df
            self.raw_metadata = raw_meta
            self.resample_stats = resample_stats
            self.splits = splits
            self.scaler = scaler
            self.split_info = split_info
            self.is_dataset_loaded = True
            self.source_type = source_type
            self.dataset_filename = os.path.basename(file_path)

            # Invalidate previous models since dataset changed
            self.models_trained = False
            self.keras_models = {}
            self.metrics = {}
            self.best_model_name = None
            self.selection_reason = "Dataset changed. Please retrain models."
            self.validation_thresholds = {}

            return {
                "raw_metadata": raw_meta,
                "resample_stats": resample_stats,
                "split_info": split_info
            }

    def _try_restore_saved_state(self):
        """Attempts to restore state if files exist on startup."""
        try:
            # Check if processed data exists
            if os.path.exists(PROCESSED_DATA_PATH):
                hourly = pd.read_csv(PROCESSED_DATA_PATH)
                hourly['timestamp'] = pd.to_datetime(hourly['timestamp'])
                splits, scaler, split_info = prepare_dataset_splits(hourly)
                self.hourly_df = hourly
                self.splits = splits
                self.scaler = scaler
                self.split_info = split_info
                self.is_dataset_loaded = True
                self.source_type = "cached"
                self.dataset_filename = "hourly_consumption.csv"

            # Check if models & metadata exist
            metadata_file = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
            lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
            gru_path = os.path.join(MODELS_DIR, "gru_model.keras")

            if os.path.exists(metadata_file) and os.path.exists(lstm_path) and os.path.exists(gru_path):
                import tensorflow as tf
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                self.keras_models["LSTM"] = tf.keras.models.load_model(lstm_path)
                self.keras_models["GRU"] = tf.keras.models.load_model(gru_path)
                self.metrics = meta.get("metrics", {})
                self.best_model_name = meta.get("best_model_name")
                self.selection_reason = meta.get("selection_reason", "")
                self.validation_thresholds = {float(k): float(v) for k, v in meta.get("validation_thresholds", {}).items()}
                self.models_trained = True
                print("Successfully restored existing trained models and metadata.")
        except Exception as e:
            print(f"Note: Could not restore previous state ({e}). Starting fresh.")

    def save_trained_state(self, metrics: Dict[str, Any], best_model_name: str, selection_reason: str, thresholds: Dict[float, float]):
        with self.state_lock:
            self.metrics = metrics
            self.best_model_name = best_model_name
            self.selection_reason = selection_reason
            self.validation_thresholds = thresholds
            self.models_trained = True

            meta_data = {
                "best_model_name": best_model_name,
                "selection_reason": selection_reason,
                "validation_thresholds": {str(k): v for k, v in thresholds.items()},
                "metrics": {
                    name: {k: v for k, v in m.items() if k not in ["val_residuals", "test_predictions", "test_actuals"]}
                    for name, m in metrics.items()
                },
                "saved_at": datetime.now().isoformat()
            }
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"), 'w') as f:
                json.dump(meta_data, f, indent=2)

state_manager = StateManager()
