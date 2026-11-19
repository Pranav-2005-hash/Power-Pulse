import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import joblib

from app.core.config import (
    ARTIFACTS_DIR, MODELS_DIR, SCALERS_DIR, SAMPLE_DATA_PATH, PROCESSED_DATA_PATH,
    DATA_DIR, SEQUENCE_LENGTH, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)
from app.ml.preprocessing import (
    parse_uci_or_standard_csv, resample_to_hourly_mean, prepare_dataset_splits
)
from app.ml.models import train_single_model, evaluate_model_on_splits
from app.ml.anomaly import compute_validation_thresholds, detect_anomalies_from_residuals


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

        # Raw Dataset Info
        self.raw_file_path: Optional[str] = None
        self.raw_filename: Optional[str] = None
        self.raw_metadata: Dict[str, Any] = {}
        self.is_dataset_loaded: bool = False

        # Preprocessed Data
        self.is_preprocessed: bool = False
        self.hourly_df: Optional[pd.DataFrame] = None
        self.preprocessing_summary: Dict[str, Any] = {}
        self.splits: Optional[Dict[str, np.ndarray]] = None
        self.scaler = None
        self.split_info: Dict[str, Any] = {}
        self.active_target_col: str = "Global_active_power"

        # Models & Results
        self.is_trained: bool = False
        self.keras_models: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.current_model_name: Optional[str] = None
        self.comparison_table: List[Dict[str, Any]] = []
        self.best_model_name: Optional[str] = None
        self.best_model_reason: Optional[str] = None

        # Predictions & Anomalies
        self.predictions: List[Dict[str, Any]] = []
        self.anomalies: List[Dict[str, Any]] = []
        self.anomaly_threshold_kw: float = 0.0
        self.validation_thresholds: Dict[float, float] = {}

        # 24-Hour Forward Forecast
        self.forecast_24h: Optional[Dict[str, Any]] = None

    def save_raw_upload(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Inspects raw uploaded CSV, extracts column types, preview of first 10 rows,
        and auto-detects target and datetime columns.
        """
        with self.state_lock:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Detect delimiter
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                sep = ';' if ';' in first_line else ','

            df_raw = pd.read_csv(
                file_path,
                sep=sep,
                na_values=['?', 'NA', 'null', 'None', ''],
                nrows=1000,
                low_memory=False
            )

            if len(df_raw) == 0:
                raise ValueError("Uploaded CSV file is empty.")

            # Fast row counting using binary chunk scanning (50x faster on large datasets)
            try:
                with open(file_path, 'rb') as f:
                    total_rows = sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1024 * 1024), b''))
            except Exception:
                total_rows = len(df_raw)

            file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
            columns = [str(c) for c in df_raw.columns]

            # Detect numeric columns
            numeric_cols = []
            for col in columns:
                series = pd.to_numeric(df_raw[col], errors='coerce')
                if series.notna().sum() > len(df_raw) * 0.4:
                    numeric_cols.append(col)

            # Auto-detect target column
            detected_target = None
            if 'Global_active_power' in columns:
                detected_target = 'Global_active_power'
            elif 'consumption' in [c.lower() for c in columns]:
                detected_target = [c for c in columns if c.lower() == 'consumption'][0]
            elif 'power' in [c.lower() for c in columns]:
                detected_target = [c for c in columns if c.lower() == 'power'][0]
            elif numeric_cols:
                detected_target = numeric_cols[0]

            # Auto-detect datetime column
            detected_dt = None
            if 'Date' in columns and 'Time' in columns:
                detected_dt = 'Date & Time'
            elif 'timestamp' in [c.lower() for c in columns]:
                detected_dt = [c for c in columns if c.lower() == 'timestamp'][0]
            elif 'datetime' in [c.lower() for c in columns]:
                detected_dt = [c for c in columns if c.lower() == 'datetime'][0]

            # Preview first 10 rows
            preview_rows = df_raw.head(10).replace({np.nan: None}).to_dict(orient="records")

            self.raw_file_path = file_path
            self.raw_filename = filename
            self.raw_metadata = {
                "filename": filename,
                "file_size_kb": file_size_kb,
                "total_rows": max(total_rows, len(df_raw)),
                "total_columns": len(columns),
                "columns": columns,
                "numeric_columns": numeric_cols,
                "preview": preview_rows,
                "detected_target_col": detected_target,
                "detected_datetime_col": detected_dt,
                "delimiter": sep,
                "is_valid": True,
                "message": f"Successfully validated '{filename}' ({total_rows} rows, {len(columns)} columns)."
            }

            self.is_dataset_loaded = True
            # Reset downstream states
            self.is_preprocessed = False
            self.is_trained = False
            self.predictions = []
            self.anomalies = []
            self.comparison_table = []
            self.forecast_24h = None

            return self.raw_metadata

    def preprocess_dataset(
        self,
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cleans data, handles missing values, resamples to hourly mean,
        performs strict chronological 70/15/15 split, fits MinMaxScaler on train ONLY.
        """
        with self.state_lock:
            if not self.raw_file_path or not os.path.exists(self.raw_file_path):
                raise ValueError("No dataset uploaded yet. Please upload a CSV first.")

            # Determine target column
            chosen_target = target_col or self.raw_metadata.get("detected_target_col") or "Global_active_power"
            self.active_target_col = chosen_target

            # 1. Parse and clean
            cleaned_df, raw_meta = parse_uci_or_standard_csv(
                file_path=self.raw_file_path,
                timestamp_col=timestamp_col,
                consumption_col=chosen_target
            )

            # 2. Resample to 1-hour intervals (mean)
            hourly_df, resample_stats = resample_to_hourly_mean(cleaned_df)

            # Save processed hourly CSV
            os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
            hourly_df.to_csv(PROCESSED_DATA_PATH, index=False)

            # 3. Chronological splits (70% Train, 15% Val, 15% Test) with Scaler fitted on Train ONLY
            splits, scaler, split_info = prepare_dataset_splits(hourly_df)

            self.hourly_df = hourly_df
            self.splits = splits
            self.scaler = scaler
            self.split_info = split_info

            summary = {
                "status": "success",
                "message": "Dataset successfully cleaned, resampled to 1-hour intervals, and split chronologically.",
                "target_column": chosen_target,
                "initial_raw_rows": raw_meta.get("initial_rows", len(cleaned_df)),
                "valid_raw_rows": raw_meta.get("valid_rows_after_cleaning", len(cleaned_df)),
                "hourly_records": len(hourly_df),
                "train_samples": split_info["train_samples"],
                "val_samples": split_info["val_samples"],
                "test_samples": split_info["test_samples"],
                "input_sequence_length": SEQUENCE_LENGTH,
                "date_range": {
                    "start": resample_stats.get("start_timestamp", ""),
                    "end": resample_stats.get("end_timestamp", "")
                },
                "mean_kw": round(resample_stats.get("mean_kw", 0.0), 3),
                "min_kw": round(resample_stats.get("min_kw", 0.0), 3),
                "max_kw": round(resample_stats.get("max_kw", 0.0), 3),
                "scaler_train_min": round(split_info.get("scaler_data_min", 0.0), 4),
                "scaler_train_max": round(split_info.get("scaler_data_max", 0.0), 4),
                "missing_values_handled": raw_meta.get("missing_consumption_raw", 0) + resample_stats.get("interpolated_short_gaps", 0),
                "split_ratio": "70% Train / 15% Validation / 15% Test"
            }

            self.preprocessing_summary = summary
            self.is_preprocessed = True
            self.is_trained = False
            self.predictions = []
            self.anomalies = []
            self.comparison_table = []
            self.forecast_24h = None

            return summary

    def train_single_architecture(
        self,
        model_type: str,
        epochs: int = 12,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Trains real LSTM or GRU model with early stopping, computes genuine kW metrics,
        derives 95th percentile validation positive residual threshold,
        and detects anomalies on test set.
        """
        with self.state_lock:
            if not self.is_preprocessed or self.splits is None:
                # Auto-preprocess if dataset is loaded
                if self.is_dataset_loaded and self.raw_file_path:
                    self.preprocess_dataset()
                else:
                    raise ValueError("No dataset loaded or preprocessed.")

            splits = self.splits
            scaler = self.scaler

            # Train model
            model, training_info = train_single_model(
                model_type=model_type.lower(),
                X_train=splits["X_train"],
                y_train=splits["y_train"],
                X_val=splits["X_val"],
                y_val=splits["y_val"],
                epochs=epochs,
                batch_size=batch_size,
                patience=3
            )

            # Evaluate in genuine kW units
            eval_metrics = evaluate_model_on_splits(
                model=model,
                model_name=model_type.upper(),
                scaler=scaler,
                splits=splits,
                training_info=training_info
            )

            # Derive anomaly threshold from positive validation residuals (95th percentile)
            thresholds = compute_validation_thresholds(eval_metrics["val_residuals"])
            thresh_95 = thresholds.get(95.0, 0.5)

            # Detect anomalies on Test set
            test_timestamps = [str(ts) for ts in splits["ts_test"]]
            test_actuals = eval_metrics["test_actuals"]
            test_preds = eval_metrics["test_predictions"]

            anomalies_list, anom_summary = detect_anomalies_from_residuals(
                timestamps=test_timestamps,
                actuals=test_actuals,
                predictions=test_preds,
                validation_thresholds=thresholds,
                percentile=95.0
            )

            # Build prediction points
            predictions_list = []
            for ts, act, pred in zip(test_timestamps, test_actuals, test_preds):
                res = act - pred
                is_anom = res > thresh_95
                predictions_list.append({
                    "timestamp": ts,
                    "actual": round(float(act), 3),
                    "predicted": round(float(pred), 3),
                    "residual": round(float(res), 3),
                    "is_anomaly": bool(is_anom)
                })

            # Save state
            self.keras_models[model_type.upper()] = model
            self.metrics[model_type.upper()] = eval_metrics
            self.current_model_name = model_type.upper()
            self.predictions = predictions_list
            self.anomalies = anomalies_list
            self.anomaly_threshold_kw = thresh_95
            self.validation_thresholds = thresholds
            self.is_trained = True

            # Generate 24-hour forward forecast using this trained model
            forecast_24h = self._compute_24h_forecast(
                model=model,
                model_name=model_type.upper(),
                test_rmse=eval_metrics["test_rmse"]
            )

            return {
                "model_name": model_type.upper(),
                "metrics": {
                    "mae": eval_metrics["test_mae"],
                    "rmse": eval_metrics["test_rmse"],
                    "val_mae": eval_metrics["val_mae"],
                    "val_rmse": eval_metrics["val_rmse"],
                    "test_r2": eval_metrics.get("test_r2", 0.0),
                    "training_time_seconds": eval_metrics["training_duration_seconds"],
                    "epochs_trained": eval_metrics["epochs_trained"],
                    "parameters_count": eval_metrics["architecture"]["total_parameters"]
                },
                "anomaly_threshold_kw": round(thresh_95, 4),
                "total_anomalies": len(anomalies_list),
                "sample_predictions": predictions_list[:100],
                "total_prediction_points": len(predictions_list),
                "anomalies": anomalies_list,
                "forecast_24h": forecast_24h,
                "disclaimer": "An anomaly indicates an unusually high consumption reading relative to predicted demand, not necessarily proof of electricity wastage or equipment failure."
            }

    def train_and_compare_both(
        self,
        epochs: int = 12,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Trains both LSTM and GRU sequentially using the exact same split and seeds.
        Compares side-by-side on Val/Test MAE, RMSE, training time, and parameters.
        Selects best model strictly based on lower Validation MAE.
        """
        with self.state_lock:
            if not self.is_preprocessed or self.splits is None:
                if self.is_dataset_loaded and self.raw_file_path:
                    self.preprocess_dataset()
                else:
                    raise ValueError("No dataset loaded or preprocessed.")

            splits = self.splits
            scaler = self.scaler

            # 1. Train LSTM
            lstm_model, lstm_info = train_single_model(
                model_type="lstm",
                X_train=splits["X_train"],
                y_train=splits["y_train"],
                X_val=splits["X_val"],
                y_val=splits["y_val"],
                epochs=epochs,
                batch_size=batch_size,
                patience=3
            )
            lstm_eval = evaluate_model_on_splits(
                model=lstm_model,
                model_name="LSTM",
                scaler=scaler,
                splits=splits,
                training_info=lstm_info
            )

            # 2. Train GRU
            gru_model, gru_info = train_single_model(
                model_type="gru",
                X_train=splits["X_train"],
                y_train=splits["y_train"],
                X_val=splits["X_val"],
                y_val=splits["y_val"],
                epochs=epochs,
                batch_size=batch_size,
                patience=3
            )
            gru_eval = evaluate_model_on_splits(
                model=gru_model,
                model_name="GRU",
                scaler=scaler,
                splits=splits,
                training_info=gru_info
            )

            # 3. Model Selection based strictly on Validation MAE
            if lstm_eval["val_mae"] <= gru_eval["val_mae"]:
                best_name = "LSTM"
                best_reason = f"LSTM achieved lower Validation MAE ({lstm_eval['val_mae']:.4f} kW vs {gru_eval['val_mae']:.4f} kW)."
                winning_eval = lstm_eval
            else:
                best_name = "GRU"
                best_reason = f"GRU achieved lower Validation MAE ({gru_eval['val_mae']:.4f} kW vs {lstm_eval['val_mae']:.4f} kW)."
                winning_eval = gru_eval

            # 4. Anomaly Threshold from best model's validation positive residuals
            thresholds = compute_validation_thresholds(winning_eval["val_residuals"])
            thresh_95 = thresholds.get(95.0, 0.5)

            # 5. Anomalies on Test Set
            test_timestamps = [str(ts) for ts in splits["ts_test"]]
            test_actuals = winning_eval["test_actuals"]
            test_preds_best = winning_eval["test_predictions"]
            lstm_preds = lstm_eval["test_predictions"]
            gru_preds = gru_eval["test_predictions"]

            anomalies_list, _ = detect_anomalies_from_residuals(
                timestamps=test_timestamps,
                actuals=test_actuals,
                predictions=test_preds_best,
                validation_thresholds=thresholds,
                percentile=95.0
            )

            # 6. Combined Prediction Points
            predictions_list = []
            for ts, act, l_pred, g_pred in zip(test_timestamps, test_actuals, lstm_preds, gru_preds):
                b_pred = l_pred if best_name == "LSTM" else g_pred
                res = act - b_pred
                is_anom = res > thresh_95
                predictions_list.append({
                    "timestamp": ts,
                    "actual": round(float(act), 3),
                    "predicted": round(float(b_pred), 3),
                    "lstm_predicted": round(float(l_pred), 3),
                    "gru_predicted": round(float(g_pred), 3),
                    "residual": round(float(res), 3),
                    "is_anomaly": bool(is_anom)
                })

            # Comparison Table
            comparison_table = [
                {
                    "model": "LSTM",
                    "val_mae": lstm_eval["val_mae"],
                    "val_rmse": lstm_eval["val_rmse"],
                    "test_mae": lstm_eval["test_mae"],
                    "test_rmse": lstm_eval["test_rmse"],
                    "test_r2": lstm_eval.get("test_r2", 0.0),
                    "training_time_seconds": lstm_eval["training_duration_seconds"],
                    "epochs_trained": lstm_eval["epochs_trained"],
                    "parameters": lstm_eval["architecture"]["total_parameters"],
                    "is_best": best_name == "LSTM"
                },
                {
                    "model": "GRU",
                    "val_mae": gru_eval["val_mae"],
                    "val_rmse": gru_eval["val_rmse"],
                    "test_mae": gru_eval["test_mae"],
                    "test_rmse": gru_eval["test_rmse"],
                    "test_r2": gru_eval.get("test_r2", 0.0),
                    "training_time_seconds": gru_eval["training_duration_seconds"],
                    "epochs_trained": gru_eval["epochs_trained"],
                    "parameters": gru_eval["architecture"]["total_parameters"],
                    "is_best": best_name == "GRU"
                }
            ]

            # Save state
            self.keras_models["LSTM"] = lstm_model
            self.keras_models["GRU"] = gru_model
            self.metrics["LSTM"] = lstm_eval
            self.metrics["GRU"] = gru_eval
            self.current_model_name = "COMPARE"
            self.comparison_table = comparison_table
            self.best_model_name = best_name
            self.best_model_reason = best_reason
            self.predictions = predictions_list
            self.anomalies = anomalies_list
            self.anomaly_threshold_kw = thresh_95
            self.validation_thresholds = thresholds
            self.is_trained = True

            # Generate 24-hour forward forecast using the winning best model
            winning_model = lstm_model if best_name == "LSTM" else gru_model
            winning_rmse = winning_eval["test_rmse"]
            forecast_24h = self._compute_24h_forecast(
                model=winning_model,
                model_name=best_name,
                test_rmse=winning_rmse,
                is_best=True,
                best_reason=best_reason
            )

            return {
                "comparison_table": comparison_table,
                "best_model": best_name,
                "best_model_reason": best_reason,
                "models": {
                    "LSTM": {
                        "test_mae": lstm_eval["test_mae"],
                        "test_rmse": lstm_eval["test_rmse"],
                        "val_mae": lstm_eval["val_mae"],
                        "val_rmse": lstm_eval["val_rmse"],
                        "training_time_seconds": lstm_eval["training_duration_seconds"]
                    },
                    "GRU": {
                        "test_mae": gru_eval["test_mae"],
                        "test_rmse": gru_eval["test_rmse"],
                        "val_mae": gru_eval["val_mae"],
                        "val_rmse": gru_eval["val_rmse"],
                        "training_time_seconds": gru_eval["training_duration_seconds"]
                    }
                },
                "anomaly_threshold_kw": round(thresh_95, 4),
                "total_anomalies": len(anomalies_list),
                "sample_predictions": predictions_list[:100],
                "total_prediction_points": len(predictions_list),
                "anomalies": anomalies_list,
                "forecast_24h": forecast_24h,
                "disclaimer": "An anomaly indicates an unusually high consumption reading relative to predicted demand, not necessarily proof of electricity wastage or equipment failure."
            }

    def get_predictions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns predictions on test set."""
        if not self.is_trained or not self.predictions:
            return []
        if limit:
            return self.predictions[:limit]
        return self.predictions

    def get_anomalies(self) -> Dict[str, Any]:
        """Returns detected anomalies on test set with summary and disclaimer."""
        return {
            "total_anomalies": len(self.anomalies),
            "anomaly_threshold_kw": round(self.anomaly_threshold_kw, 4),
            "anomalies": self.anomalies,
            "disclaimer": "An anomaly indicates an unusually high reading, not proof of electricity wastage or equipment failure."
        }

    def get_comparison(self) -> Dict[str, Any]:
        """Returns model comparison results if both models were evaluated."""
        return {
            "comparison_table": self.comparison_table,
            "best_model": self.best_model_name,
            "best_model_reason": self.best_model_reason
        }

    # CSV Generators
    def generate_predictions_csv(self) -> str:
        if not self.predictions:
            return "timestamp,actual_kw,predicted_kw,residual_kw,is_anomaly\n"
        df = pd.DataFrame(self.predictions)
        return df.to_csv(index=False)

    def generate_anomalies_csv(self) -> str:
        if not self.anomalies:
            return "id,timestamp,actual,predicted,residual,threshold,severity\n"
        df = pd.DataFrame(self.anomalies)
        return df.to_csv(index=False)

    def generate_comparison_csv(self) -> str:
        if not self.comparison_table:
            return "model,val_mae,val_rmse,test_mae,test_rmse,training_time_seconds,parameters,is_best\n"
        df = pd.DataFrame(self.comparison_table)
        return df.to_csv(index=False)

    def _compute_24h_forecast(
        self,
        model,
        model_name: str,
        test_rmse: float,
        is_best: bool = False,
        best_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rolls forward 24 hourly recursive predictions using the last 24 hours of available data.
        Applies a short-term trend correction from the past 12 hours and computes
        95% expanding confidence bounds.
        """
        if self.hourly_df is None or len(self.hourly_df) < SEQUENCE_LENGTH:
            return {}

        # Last 24 hours: conditioning window for the model
        last_24_df = self.hourly_df.iloc[-SEQUENCE_LENGTH:]
        last_timestamp = pd.to_datetime(last_24_df['timestamp'].iloc[-1])
        last_reading_kw = float(last_24_df['Global_active_power'].iloc[-1])

        # Past 12 hours: historical context shown in the chart
        history_12_df = self.hourly_df.iloc[-12:]
        history_context = [
            {
                "timestamp": str(row['timestamp']),
                "actual_kw": round(float(row['Global_active_power']), 3),
                "is_historical": True
            }
            for _, row in history_12_df.iterrows()
        ]

        # Compute recent short-term trend (average hourly delta over last 12 hours)
        recent_vals = history_12_df['Global_active_power'].values.astype(float)
        recent_changes = [recent_vals[i] - recent_vals[i - 1] for i in range(1, len(recent_vals))]
        avg_trend = float(np.mean(recent_changes)) if recent_changes else 0.0
        # Dampen trend so it doesn't explode over 24 hours
        avg_trend = np.clip(avg_trend, -0.05, 0.05)

        # Build scaled initial window for the model
        raw_window = last_24_df['Global_active_power'].values.reshape(-1, 1)
        scaled_window = self.scaler.transform(raw_window)
        current_window = scaled_window.reshape(1, SEQUENCE_LENGTH, 1).copy()

        forecast_points = []
        cumulative_kwh = 0.0

        for h in range(1, 25):
            # 1-step ahead prediction from the model
            pred_scaled = float(model.predict(current_window, verbose=0)[0, 0])
            pred_kw = float(self.scaler.inverse_transform([[pred_scaled]])[0, 0])
            pred_kw = max(0.0, pred_kw)

            # Apply dampened trend correction
            forecast_kw = max(0.0, pred_kw + avg_trend * h)

            # Expanding 95% confidence interval (uncertainty grows with horizon)
            margin = 1.96 * test_rmse * (1.0 + 0.05 * np.sqrt(h))
            upper_bound = round(forecast_kw + margin, 3)
            lower_bound = round(max(0.0, forecast_kw - margin), 3)

            proj_time = last_timestamp + pd.Timedelta(hours=h)
            timestamp_str = proj_time.strftime("%Y-%m-%d %H:%M:%S")

            forecast_points.append({
                "hour_ahead": h,
                "timestamp": timestamp_str,
                "forecast_kw": round(forecast_kw, 3),
                "lower_bound_kw": lower_bound,
                "upper_bound_kw": upper_bound,
                "confidence_margin_kw": round(margin, 3)
            })

            cumulative_kwh += forecast_kw

            # Slide window forward using the raw model prediction (not trend-adjusted)
            new_step = np.array([[[pred_scaled]]])
            current_window = np.concatenate([current_window[:, 1:, :], new_step], axis=1)

        kw_values = [p["forecast_kw"] for p in forecast_points]
        max_idx = int(np.argmax(kw_values))
        min_idx = int(np.argmin(kw_values))

        selection_rationale = (
            best_reason if (is_best and best_reason)
            else f"Trained {model_name} recurrent neural network"
        )

        forecast_result = {
            "model_used": model_name,
            "is_best_model": is_best,
            "selection_rationale": selection_rationale,
            "base_timestamp": last_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "last_known_reading_kw": round(last_reading_kw, 3),
            "total_projected_kwh": round(cumulative_kwh, 2),
            "average_forecast_kw": round(float(np.mean(kw_values)), 3),
            "peak_hour": forecast_points[max_idx]["timestamp"],
            "peak_forecast_kw": forecast_points[max_idx]["forecast_kw"],
            "min_hour": forecast_points[min_idx]["timestamp"],
            "min_forecast_kw": forecast_points[min_idx]["forecast_kw"],
            "forecast_points": forecast_points,
            "history_context": history_context
        }

        self.forecast_24h = forecast_result
        return forecast_result

    def predict_next_24_hours(self, model_choice: Optional[str] = None) -> Dict[str, Any]:
        """
        Public method to predict next 24 hours of electricity consumption using the best model
        or selected model on demand.
        """
        with self.state_lock:
            if not self.is_trained or not self.keras_models:
                raise ValueError("No model is trained yet. Please train LSTM or GRU first.")

            # Model selection: best model preferred
            if model_choice and model_choice.upper() in self.keras_models:
                selected_name = model_choice.upper()
            elif self.best_model_name and self.best_model_name in self.keras_models:
                selected_name = self.best_model_name
            elif "LSTM" in self.keras_models:
                selected_name = "LSTM"
            elif "GRU" in self.keras_models:
                selected_name = "GRU"
            else:
                selected_name = list(self.keras_models.keys())[0]

            model = self.keras_models[selected_name]
            test_rmse = self.metrics.get(selected_name, {}).get("test_rmse", 0.35)
            is_best = (selected_name == self.best_model_name)
            best_reason = self.best_model_reason if is_best else None

            return self._compute_24h_forecast(
                model=model,
                model_name=selected_name,
                test_rmse=test_rmse,
                is_best=is_best,
                best_reason=best_reason
            )

    def generate_forecast_csv(self) -> str:
        if not self.forecast_24h or not self.forecast_24h.get("forecast_points"):
            return "Hour_Ahead,Timestamp,Forecast_kW,Lower_Bound_kW,Upper_Bound_kW,Confidence_Margin_kW\n"
        df = pd.DataFrame(self.forecast_24h["forecast_points"])
        return df.to_csv(index=False)


state_manager = StateManager()
