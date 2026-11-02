import threading
import time
from datetime import datetime
from typing import Optional

from app.storage.state_manager import state_manager
from app.ml.models import train_single_model, evaluate_model_on_splits
from app.ml.anomaly import compute_validation_thresholds

_training_thread: Optional[threading.Thread] = None

def run_training_background_job(epochs: int = 15, batch_size: int = 64, patience: int = 4):
    """
    Executes training in a background daemon thread so it doesn't block FastAPI requests.
    Updates state_manager.train_status live after each epoch.
    """
    global _training_thread
    
    if not state_manager.is_dataset_loaded or state_manager.splits is None:
        raise ValueError("Cannot train models without a loaded dataset.")

    if state_manager.train_status["status"] == "training":
        raise RuntimeError("A model training process is already actively running.")

    def _train_worker():
        try:
            state_manager.train_status.update({
                "status": "training",
                "progress_percentage": 5.0,
                "current_model": "LSTM",
                "current_epoch": 0,
                "total_epochs": epochs,
                "error_message": None,
                "started_at": datetime.now().isoformat(),
                "completed_at": None
            })

            splits = state_manager.splits
            scaler = state_manager.scaler

            # Step 1: Train LSTM
            def lstm_progress(model_name, ep, total, logs):
                pct = 5.0 + (ep / total) * 45.0  # 5% to 50%
                state_manager.train_status["current_epoch"] = ep
                state_manager.train_status["progress_percentage"] = round(pct, 1)

            lstm_model, lstm_info = train_single_model(
                "lstm",
                splits["X_train"], splits["y_train"],
                splits["X_val"], splits["y_val"],
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                progress_callback=lstm_progress
            )

            # Step 2: Train GRU
            state_manager.train_status.update({
                "current_model": "GRU",
                "current_epoch": 0,
                "progress_percentage": 50.0
            })

            def gru_progress(model_name, ep, total, logs):
                pct = 50.0 + (ep / total) * 40.0  # 50% to 90%
                state_manager.train_status["current_epoch"] = ep
                state_manager.train_status["progress_percentage"] = round(pct, 1)

            gru_model, gru_info = train_single_model(
                "gru",
                splits["X_train"], splits["y_train"],
                splits["X_val"], splits["y_val"],
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                progress_callback=gru_progress
            )

            # Step 3: Evaluate both models on test set and validation set
            state_manager.train_status.update({
                "current_model": "Evaluating Models & Computing Metrics",
                "progress_percentage": 92.0
            })

            lstm_metrics = evaluate_model_on_splits(lstm_model, "LSTM", scaler, splits, lstm_info)
            gru_metrics = evaluate_model_on_splits(gru_model, "GRU", scaler, splits, gru_info)

            # Step 4: Model Selection based strictly on Validation MAE
            val_mae_lstm = lstm_metrics["val_mae"]
            val_mae_gru = gru_metrics["val_mae"]

            if val_mae_lstm <= val_mae_gru:
                best_model_name = "LSTM"
                selection_reason = (
                    f"LSTM achieved lower Validation MAE ({val_mae_lstm:.4f} kW) "
                    f"compared to GRU ({val_mae_gru:.4f} kW)."
                )
                best_model = lstm_model
                best_residuals = lstm_metrics["val_residuals"]
            else:
                best_model_name = "GRU"
                selection_reason = (
                    f"GRU achieved lower Validation MAE ({val_mae_gru:.4f} kW) "
                    f"compared to LSTM ({val_mae_lstm:.4f} kW)."
                )
                best_model = gru_model
                best_residuals = gru_metrics["val_residuals"]

            # Step 5: Derive Anomaly Thresholds strictly from validation residuals
            thresholds = compute_validation_thresholds(best_residuals)

            # Step 6: Store in StateManager
            state_manager.keras_models["LSTM"] = lstm_model
            state_manager.keras_models["GRU"] = gru_model
            
            all_metrics = {
                "LSTM": lstm_metrics,
                "GRU": gru_metrics
            }

            state_manager.save_trained_state(
                metrics=all_metrics,
                best_model_name=best_model_name,
                selection_reason=selection_reason,
                thresholds=thresholds
            )

            state_manager.train_status.update({
                "status": "completed",
                "progress_percentage": 100.0,
                "current_model": None,
                "completed_at": datetime.now().isoformat()
            })

        except Exception as e:
            print(f"Error in training worker: {e}")
            state_manager.train_status.update({
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now().isoformat()
            })

    _training_thread = threading.Thread(target=_train_worker, daemon=True)
    _training_thread.start()
    return True
