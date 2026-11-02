import os
import time
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Configure TensorFlow warnings and reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from app.core.config import (
    SEQUENCE_LENGTH, RANDOM_SEED, MODELS_DIR, ARTIFACTS_DIR,
    DEFAULT_BATCH_SIZE, DEFAULT_MAX_EPOCHS
)

def set_all_seeds(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_lstm_model(input_shape=(SEQUENCE_LENGTH, 1)) -> tf.keras.Model:
    """
    Builds LSTM model per prompt specification:
    Input(24,1) -> LSTM(32) -> Dense(16, ReLU) -> Dense(1)
    """
    set_all_seeds()
    model = models.Sequential([
        layers.Input(shape=input_shape, name="lstm_input"),
        layers.LSTM(32, name="lstm_layer"),
        layers.Dense(16, activation="relu", name="dense_relu"),
        layers.Dense(1, name="output_prediction")
    ], name="PowerPulse_LSTM")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model

def build_gru_model(input_shape=(SEQUENCE_LENGTH, 1)) -> tf.keras.Model:
    """
    Builds GRU model per prompt specification:
    Input(24,1) -> GRU(32) -> Dense(16, ReLU) -> Dense(1)
    """
    set_all_seeds()
    model = models.Sequential([
        layers.Input(shape=input_shape, name="gru_input"),
        layers.GRU(32, name="gru_layer"),
        layers.Dense(16, activation="relu", name="dense_relu"),
        layers.Dense(1, name="output_prediction")
    ], name="PowerPulse_GRU")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model

class ProgressCallback(callbacks.Callback):
    def __init__(self, model_name: str, total_epochs: int, on_epoch_end_callback: Optional[Callable] = None):
        super().__init__()
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.on_epoch_end_callback = on_epoch_end_callback
        self.logs_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        entry = {
            "epoch": epoch + 1,
            "train_loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "train_mae": float(logs.get("mae", 0.0)),
            "val_mae": float(logs.get("val_mae", 0.0))
        }
        self.logs_history.append(entry)
        if self.on_epoch_end_callback:
            self.on_epoch_end_callback(self.model_name, epoch + 1, self.total_epochs, entry)

def train_single_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = 4,
    progress_callback: Optional[Callable] = None
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Trains either LSTM or GRU with EarlyStopping, restoring best weights.
    Returns trained model, training history logs, and duration.
    """
    set_all_seeds()
    
    if model_type.lower() == "lstm":
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    elif model_type.lower() == "gru":
        model = build_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    prog_cb = ProgressCallback(model_type.upper(), epochs, progress_callback)
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[prog_cb, early_stop],
        verbose=0,
        shuffle=False  # Preserve chronological order in batches
    )
    training_duration = time.time() - start_time

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, f"{model_type.lower()}_model.keras")
    model.save(model_save_path)

    training_info = {
        "model_name": model_type.upper(),
        "epochs_trained": len(prog_cb.logs_history),
        "total_requested_epochs": epochs,
        "training_duration_seconds": round(training_duration, 2),
        "loss_history": prog_cb.logs_history,
        "save_path": model_save_path,
        "parameters_count": int(model.count_params())
    }

    return model, training_info

def evaluate_model_on_splits(
    model: tf.keras.Model,
    model_name: str,
    scaler,
    splits: Dict[str, np.ndarray],
    training_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates model in original kW units on test and validation sets.
    Computes:
      - Validation MAE & RMSE (in original kW units)
      - Test MAE & RMSE (in original kW units)
      - R² Score on Test set
      - Inference latency (ms per sample)
    """
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # Measure inference latency on test set
    t0 = time.time()
    val_preds_scaled = model.predict(X_val, verbose=0)
    test_preds_scaled = model.predict(X_test, verbose=0)
    inference_time = time.time() - t0
    latency_ms = (inference_time / (len(X_val) + len(X_test))) * 1000.0

    # Invert scaling back to original kW units
    val_actual_kw = scaler.inverse_transform(y_val)
    val_pred_kw = scaler.inverse_transform(val_preds_scaled)

    test_actual_kw = scaler.inverse_transform(y_test)
    test_pred_kw = scaler.inverse_transform(test_preds_scaled)

    # Compute genuine metrics
    val_mae = float(mean_absolute_error(val_actual_kw, val_pred_kw))
    val_rmse = float(root_mean_squared_error(val_actual_kw, val_pred_kw))

    test_mae = float(mean_absolute_error(test_actual_kw, test_pred_kw))
    test_rmse = float(root_mean_squared_error(test_actual_kw, test_pred_kw))
    test_r2 = float(r2_score(test_actual_kw, test_pred_kw))

    # Architecture summary
    layer_list = []
    for l in model.layers:
        shape_str = "dynamic"
        try:
            shape_str = str(getattr(l, "output_shape", None))
        except Exception:
            shape_str = "dynamic"
        layer_list.append({"name": l.name, "type": l.__class__.__name__, "output_shape": shape_str})

    arch_summary = {
        "name": model.name,
        "layers": layer_list,
        "total_parameters": int(model.count_params()),
        "optimizer": "Adam (lr=0.001)",
        "loss": "MSE",
        "input_shape": [SEQUENCE_LENGTH, 1]
    }

    metrics = {
        "model_name": model_name.upper(),
        "val_mae": round(val_mae, 4),
        "val_rmse": round(val_rmse, 4),
        "test_mae": round(test_mae, 4),
        "test_rmse": round(test_rmse, 4),
        "test_r2": round(test_r2, 4),
        "training_duration_seconds": training_info.get("training_duration_seconds", 0.0),
        "inference_latency_ms": round(latency_ms, 3),
        "epochs_trained": training_info.get("epochs_trained", 0),
        "loss_history": training_info.get("loss_history", []),
        "architecture": arch_summary,
        # Residuals in validation set used for anomaly detection
        "val_residuals": (val_actual_kw - val_pred_kw).flatten().tolist(),
        "test_predictions": test_pred_kw.flatten().tolist(),
        "test_actuals": test_actual_kw.flatten().tolist()
    }

    return metrics
