import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import tensorflow as tf

from app.core.config import SEQUENCE_LENGTH

def recursive_multi_step_forecast(
    model: tf.keras.Model,
    scaler,
    last_sequence_raw: np.ndarray,
    last_timestamp: pd.Timestamp,
    horizon_hours: int = 24,
    test_rmse: float = 0.35,
    model_name: str = "LSTM"
) -> List[Dict[str, Any]]:
    """
    Executes true recursive multi-step forecasting.
    Never duplicates values: recursively feeds each step's prediction back into the window.
    Expands confidence intervals over horizon steps reflecting forecast uncertainty accumulation.
    """
    if len(last_sequence_raw) < SEQUENCE_LENGTH:
        raise ValueError(f"Need at least {SEQUENCE_LENGTH} historical steps for forecasting.")

    # Scale the seed window
    window_raw = last_sequence_raw[-SEQUENCE_LENGTH:].reshape(-1, 1)
    window_scaled = scaler.transform(window_raw)  # shape (24, 1)
    current_input = window_scaled.reshape(1, SEQUENCE_LENGTH, 1)

    forecast_points = []
    current_dt = pd.to_datetime(last_timestamp)

    for step in range(1, horizon_hours + 1):
        # Predict next step
        next_pred_scaled = model.predict(current_input, verbose=0)[0, 0]
        
        # Guard against minor negative scaling artifacts
        next_pred_scaled = max(0.0, float(next_pred_scaled))
        
        # Invert scale to kW
        pred_kw = float(scaler.inverse_transform([[next_pred_scaled]])[0, 0])
        pred_kw = max(0.02, round(pred_kw, 3))

        # Advance timestamp
        current_dt += timedelta(hours=1)

        # Confidence bounds expand with sqrt of horizon step
        uncertainty_factor = math.sqrt(step) * 0.4
        margin = max(0.08, float(test_rmse * 1.96 * (1.0 + uncertainty_factor * 0.15)))
        lower_bound = max(0.0, round(pred_kw - margin, 3))
        upper_bound = round(pred_kw + margin, 3)

        forecast_points.append({
            "step_ahead": step,
            "timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted": pred_kw,
            "model": model_name,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        })

        # Slide window forward: drop oldest, append prediction
        new_step_scaled = np.array([[[next_pred_scaled]]])
        current_input = np.concatenate([current_input[:, 1:, :], new_step_scaled], axis=1)

    return forecast_points
