import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

from app.core.config import REPORTS_DIR, PROCESSED_DATA_PATH
from app.storage.state_manager import state_manager
from app.ml.anomaly import detect_anomalies_from_residuals
from app.ml.forecasting import recursive_multi_step_forecast

def generate_forecast_csv() -> str:
    """Generates 24h forecast CSV from the best trained model."""
    if not state_manager.models_trained or not state_manager.best_model_name:
        raise ValueError("Models must be trained before generating forecast CSV.")

    model = state_manager.keras_models.get(state_manager.best_model_name)
    scaler = state_manager.scaler
    hourly = state_manager.hourly_df

    last_raw = hourly['Global_active_power'].values
    last_ts = hourly['timestamp'].iloc[-1]
    best_metric = state_manager.metrics.get(state_manager.best_model_name, {})
    test_rmse = best_metric.get("test_rmse", 0.35)

    forecast_pts = recursive_multi_step_forecast(
        model=model,
        scaler=scaler,
        last_sequence_raw=last_raw,
        last_timestamp=last_ts,
        horizon_hours=24,
        test_rmse=test_rmse,
        model_name=state_manager.best_model_name
    )

    df_out = pd.DataFrame(forecast_pts)
    return df_out.to_csv(index=False)

def generate_anomalies_csv(percentile: float = 95.0) -> str:
    """Generates flagged anomalies CSV."""
    if not state_manager.models_trained or not state_manager.best_model_name:
        raise ValueError("Models must be trained before extracting anomalies.")

    splits = state_manager.splits
    best_m = state_manager.metrics.get(state_manager.best_model_name, {})
    
    test_actuals = best_m.get("test_actuals", [])
    test_preds = best_m.get("test_predictions", [])
    test_ts = [str(t) for t in splits["ts_test"]]

    anomalies, _ = detect_anomalies_from_residuals(
        timestamps=test_ts,
        actuals=test_actuals,
        predictions=test_preds,
        validation_thresholds=state_manager.validation_thresholds,
        percentile=percentile,
        reviewed_ids=state_manager.anomaly_reviews
    )

    if not anomalies:
        df_out = pd.DataFrame(columns=["id", "timestamp", "actual", "predicted", "residual", "threshold", "severity", "percentile", "reviewed", "review_notes"])
    else:
        df_out = pd.DataFrame(anomalies)
    return df_out.to_csv(index=False)

def generate_model_comparison_csv() -> str:
    """Generates comparison CSV comparing LSTM and GRU."""
    if not state_manager.models_trained:
        raise ValueError("Models must be trained before generating comparison CSV.")

    rows = []
    for model_name, m in state_manager.metrics.items():
        rows.append({
            "model": model_name,
            "validation_mae_kw": m.get("val_mae"),
            "validation_rmse_kw": m.get("val_rmse"),
            "test_mae_kw": m.get("test_mae"),
            "test_rmse_kw": m.get("test_rmse"),
            "test_r2_score": m.get("test_r2"),
            "training_duration_sec": m.get("training_duration_seconds"),
            "inference_latency_ms": m.get("inference_latency_ms"),
            "epochs_trained": m.get("epochs_trained"),
            "is_best_model": (model_name == state_manager.best_model_name)
        })

    df_out = pd.DataFrame(rows)
    return df_out.to_csv(index=False)

def get_processed_data_csv() -> str:
    """Returns the cleaned, resampled hourly consumption CSV."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed hourly data not found.")
    with open(PROCESSED_DATA_PATH, 'r') as f:
        return f.read()

def generate_evaluation_json() -> Dict[str, Any]:
    """Generates complete model evaluation payload in JSON format."""
    if not state_manager.models_trained:
        raise ValueError("Models must be trained before generating evaluation JSON.")

    return {
        "project": "PowerPulse: AI-Powered Energy Forecasting & Anomaly Detection",
        "generated_at": datetime.now().isoformat(),
        "dataset_info": {
            "source_type": state_manager.source_type,
            "filename": state_manager.dataset_filename,
            "sampling_interval": "1 hour (mean)",
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "split_samples": state_manager.split_info
        },
        "model_selection": {
            "selected_best_model": state_manager.best_model_name,
            "selection_criterion": "Lowest Validation MAE (in kW)",
            "selection_rationale": state_manager.selection_reason
        },
        "validation_anomaly_thresholds": state_manager.validation_thresholds,
        "metrics": {
            k: {
                metric: v for metric, v in val.items() 
                if metric not in ["val_residuals", "test_predictions", "test_actuals"]
            }
            for k, val in state_manager.metrics.items()
        }
    }

def generate_html_summary_report() -> str:
    """Generates a styled, executive-grade HTML consolidated report."""
    eval_data = generate_evaluation_json() if state_manager.models_trained else {}
    best_m = eval_data.get("model_selection", {}).get("selected_best_model", "N/A")
    rationale = eval_data.get("model_selection", {}).get("selection_rationale", "Models not yet trained.")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PowerPulse — Executive Deep Learning Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 40px; }}
    .container {{ max-width: 900px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 36px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
    .header {{ border-bottom: 2px solid #10b981; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }}
    .brand {{ font-size: 26px; font-weight: 800; color: #0b132b; }}
    .brand span {{ color: #10b981; }}
    .badge {{ background: #ecfdf5; color: #059669; font-weight: 600; padding: 6px 14px; border-radius: 20px; font-size: 13px; }}
    h2 {{ color: #0b132b; font-size: 19px; margin-top: 28px; border-left: 4px solid #06b6d4; padding-left: 10px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 10px 14px; text-align: left; font-size: 14px; }}
    th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
    .highlight {{ background: #f0fdf4; font-weight: bold; color: #15803d; }}
    .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #64748b; text-align: center; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="brand">Power<span>Pulse</span></div>
      <div class="badge">Official ML Evaluation Report</div>
    </div>
    
    <p><strong>Report Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
    <p><strong>Project:</strong> AI-Powered Energy Forecasting & Anomaly Detection (College Deep Learning Project)</p>

    <h2>1. Model Selection & Executive Summary</h2>
    <p><strong>Selected Best Model:</strong> <span class="badge">{best_m}</span></p>
    <p><strong>Selection Basis:</strong> {rationale}</p>

    <h2>2. Comparative Deep Learning Metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Model Architecture</th>
          <th>Val MAE (kW)</th>
          <th>Val RMSE (kW)</th>
          <th>Test MAE (kW)</th>
          <th>Test RMSE (kW)</th>
          <th>Test R² Score</th>
          <th>Train Time (s)</th>
          <th>Inference (ms)</th>
        </tr>
      </thead>
      <tbody>
"""
    if state_manager.models_trained:
        for name, m in state_manager.metrics.items():
            is_best = (name == best_m)
            row_cls = ' class="highlight"' if is_best else ''
            html += f"""
        <tr{row_cls}>
          <td><strong>{name}</strong> {'(Selected Best)' if is_best else ''}</td>
          <td>{m.get('val_mae')}</td>
          <td>{m.get('val_rmse')}</td>
          <td>{m.get('test_mae')}</td>
          <td>{m.get('test_rmse')}</td>
          <td>{m.get('test_r2')}</td>
          <td>{m.get('training_duration_seconds')}s</td>
          <td>{m.get('inference_latency_ms')}ms</td>
        </tr>"""
    else:
        html += """<tr><td colspan="8">No model training results available yet.</td></tr>"""

    html += f"""
      </tbody>
    </table>

    <h2>3. Anomaly Detection Calibration</h2>
    <p>Anomaly detection relies on statistical thresholds derived exclusively from positive validation residuals:</p>
    <ul>
      <li><strong>90th Percentile:</strong> {state_manager.validation_thresholds.get(90.0, 'N/A')} kW residual</li>
      <li><strong>95th Percentile (Standard):</strong> {state_manager.validation_thresholds.get(95.0, 'N/A')} kW residual</li>
      <li><strong>99th Percentile (Critical):</strong> {state_manager.validation_thresholds.get(99.0, 'N/A')} kW residual</li>
    </ul>

    <h2>4. Methodology & Data Hygiene</h2>
    <p>Data was strictly partitioned chronologically into 70% Train, 15% Validation, and 15% Test. The MinMaxScaler was fitted strictly on training data alone to eliminate data leakage. Sliding 24-hour input windows are used to forecast target variable Global_active_power (kW).</p>

    <div class="footer">
      Generated automatically by PowerPulse ML Engine &bull; Zero hardcoded demo metrics &bull; College Deep Learning Project
    </div>
  </div>
</body>
</html>
"""
    return html
