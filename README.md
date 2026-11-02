# PowerPulse: AI-Powered Energy Forecasting & Anomaly Detection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?logo=tailwind-css)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PowerPulse is a college deep-learning project and full-stack energy analytics web application that predicts household electrical power consumption and identifies unusual demand anomalies using deep recurrent neural networks (**LSTM** and **GRU**).

Unlike static mockups, PowerPulse provides an end-to-end data processing, neural training, multi-step recursive forecasting, statistical anomaly detection, and export reporting pipeline.

---

## 1. Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Deep Learning Pipeline](#deep-learning-pipeline)
- [Forecasting Engine](#forecasting-engine)
- [Anomaly Detection Methodology](#anomaly-detection-methodology)
- [Installation & Run Commands](#installation--run-commands)
- [Automated Tests](#automated-tests)
- [API Endpoints Reference](#api-endpoints-reference)
- [Dataset Citation](#dataset-citation)
- [Limitations & Future Work](#limitations--future-work)

---

## 2. Project Overview
- **Predictive Deep Learning**: Implements genuine Keras/TensorFlow sequential models:
  - **LSTM**: `Input(24, 1) -> LSTM(32) -> Dense(16, ReLU) -> Dense(1)`
  - **GRU**: `Input(24, 1) -> GRU(32) -> Dense(16, ReLU) -> Dense(1)`
- **Data Leakage Prevention**: Strict 70% Train, 15% Validation, 15% Test chronological split. The `MinMaxScaler` is fitted solely on the training partition.
- **Validation-Derived Anomaly Detection**: Calculates residual $= \text{actual} - \text{predicted}$. Derives 90th, 95th, and 99th percentile alert thresholds strictly from positive validation residuals—never tuned on the held-out test set.
- **Recursive Multi-Step Forecasting**: Real roll-forward inference across 1, 6, 12, 24, and 48-hour horizons with expanding $95\%$ confidence bounds.
- **Reporting & Exports**: One-click downloads for forecast CSVs, anomaly logs, comparative evaluation matrices, JSON telemetry, and executive HTML reports.

---

## 3. System Architecture

```text
[Raw Data Ingestion]
        │ (UCI Individual Household Electric Power CSV: Date, Time, Global_active_power)
        ▼
[Preprocessing & Cleaning]
        │ Parse timestamps, convert '?' to NaN, conservative short-gap check (<=2h)
        ▼
[Hourly Mean Resampling]
        │ Aggregates minute recordings to continuous 1-hour interval mean kW
        ▼
[Strict Chronological Split (70% / 15% / 15%)]
   ┌────┴────────────────────────┬─────────────────────────┐
   ▼                             ▼                         ▼
Train Set (70%)             Val Set (15%)              Test Set (15%)
[Fit MinMaxScaler]         [Transform Scaler]         [Transform Scaler]
   │                             │                         │
   ▼                             ▼                         ▼
24h Window Generator       24h Window Generator       24h Window Generator
(X_train: N, 24, 1)        (X_val: N, 24, 1)          (X_test: N, 24, 1)
   │                             │                         │
   ▼                             │                         │
[Model Training: LSTM & GRU]     │                         │
   │ EarlyStopping on Val Loss   │                         │
   ▼                             ▼                         ▼
[Model Selection] ────────> Validation MAE <────────── Evaluation
   │ (Selects Best Architecture) (In Real kW Units)    (MAE, RMSE, R²)
   ▼                                                       │
[Residual Anomaly Detector] <──────────────────────────────┘
   │ Validation Residuals Threshold (90th, 95th, 99th %)
   ▼
[Recursive Multi-Step Forecaster] (1h, 6h, 12h, 24h, 48h Horizons)
   ▼
[FastAPI REST Backend] <──── HTTP/JSON ────> [React + Vite + Tailwind UI]
```

---

## 4. Dataset & Preprocessing

The primary benchmark dataset is the **UCI Individual Household Electric Power Consumption** dataset:
- Target Variable: `Global_active_power` (household global minute-averaged active power in kilowatts).
- Cleaning pipeline:
  1. Parse standard date and time stamps (`DD/MM/YYYY HH:MM:SS` or standard ISO 8601).
  2. Map missing or corrupt entries (`'?'`, `'NA'`, empty strings) to `NaN`.
  3. Sort records in chronological sequence and eliminate duplicate timestamps.
  4. Resample minute readings to hourly arithmetic mean.
  5. Interpolate only short duration gaps ($\le 2$ consecutive hours) without inventing artificial load across extended power outages.

---

## 5. Deep Learning Pipeline

### Architecture Specification
- **LSTM Architecture**:
  - `layers.Input(shape=(24, 1))`
  - `layers.LSTM(32)`
  - `layers.Dense(16, activation='relu')`
  - `layers.Dense(1)`
- **GRU Architecture**:
  - `layers.Input(shape=(24, 1))`
  - `layers.GRU(32)`
  - `layers.Dense(16, activation='relu')`
  - `layers.Dense(1)`
- **Optimizer & Loss**:
  - Optimizer: `Adam(learning_rate=0.001)`
  - Loss: `Mean Squared Error (MSE)`
  - Metric: `Mean Absolute Error (MAE)`
  - Callbacks: `EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)`

### Model Selection Rule
Per rigorous scientific methodology, the winning model is selected based on **Validation MAE** (in original kilowatt units) rather than test set performance, preventing post-selection bias.

---

## 6. Forecasting Engine

PowerPulse implements **recursive multi-step forecasting**:
1. Takes the preceding 24-hour sequence of scaled consumption readings.
2. Iterates across horizon steps $t+1, \dots, t+H$ ($H \in \{1, 6, 12, 24, 48\}$):
   $$\hat{y}_{t+k} = f(\hat{y}_{t+k-24}, \dots, \hat{y}_{t+k-1})$$
3. Inverts the scaling transformation back to true kilowatt ($kW$) units.
4. Generates an expanding uncertainty envelope based on test RMSE:
   $$\text{Margin} = 1.96 \times \text{RMSE}_{\text{test}} \times (1 + 0.06\sqrt{k})$$

---

## 7. Anomaly Detection Methodology

PowerPulse avoids arbitrary magic numbers by calibrating statistical thresholds directly against validation residuals:
$$\text{Residual}_t = \text{Actual}_t - \text{Predicted}_t$$

Only positive residuals ($\text{Residual}_t > 0$) on the validation split are pooled to establish cutoffs:
- **90th Percentile**: High sensitivity threshold for early anomaly warnings.
- **95th Percentile**: Standard operational threshold for atypical energy spikes.
- **99th Percentile**: Critical outlier threshold for major appliance or electrical surges.

Anomalies can be reviewed, marked as acknowledged, and exported to CSV.

---

## 8. Installation & Run Commands

### Backend Setup (PowerShell / Windows / Linux)

```powershell
# 1. Navigate to backend directory
cd backend

# 2. Create virtual environment (optional if using global python)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI development server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The backend API will be available at `http://127.0.0.1:8000` with interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### Frontend Setup (In another terminal)

```powershell
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start Vite development server
npm run dev
```

The frontend application will be accessible at `http://localhost:5173`.

---

## 9. Automated Tests

Run the complete automated pytest suite:

```powershell
# From project root
$env:PYTHONPATH="backend"
pytest backend/tests -v
```

### Verified Test Cases (16/16 Passed):
- `test_timestamp_parsing_and_cleaning`: Date/Time parser, '?' handling, duplicate removal.
- `test_invalid_csv_rejection`: Corrupt CSV upload rejection with HTTP 422.
- `test_hourly_resampling_and_gap_handling`: Conservative 1-hour mean aggregation.
- `test_chronological_split_no_scaler_leakage`: Confirms scaler min/max are derived exclusively from training data.
- `test_sequence_generator_shapes`: Input shape `(samples, 24, 1)` and target shape `(samples, 1)`.
- `test_lstm_architecture_and_output_shape`: Layer verification and forward pass.
- `test_gru_architecture_and_output_shape`: Layer verification and forward pass.
- `test_small_end_to_end_training`: End-to-end 1-epoch training and kW-unit metric evaluation.
- `test_validation_threshold_calculation`: 90th, 95th, 99th percentile positive residual logic.
- `test_anomaly_detection_and_severity`: Categorization into Critical, High, and Moderate.
- `test_anomaly_review_persistence`: Persistent review toggles.
- `test_health_endpoint`: Health and version endpoint check.
- `test_dataset_status_and_preview`: Status metadata and pagination preview.
- `test_analytics_endpoints`: Descriptive stats, moving averages, and diurnal patterns.
- `test_forecast_and_reports_error_handling`: Untrained model protection and validation.
- `test_processed_data_report_download`: CSV download stream integrity.

---

## 10. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check and system version |
| `GET` | `/api/dataset/status` | Dataset rows, date range, and model compatibility |
| `POST` | `/api/dataset/load-sample` | Preloads UCI Household sample dataset |
| `POST` | `/api/dataset/upload` | Multipart upload for custom user CSVs |
| `GET` | `/api/dataset/preview` | Paginated preview of cleaned hourly records |
| `GET` | `/api/analytics/summary` | Descriptive statistics (mean, std, IQR, skewness) |
| `GET` | `/api/analytics/consumption` | Hourly time-series with 6h/24h rolling means |
| `GET` | `/api/analytics/patterns` | Hourly diurnal and day-of-week load profiles |
| `POST` | `/api/models/train` | Triggers background LSTM/GRU training |
| `GET` | `/api/models/status` | Real-time training progress polling |
| `GET` | `/api/models/metrics` | Evaluation metrics in kW units |
| `GET` | `/api/models/compare` | Side-by-side comparison & best model rationale |
| `POST` | `/api/forecast` | Recursive multi-step forecasting (1-48 hours) |
| `GET` | `/api/anomalies` | Filterable list of residual anomalies |
| `POST` | `/api/anomalies/recalculate` | Recalculate cutoffs at 90/95/99 percentiles |
| `POST` | `/api/anomalies/review` | Toggle review status of an anomaly event |
| `GET` | `/api/reports/forecast.csv` | Download forecast table as CSV |
| `GET` | `/api/reports/anomalies.csv` | Download flagged anomalies as CSV |
| `GET` | `/api/reports/model-comparison.csv` | Download comparative metrics CSV |
| `GET` | `/api/reports/processed-data.csv` | Download cleaned hourly dataset CSV |
| `GET` | `/api/reports/evaluation.json` | Download full evaluation telemetry in JSON |
| `GET` | `/api/reports/summary.html` | Consolidated executive report (printable) |

---

## 11. Dataset Citation
```bibtex
@misc{hebrail_berard_2012,
  author       = {Hébrail, Georges and Bérard, Alice},
  title        = {{Individual Household Electric Power Consumption Data Set}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository},
  note         = {{EDF R\&D}}
}
```

---

## 12. Limitations & Future Work
- **Limitations**:
  - Pure autoregressive lag inputs without ambient weather, temperature, or occupancy telemetry.
  - Multi-step recursive error compounding across longer horizons ($\ge 48$ hours).
  - Assumption of stationary consumption baseline over seasonal shifts.
- **Future Directions**:
  - Exogenous sensor fusion (temperature, humidity, solar radiation).
  - Attention-based Temporal Fusion Transformers (TFT) or Informer models.
  - TFLite quantization for deployment on smart metering edge microcontrollers.
