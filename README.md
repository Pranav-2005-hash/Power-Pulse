# PowerPulse: Electricity Consumption Prediction & Anomaly Detection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript)](https://www.typescriptlang.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?logo=tailwind-css)](https://tailwindcss.com)

PowerPulse is an academic deep-learning project and single-page web application designed to predict household electrical power consumption and detect unusual demand anomalies using deep recurrent neural networks (**LSTM** and **GRU**).

The application focuses on demonstrating the genuine deep-learning and statistical pipeline without commercial SaaS clutter.

---

## 1. Features & Academic Workflow

1. **Dataset Upload & Validation**:
   - Drag-and-drop or file browser for CSV datasets.
   - Quick "⚡ Load UCI Household Power Demo" button for instant demonstration.
   - 10-row dataset preview table with column counts, row counts, and data hygiene indicators.
   - Supports the benchmark UCI Individual Household Electric Power Consumption dataset and user-supplied CSV files.

2. **Data Processing Pipeline**:
   - Automatic detection or user selection of the target electricity consumption column.
   - Date/Time parsing and chronological sorting.
   - Conservative short-gap interpolation ($\le 2$ consecutive hours) without fabricating artificial load.
   - Resampling minute-level readings to hourly arithmetic mean ($kW$).
   - **Data Leakage Prevention**: Strict chronological 70% Train, 15% Validation, and 15% Test split. The `MinMaxScaler` is fitted solely on the training partition.
   - Sliding 24-hour sequence generation: $(N, 24, 1) \rightarrow (N, 1)$.

3. **Model Training**:
   - **Train LSTM**: 32 units, Dense(16, ReLU), Dense(1, Linear).
   - **Train GRU**: 32 units, Dense(16, ReLU), Dense(1, Linear).
   - **Train & Compare Both**: Trains both models sequentially on identical splits to evaluate performance.
   - Genuine TensorFlow/Keras implementation with Adam ($lr=0.001$), MSE loss, and EarlyStopping on validation loss.

4. **Results & Evaluation**:
   - **Prediction Results**: Actual vs. Predicted electricity consumption line chart (Recharts) and sample predictions table.
   - **Model Performance**: Test MAE, Test RMSE, Validation MAE, and Training Duration in real kilowatt ($kW$) units.
   - **Model Comparison Table**: Side-by-side LSTM vs. GRU comparison table (Val MAE, Val RMSE, Test MAE, Test RMSE, Duration, Parameter Count) with scientific model selection rationale based on Validation MAE.
   - **Statistical Anomaly Detection**:
     - Calculates residual $= \text{actual} - \text{predicted}$.
     - Derives anomaly threshold from the **95th percentile of positive validation residuals** (no test tuning).
     - Identifies and highlights test set spikes exceeding the threshold.
     - Severity classification: Critical ($\ge 99\%$), High ($\ge 95\%$), Moderate.
     - Academic disclaimer: *An anomaly indicates an unusually high consumption reading relative to historical patterns, not necessarily proof of equipment failure or energy waste.*

5. **Next 24-Hour Rollout Forecasting (Best Model)**:
   - **Autonomous Model Selection**: Automatically selects the best-performing architecture (LSTM or GRU) based on lowest Validation MAE, or evaluates the single model trained.
   - **Recursive Auto-Regressive Rollout**: Ingests the final 24 historical hourly readings $(X_{t-23}, \dots, X_t)$ and iteratively feeds each prediction $\hat{y}_{t+k}$ into the input sequence to generate the next 24 consecutive future hourly points ($t+1$ to $t+24$).
   - **Dynamic 95% Confidence Bounds**: Multi-step uncertainty propagation envelope calculated via $\text{margin}_k = 1.96 \times \text{RMSE}_{\text{test}} \times (1 + 0.05\sqrt{k})$.
   - **Key Consumption Telemetry**: Total 24-hour Projected Energy ($kWh$), Projected Peak Demand ($kW$ + timestamp), Base Load ($kW$ + timestamp), and Average Projected Demand ($kW$).
   - **Dual Horizon Chart**: Interactive Recharts timeline rendering the last 12 observed hours of historical context alongside the next 24 hours of predictions with shaded confidence bands.

6. **Download Results & Forecasts**:
   - One-click CSV export for **Historical Predictions**, **Detected Anomalies**, **Model Comparison**, and **Next 24-Hour Forecasts**.

---

## 2. System Architecture

```text
[Raw CSV Upload / UCI Sample]
       │
       ▼
[Preprocessing & Cleaning]
       │ Handle '?' / NaNs, parse timestamps, duplicate removal
       ▼
[Hourly Mean Resampling]
       │ 1-hour arithmetic mean kW, conservative short-gap check (<=2h)
       ▼
[Chronological Split (70% Train / 15% Val / 15% Test)]
       ├── Train Split (70%)  ──> [Fit MinMaxScaler] ──> Scaler Parameters (min, max)
       ├── Val Split (15%)    ──> [Transform Scaler] ──> Validation Sequences (N, 24, 1)
       └── Test Split (15%)   ──> [Transform Scaler] ──> Test Sequences (N, 24, 1)
       │
       ▼
[Model Training: LSTM & GRU]
       │ Sequential Keras models, Adam optimizer, EarlyStopping (patience=3)
       ▼
[Evaluation in Real kW Units]
       │ Inverse scaling: Test MAE, Test RMSE, Val MAE, Training Duration
       ▼
[Validation Residual Threshold (95th %)]
       │ Positive residuals: Residual = Actual - Predicted
       ▼
[Test Set Anomaly Detection & Comparison Table]
       │
       ▼
[Next 24-Hour Recursive Rollout Forecasting (Best Model: LSTM/GRU)]
       │ Multi-step rolling forward inference + 95% Confidence Bounds
       ▼
[Single-Page Responsive UI: Recharts Graphs, Tables & CSV Downloads]
```

---

## 3. Installation & Run Commands

### Backend Setup (PowerShell / Windows / Linux)

```powershell
# 1. Navigate to backend directory
cd backend

# 2. (Optional) Activate your virtual environment
# python -m venv .venv
# .venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start FastAPI server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- API root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Frontend Setup (Second Terminal)

```powershell
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
```
- Web Application: [http://localhost:5173](http://localhost:5173)

---

## 4. Automated Tests

Run the full pytest suite (18/18 tests):

```powershell
$env:PYTHONPATH="backend"
pytest backend/tests -v
```

### Verified Test Cases:
- `test_timestamp_parsing_and_cleaning`: Date/Time parser, '?' missing value handling, duplicate removal.
- `test_invalid_csv_rejection`: Unparseable / non-numeric CSV rejection.
- `test_hourly_resampling_and_gap_handling`: 1-hour mean aggregation and short-gap interpolation.
- `test_chronological_split_no_scaler_leakage`: Confirms scaler min/max are derived exclusively from the training partition.
- `test_sequence_generator_shapes`: Input shape `(samples, 24, 1)` and target shape `(samples, 1)`.
- `test_lstm_architecture_and_output_shape`: LSTM layer verification and forward pass.
- `test_gru_architecture_and_output_shape`: GRU layer verification and forward pass.
- `test_small_end_to_end_training`: End-to-end 1-epoch training and metric evaluation.
- `test_validation_threshold_calculation`: 90th, 95th, 99th percentile positive residual logic.
- `test_anomaly_detection_and_severity`: Severity grading into Critical, High, and Moderate.
- `test_academic_anomaly_disclaimer`: Verifies standard academic disclaimer text.
- `test_health_endpoint`: System status check.
- `test_upload_sample_and_preview`: Sample dataset ingestion and preview generation.
- `test_preprocess_endpoint`: Chronological split and preprocessing execution.
- `test_train_lstm_endpoint`: Real LSTM training and metric computation.
- `test_train_compare_endpoint`: Sequential training of both LSTM and GRU, comparison matrix generation.
- `test_results_endpoints_and_downloads`: Predictions, anomalies, and CSV streaming downloads.
- `test_24h_forecast_endpoint_and_download`: 24-step recursive rollout generation, telemetry, and CSV download.

---

## 5. Backend REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health status |
| `POST` | `/upload` | Upload and validate electricity consumption CSV |
| `POST` | `/upload/sample` | Load bundled UCI 90-day demo dataset |
| `GET` | `/dataset/preview` | Preview first 10 rows and dataset summary |
| `POST` | `/preprocess` | Clean, resample to hourly mean, and split chronologically |
| `POST` | `/train/lstm` | Train LSTM model and return metrics + predictions + anomalies + 24h forecast |
| `POST` | `/train/gru` | Train GRU model and return metrics + predictions + anomalies + 24h forecast |
| `POST` | `/train/compare` | Train both LSTM and GRU, compare models and generate 24h forecast with the best model |
| `GET` | `/results/predictions` | Return actual vs. predicted readings on held-out test data |
| `GET` | `/results/anomalies` | Return detected statistical anomalies on test data |
| `GET` | `/results/forecast/24h` | Return next 24-hour recursive rollout forecast with 95% confidence intervals |
| `POST` | `/results/forecast/24h` | Trigger 24-hour forecast with a specific model (`lstm` or `gru`) |
| `GET` | `/download/predictions` | Download predictions table as CSV |
| `GET` | `/download/anomalies` | Download detected anomalies as CSV |
| `GET` | `/download/comparison` | Download LSTM vs. GRU comparison table as CSV |
| `GET` | `/download/forecast` | Download 24-hour forecast table as CSV |

*(All endpoints are also available with the `/api` prefix, e.g. `/api/upload`)*

---

## 6. Dataset Citation
```bibtex
@misc{hebrail_berard_2012,
  author       = {Hébrail, Georges and Bérard, Alice},
  title        = {{Individual Household Electric Power Consumption Data Set}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository},
  note         = {{EDF R\&D}}
}
```
