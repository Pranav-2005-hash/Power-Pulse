export interface DatasetPreview {
  is_loaded: boolean;
  filename: string | null;
  total_rows: number;
  total_columns: number;
  columns: string[];
  numeric_columns: string[];
  preview: Record<string, any>[];
  detected_target_col: string | null;
  detected_datetime_col: string | null;
  is_preprocessed: boolean;
  preprocessing_summary?: PreprocessingSummary | null;
}

export interface PreprocessingSummary {
  status: string;
  message: string;
  target_column: string;
  initial_raw_rows: number;
  valid_raw_rows: number;
  hourly_records: number;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  input_sequence_length: number;
  date_range: {
    start: string;
    end: string;
  };
  mean_kw: number;
  min_kw: number;
  max_kw: number;
  scaler_train_min: number;
  scaler_train_max: number;
  missing_values_handled: number;
  split_ratio: string;
}

export interface TrainMetrics {
  mae: number;
  rmse: number;
  val_mae: number;
  val_rmse: number;
  test_r2?: number;
  training_time_seconds: number;
  epochs_trained: number;
  parameters_count: number;
}

export interface ComparisonRow {
  model: string;
  val_mae: number;
  val_rmse: number;
  test_mae: number;
  test_rmse: number;
  test_r2?: number;
  training_time_seconds: number;
  epochs_trained: number;
  parameters: number;
  is_best: boolean;
}

export interface PredictionPoint {
  timestamp: string;
  actual: number;
  predicted: number;
  lstm_predicted?: number;
  gru_predicted?: number;
  residual: number;
  is_anomaly: boolean;
}

export interface AnomalyPoint {
  id: string;
  timestamp: string;
  actual: number;
  predicted: number;
  residual: number;
  threshold: number;
  severity: 'Critical' | 'High' | 'Moderate';
}

export interface AnomaliesResponse {
  total_anomalies: number;
  anomaly_threshold_kw: number;
  anomalies: AnomalyPoint[];
  disclaimer: string;
}

export interface ForecastPoint24h {
  hour_ahead: number;
  timestamp: string;
  forecast_kw: number;
  lower_bound_kw: number;
  upper_bound_kw: number;
  confidence_margin_kw: number;
}

export interface HistoryContextPoint {
  timestamp: string;
  actual_kw: number;
  is_historical: boolean;
}

export interface Forecast24hResponse {
  model_used: string;
  is_best_model: boolean;
  selection_rationale: string;
  base_timestamp: string;
  last_known_reading_kw: number;
  total_projected_kwh: number;
  average_forecast_kw: number;
  peak_hour: string;
  peak_forecast_kw: number;
  min_hour: string;
  min_forecast_kw: number;
  forecast_points: ForecastPoint24h[];
  history_context: HistoryContextPoint[];
}

export interface TrainResponse {
  model_name?: string;
  metrics?: TrainMetrics;
  anomaly_threshold_kw: number;
  total_anomalies: number;
  sample_predictions: PredictionPoint[];
  total_prediction_points?: number;
  anomalies: AnomalyPoint[];
  comparison_table?: ComparisonRow[];
  best_model?: string;
  best_model_reason?: string;
  forecast_24h?: Forecast24hResponse;
  disclaimer: string;
}
