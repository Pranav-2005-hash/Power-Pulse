export interface DatasetStatus {
  is_loaded: boolean;
  filename: string | null;
  source_type: string;
  total_raw_rows: number;
  total_hourly_rows: number;
  start_date: string | null;
  end_date: string | null;
  missing_values_handled: number;
  sampling_rate: string;
  target_column: string;
  units: string;
  columns: string[];
  models_trained: boolean;
  last_updated: string | null;
}

export interface DatasetPreview {
  columns: string[];
  rows: Record<string, any>[];
  total_rows: number;
  page: number;
  page_size: number;
}

export interface DescriptiveStats {
  count: number;
  mean: number;
  std: number;
  min: number;
  p25: number;
  median: number;
  p75: number;
  max: number;
  skewness: number;
  iqr: number;
}

export interface AnalyticsSummary {
  stats: DescriptiveStats;
  peak_consumption: number;
  peak_timestamp: string;
  lowest_consumption: number;
  lowest_timestamp: string;
  total_energy_kwh: number;
  avg_daily_kwh: number;
  current_trend: string;
}

export interface ConsumptionPoint {
  timestamp: string;
  actual: number;
  rolling_6h?: number | null;
  rolling_24h?: number | null;
}

export interface HourlyPattern {
  hour: number;
  mean_kw: number;
  std_kw: number;
  min_kw: number;
  max_kw: number;
}

export interface DayOfWeekPattern {
  day_index: number;
  day_name: string;
  mean_kw: number;
  std_kw: number;
  peak_hour: number;
}

export interface PatternsResponse {
  hourly: HourlyPattern[];
  daily: DayOfWeekPattern[];
  peak_hour_overall: number;
  peak_day_overall: string;
}

export interface EpochLog {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_mae: number;
  val_mae: number;
}

export interface ModelMetrics {
  model_name: string;
  test_mae: number;
  test_rmse: number;
  test_r2: number;
  val_mae: number;
  val_rmse: number;
  training_duration_seconds: number;
  inference_latency_ms: number;
  epochs_trained: number;
  loss_history: EpochLog[];
  architecture: Record<string, any>;
}

export interface TrainStatus {
  status: 'idle' | 'training' | 'completed' | 'failed';
  progress_percentage: number;
  current_model: string | null;
  current_epoch: number | null;
  total_epochs: number | null;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface ModelComparison {
  is_trained: boolean;
  best_model: string | null;
  selection_reason: string;
  models: Record<string, ModelMetrics>;
  recommendation: string | null;
}

export interface ForecastPoint {
  step_ahead: number;
  timestamp: string;
  predicted: number;
  model: string;
  lower_bound: number;
  upper_bound: number;
}

export interface HistoricalPoint {
  timestamp: string;
  actual: number;
}

export interface ForecastResponse {
  horizon_hours: number;
  models_used: string[];
  forecast_points: ForecastPoint[];
  historical_context: HistoricalPoint[];
  metrics_reference: Record<string, any>;
  disclaimer: string;
}

export interface AnomalyEvent {
  id: string;
  timestamp: string;
  actual: number;
  predicted: number;
  residual: number;
  threshold: number;
  severity: 'Critical' | 'High' | 'Moderate';
  percentile: number;
  reviewed: boolean;
  review_notes?: string | null;
}

export interface AnomaliesResponse {
  total_anomalies: number;
  unreviewed_count: number;
  reviewed_count: number;
  severity_counts: Record<string, number>;
  threshold_percentile: number;
  threshold_value_kw: number;
  anomalies: AnomalyEvent[];
  series_sample: Array<{
    timestamp: string;
    actual: number;
    predicted: number;
    residual: number;
    is_anomaly: boolean;
    severity?: string | null;
  }>;
  disclaimer: string;
}
