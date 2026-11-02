import axios from 'axios';
import {
  DatasetStatus, DatasetPreview, AnalyticsSummary,
  ConsumptionPoint, PatternsResponse, TrainStatus,
  ModelComparison, ForecastResponse, AnomaliesResponse
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 45000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health
  checkHealth: async () => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  // Dataset
  getDatasetStatus: async (): Promise<DatasetStatus> => {
    const res = await apiClient.get<DatasetStatus>('/dataset/status');
    return res.data;
  },

  loadSampleDataset: async (): Promise<DatasetStatus> => {
    const res = await apiClient.post<DatasetStatus>('/dataset/load-sample');
    return res.data;
  },

  uploadDataset: async (file: File, timestampCol?: string, consumptionCol?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (timestampCol) formData.append('timestamp_column', timestampCol);
    if (consumptionCol) formData.append('consumption_column', consumptionCol);

    const res = await apiClient.post('/dataset/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  getDatasetPreview: async (page = 1, pageSize = 50): Promise<DatasetPreview> => {
    const res = await apiClient.get<DatasetPreview>('/dataset/preview', {
      params: { page, page_size: pageSize },
    });
    return res.data;
  },

  // Analytics
  getAnalyticsSummary: async (): Promise<AnalyticsSummary> => {
    const res = await apiClient.get<AnalyticsSummary>('/analytics/summary');
    return res.data;
  },

  getConsumptionSeries: async (startDate?: string, endDate?: string, limit = 1000): Promise<ConsumptionPoint[]> => {
    const res = await apiClient.get<ConsumptionPoint[]>('/analytics/consumption', {
      params: { start_date: startDate, end_date: endDate, limit },
    });
    return res.data;
  },

  getConsumptionPatterns: async (): Promise<PatternsResponse> => {
    const res = await apiClient.get<PatternsResponse>('/analytics/patterns');
    return res.data;
  },

  // Models
  triggerTraining: async (epochs = 15, batchSize = 64): Promise<TrainStatus> => {
    const res = await apiClient.post<TrainStatus>('/models/train', {
      epochs,
      batch_size: batchSize,
    });
    return res.data;
  },

  getTrainingStatus: async (): Promise<TrainStatus> => {
    const res = await apiClient.get<TrainStatus>('/models/status');
    return res.data;
  },

  getModelMetrics: async () => {
    const res = await apiClient.get('/models/metrics');
    return res.data;
  },

  getModelComparison: async (): Promise<ModelComparison> => {
    const res = await apiClient.get<ModelComparison>('/models/compare');
    return res.data;
  },

  // Forecasting
  executeForecast: async (model: string, horizon: number): Promise<ForecastResponse> => {
    const res = await apiClient.post<ForecastResponse>('/forecast', {
      model,
      horizon,
    });
    return res.data;
  },

  // Anomalies
  getAnomalies: async (thresholdPercentile = 95.0, status = 'all'): Promise<AnomaliesResponse> => {
    const res = await apiClient.get<AnomaliesResponse>('/anomalies', {
      params: { threshold_percentile: thresholdPercentile, status },
    });
    return res.data;
  },

  recalculateAnomalies: async (percentile: number): Promise<AnomaliesResponse> => {
    const res = await apiClient.post<AnomaliesResponse>('/anomalies/recalculate', {
      percentile,
    });
    return res.data;
  },

  reviewAnomaly: async (anomalyId: string, reviewed: boolean) => {
    const res = await apiClient.post('/anomalies/review', {
      anomaly_id: anomalyId,
      reviewed,
    });
    return res.data;
  },

  // Reports download URLs
  getForecastCsvUrl: () => `${BASE_URL}/reports/forecast.csv`,
  getAnomaliesCsvUrl: () => `${BASE_URL}/reports/anomalies.csv`,
  getModelComparisonCsvUrl: () => `${BASE_URL}/reports/model-comparison.csv`,
  getProcessedDataCsvUrl: () => `${BASE_URL}/reports/processed-data.csv`,
  getEvaluationJsonUrl: () => `${BASE_URL}/reports/evaluation.json`,
  getHtmlReportUrl: () => `${BASE_URL}/reports/summary.html`,
};
