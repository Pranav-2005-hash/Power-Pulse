import axios from 'axios';
import {
  DatasetPreview,
  PreprocessingSummary,
  TrainResponse,
  PredictionPoint,
  AnomaliesResponse,
  Forecast24hResponse
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 120000, // 2 minutes for training deep learning models
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // System Health
  checkHealth: async () => {
    const res = await apiClient.get('/health');
    return res.data;
  },

  // Dataset Upload & Preview
  uploadDataset: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  loadSampleDataset: async () => {
    const res = await apiClient.post('/upload/sample');
    return res.data;
  },

  getDatasetPreview: async (): Promise<DatasetPreview> => {
    const res = await apiClient.get<DatasetPreview>('/dataset/preview');
    return res.data;
  },

  // Preprocessing
  preprocessDataset: async (targetColumn?: string, datetimeColumn?: string): Promise<PreprocessingSummary> => {
    const res = await apiClient.post<PreprocessingSummary>('/preprocess', {
      target_column: targetColumn,
      datetime_column: datetimeColumn,
    });
    return res.data;
  },

  // Model Training
  trainLSTM: async (epochs = 12, batchSize = 64): Promise<TrainResponse> => {
    const res = await apiClient.post<TrainResponse>('/train/lstm', {
      epochs,
      batch_size: batchSize,
    });
    return res.data;
  },

  trainGRU: async (epochs = 12, batchSize = 64): Promise<TrainResponse> => {
    const res = await apiClient.post<TrainResponse>('/train/gru', {
      epochs,
      batch_size: batchSize,
    });
    return res.data;
  },

  trainCompare: async (epochs = 12, batchSize = 64): Promise<TrainResponse> => {
    const res = await apiClient.post<TrainResponse>('/train/compare', {
      epochs,
      batch_size: batchSize,
    });
    return res.data;
  },

  // Results
  getPredictions: async (limit = 500): Promise<{ predictions: PredictionPoint[]; total_test_samples: number }> => {
    const res = await apiClient.get('/results/predictions', {
      params: { limit },
    });
    return res.data;
  },

  getAnomalies: async (): Promise<AnomaliesResponse> => {
    const res = await apiClient.get<AnomaliesResponse>('/results/anomalies');
    return res.data;
  },

  get24hForecast: async (model?: string): Promise<Forecast24hResponse> => {
    const res = await apiClient.get<Forecast24hResponse>('/results/forecast/24h', {
      params: model ? { model } : {},
    });
    return res.data;
  },

  // Download URLs
  getPredictionsCsvUrl: () => `${BASE_URL}/download/predictions`,
  getAnomaliesCsvUrl: () => `${BASE_URL}/download/anomalies`,
  getComparisonCsvUrl: () => `${BASE_URL}/download/comparison`,
  getForecastCsvUrl: () => `${BASE_URL}/download/forecast`,
};
