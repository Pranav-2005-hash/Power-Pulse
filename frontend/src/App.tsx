import React, { useState, useEffect, useRef } from 'react';
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  Cpu,
  TrendingUp,
  AlertTriangle,
  Download,
  RefreshCw,
  Sliders,
  Database,
  Layers,
  Table as TableIcon,
  Info,
  Clock,
  Activity,
  ArrowRight,
  Zap,
  Sparkles,
  Calendar
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { api } from './services/api';
import {
  DatasetPreview,
  PreprocessingSummary,
  TrainResponse,
  PredictionPoint,
  ComparisonRow,
  AnomalyPoint,
  Forecast24hResponse,
  ForecastPoint24h
} from './types';

export const App: React.FC = () => {
  // System State
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);

  // Dataset State
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);
  const [selectedTargetCol, setSelectedTargetCol] = useState<string>('Global_active_power');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Preprocessing State
  const [preprocessingSummary, setPreprocessingSummary] = useState<PreprocessingSummary | null>(null);
  const [isPreprocessing, setIsPreprocessing] = useState<boolean>(false);
  const [preprocessError, setPreprocessError] = useState<string | null>(null);

  // Training State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingMessage, setTrainingMessage] = useState<string>('');
  const [trainError, setTrainError] = useState<string | null>(null);

  // Results State
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [activeModelName, setActiveModelName] = useState<string | null>(null);
  const [chartSlice, setChartSlice] = useState<number>(100); // points to display in chart

  // 24-Hour Forecast State
  const [forecast24h, setForecast24h] = useState<Forecast24hResponse | null>(null);
  const [isForecasting, setIsForecasting] = useState<boolean>(false);

  useEffect(() => {
    if (trainResult?.forecast_24h) {
      setForecast24h(trainResult.forecast_24h);
    }
  }, [trainResult]);

  const handleReforecast = async (modelName?: string) => {
    setIsForecasting(true);
    try {
      const data = await api.get24hForecast(modelName);
      setForecast24h(data);
    } catch (err: any) {
      console.error(err);
    } finally {
      setIsForecasting(false);
    }
  };

  useEffect(() => {
    checkHealthAndInitialDataset();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      await api.checkHealth();
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  };

  const checkHealthAndInitialDataset = async () => {
    setIsLoadingHealth(true);
    try {
      await api.checkHealth();
      setBackendOnline(true);
      const preview = await api.getDatasetPreview();
      if (preview && preview.is_loaded) {
        setDatasetPreview(preview);
        if (preview.detected_target_col) {
          setSelectedTargetCol(preview.detected_target_col);
        }
        if (preview.is_preprocessed && preview.preprocessing_summary) {
          setPreprocessingSummary(preview.preprocessing_summary);
        }
      }
    } catch (err) {
      console.error(err);
      setBackendOnline(false);
    } finally {
      setIsLoadingHealth(false);
    }
  };

  // Upload Handlers
  const handleFileUpload = async (file: File) => {
    if (!file) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      const data = await api.uploadDataset(file);
      setDatasetPreview({
        is_loaded: true,
        filename: data.filename,
        total_rows: data.total_rows,
        total_columns: data.total_columns,
        columns: data.columns,
        numeric_columns: data.numeric_columns,
        preview: data.preview,
        detected_target_col: data.detected_target_col,
        detected_datetime_col: data.detected_datetime_col,
        is_preprocessed: false,
      });
      if (data.detected_target_col) {
        setSelectedTargetCol(data.detected_target_col);
      }
      setPreprocessingSummary(null);
      setTrainResult(null);
      setActiveModelName(null);
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || 'Failed to upload CSV. Please verify the format.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSample = async () => {
    setIsUploading(true);
    setUploadError(null);
    try {
      const data = await api.loadSampleDataset();
      setDatasetPreview({
        is_loaded: true,
        filename: data.filename,
        total_rows: data.total_rows,
        total_columns: data.total_columns,
        columns: data.columns,
        numeric_columns: data.numeric_columns,
        preview: data.preview,
        detected_target_col: data.detected_target_col,
        detected_datetime_col: data.detected_datetime_col,
        is_preprocessed: false,
      });
      if (data.detected_target_col) {
        setSelectedTargetCol(data.detected_target_col);
      }
      setPreprocessingSummary(null);
      setTrainResult(null);
      setActiveModelName(null);
    } catch (err: any) {
      setUploadError(err.response?.data?.detail || 'Error loading sample dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  // Preprocessing Handler
  const handlePreprocess = async () => {
    if (!datasetPreview) return;
    setIsPreprocessing(true);
    setPreprocessError(null);
    try {
      const summary = await api.preprocessDataset(selectedTargetCol);
      setPreprocessingSummary(summary);
      setDatasetPreview(prev => (prev ? { ...prev, is_preprocessed: true } : null));
      setTrainResult(null);
      setActiveModelName(null);
    } catch (err: any) {
      setPreprocessError(err.response?.data?.detail || 'Preprocessing error.');
    } finally {
      setIsPreprocessing(false);
    }
  };

  // Model Training Handlers
  const handleTrainModel = async (modelType: 'lstm' | 'gru' | 'compare') => {
    setIsTraining(true);
    setTrainError(null);

    const modelName = modelType === 'lstm' ? 'LSTM' : modelType === 'gru' ? 'GRU' : 'LSTM & GRU';
    setTrainingMessage(`Training TensorFlow ${modelName} neural network with Early Stopping...`);

    try {
      let result: TrainResponse;
      if (modelType === 'lstm') {
        result = await api.trainLSTM(12, 64);
        setActiveModelName('LSTM');
      } else if (modelType === 'gru') {
        result = await api.trainGRU(12, 64);
        setActiveModelName('GRU');
      } else {
        result = await api.trainCompare(12, 64);
        setActiveModelName('COMPARE');
      }

      setTrainResult(result);

      // Scroll smoothly to results
      setTimeout(() => {
        document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' });
      }, 200);
    } catch (err: any) {
      setTrainError(err.response?.data?.detail || 'Model training failed.');
    } finally {
      setIsTraining(false);
      setTrainingMessage('');
    }
  };

  // Prepare chart data from sample_predictions
  const chartData = (trainResult?.sample_predictions || []).slice(0, chartSlice).map(p => ({
    timestamp: p.timestamp.includes('T') ? p.timestamp.split('T')[1].slice(0, 5) : p.timestamp.slice(11, 16) || p.timestamp,
    fullTimestamp: p.timestamp,
    Actual: p.actual,
    Predicted: p.predicted,
    LSTM: p.lstm_predicted !== undefined ? p.lstm_predicted : undefined,
    GRU: p.gru_predicted !== undefined ? p.gru_predicted : undefined,
    isAnomaly: p.is_anomaly,
  }));

  // Prepare 24-hour future forecast trajectory data
  const forecastChartData = React.useMemo(() => {
    if (!forecast24h) return [];
    const points: any[] = [];

    // Historical 12 hours context
    (forecast24h.history_context || []).forEach(pt => {
      const timeStr = pt.timestamp.includes(' ') ? pt.timestamp.split(' ')[1].slice(0, 5) : pt.timestamp.slice(11, 16);
      points.push({
        label: timeStr,
        fullTimestamp: pt.timestamp,
        HistoricalActual: pt.actual_kw,
        Forecast24h: null,
        UpperBound: null,
        LowerBound: null,
        isHistorical: true,
      });
    });

    // Bridge historical to future
    if (points.length > 0 && forecast24h.forecast_points && forecast24h.forecast_points.length > 0) {
      const lastHist = points[points.length - 1];
      lastHist.Forecast24h = lastHist.HistoricalActual;
    }

    // 24-hour future predictions
    (forecast24h.forecast_points || []).forEach(pt => {
      const timeStr = pt.timestamp.includes(' ') ? pt.timestamp.split(' ')[1].slice(0, 5) : pt.timestamp.slice(11, 16);
      points.push({
        label: `+${pt.hour_ahead}h (${timeStr})`,
        fullTimestamp: pt.timestamp,
        HistoricalActual: null,
        Forecast24h: pt.forecast_kw,
        UpperBound: pt.upper_bound_kw,
        LowerBound: pt.lower_bound_kw,
        isHistorical: false,
      });
    });

    return points;
  }, [forecast24h]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans flex flex-col">
      {/* ── A. HEADER ──────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold shadow-sm">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-xl tracking-tight text-blue-900">PowerPulse</span>
                <span className="text-xs px-2 py-0.5 rounded font-medium bg-blue-100 text-blue-700">Deep Learning</span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">Electricity Consumption Prediction and Anomaly Detection</p>
            </div>
          </div>

          {/* In-page Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            <a href="#dataset" className="px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-blue-700 hover:bg-slate-100 rounded-md transition-colors">
              1. Dataset
            </a>
            <a href="#processing" className="px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-blue-700 hover:bg-slate-100 rounded-md transition-colors">
              2. Processing
            </a>
            <a href="#training" className="px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-blue-700 hover:bg-slate-100 rounded-md transition-colors">
              3. Training
            </a>
            {trainResult && (
              <>
                <a href="#results" className="px-3 py-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 hover:bg-emerald-100 rounded-md transition-colors">
                  4. Results
                </a>
                <a href="#forecast24h" className="px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors flex items-center space-x-1">
                  <Sparkles className="w-3 h-3 text-blue-600" />
                  <span>5. Next 24h Forecast</span>
                </a>
              </>
            )}
          </nav>

          {/* Backend Status */}
          <div className="flex items-center space-x-3">
            {backendOnline ? (
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
                <span className="w-2 h-2 mr-1.5 bg-emerald-500 rounded-full animate-pulse"></span>
                API Online
              </span>
            ) : (
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-rose-100 text-rose-800">
                <span className="w-2 h-2 mr-1.5 bg-rose-500 rounded-full"></span>
                API Offline
              </span>
            )}
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT CONTAINER ─────────────────────────────────────────── */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-8 space-y-8">

        {/* ── B. DATASET UPLOAD ──────────────────────────────────────────────── */}
        <section id="dataset" className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 mb-4 border-b border-slate-100 gap-2">
            <div>
              <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
                <Database className="w-5 h-5 text-blue-600" />
                <span>Dataset Upload</span>
              </h2>
              <p className="text-sm text-slate-500">
                Upload your electricity consumption CSV or load the benchmark UCI Household Power sample dataset.
              </p>
            </div>
            <button
              onClick={handleLoadSample}
              disabled={isUploading}
              className="inline-flex items-center justify-center px-3.5 py-2 text-xs font-semibold rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${isUploading ? 'animate-spin' : ''}`} />
              ⚡ Load UCI Household Power Demo (90 Days)
            </button>
          </div>

          {/* Drag & Drop Area */}
          <div
            onDragOver={e => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={e => {
              e.preventDefault();
              setIsDragOver(false);
              if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileUpload(e.dataTransfer.files[0]);
              }
            }}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
              isDragOver ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-blue-400 bg-slate-50/50'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept=".csv,.txt"
              className="hidden"
              onChange={e => {
                if (e.target.files && e.target.files[0]) {
                  handleFileUpload(e.target.files[0]);
                }
              }}
            />
            <div className="mx-auto w-12 h-12 mb-3 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-sm font-semibold text-slate-700">
              {isUploading ? 'Uploading and validating CSV...' : 'Drag and drop your electricity consumption CSV here, or click to browse'}
            </p>
            <p className="text-xs text-slate-500 mt-1">Supports standard CSV or UCI semicolon format (.csv, .txt)</p>
          </div>

          {uploadError && (
            <div className="mt-4 p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{uploadError}</span>
            </div>
          )}

          {/* Uploaded Dataset Metadata & 10-Row Preview */}
          {datasetPreview && datasetPreview.is_loaded && (
            <div className="mt-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between bg-slate-50 border border-slate-200 p-3.5 rounded-lg text-sm gap-2">
                <div className="flex items-center space-x-2">
                  <FileText className="w-4 h-4 text-blue-600" />
                  <span className="font-semibold text-slate-800">{datasetPreview.filename}</span>
                  <span className="text-slate-400">•</span>
                  <span className="text-slate-600">{datasetPreview.total_rows.toLocaleString()} rows</span>
                  <span className="text-slate-400">•</span>
                  <span className="text-slate-600">{datasetPreview.total_columns} columns</span>
                </div>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">
                  <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Valid CSV
                </span>
              </div>

              {/* Preview Table */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Preview of First 10 Rows
                  </span>
                  <span className="text-xs text-slate-400">Scroll horizontally to view all attributes</span>
                </div>
                <div className="overflow-x-auto border border-slate-200 rounded-lg">
                  <table className="min-w-full divide-y divide-slate-200 text-xs text-left">
                    <thead className="bg-slate-100 text-slate-700 font-semibold">
                      <tr>
                        <th className="px-3 py-2 w-12 text-slate-400">#</th>
                        {datasetPreview.columns.map(col => (
                          <th key={col} className={`px-3 py-2 ${col === selectedTargetCol ? 'bg-blue-100 text-blue-900 font-bold' : ''}`}>
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 bg-white">
                      {datasetPreview.preview.map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-50/80">
                          <td className="px-3 py-2 font-mono text-slate-400">{idx + 1}</td>
                          {datasetPreview.columns.map(col => (
                            <td key={col} className={`px-3 py-2 font-mono ${col === selectedTargetCol ? 'bg-blue-50/50 font-semibold text-blue-900' : 'text-slate-700'}`}>
                              {row[col] !== null && row[col] !== undefined ? String(row[col]) : <span className="text-slate-400">NaN</span>}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ── C. DATA PROCESSING ────────────────────────────────────────────── */}
        <section id="processing" className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
          <div className="pb-4 mb-4 border-b border-slate-100">
            <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
              <Sliders className="w-5 h-5 text-blue-600" />
              <span>Data Processing & Hygiene Pipeline</span>
            </h2>
            <p className="text-sm text-slate-500">
              Clean missing readings, resample to hourly means, split chronologically (70% Train, 15% Val, 15% Test), and normalize with MinMaxScaler.
            </p>
          </div>

          {!datasetPreview ? (
            <div className="text-center py-8 text-slate-400 text-sm">
              Please upload or load a dataset above to configure data preprocessing.
            </div>
          ) : (
            <div className="space-y-6">
              {/* Column Selection Controls */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-50 p-4 rounded-xl border border-slate-200">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Target Electricity Consumption Column
                  </label>
                  <select
                    value={selectedTargetCol}
                    onChange={e => setSelectedTargetCol(e.target.value)}
                    className="w-full text-sm bg-white border border-slate-300 rounded-lg px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {datasetPreview.numeric_columns.length > 0
                      ? datasetPreview.numeric_columns.map(col => (
                          <option key={col} value={col}>
                            {col} {col === 'Global_active_power' ? '(UCI Active Power kW)' : ''}
                          </option>
                        ))
                      : datasetPreview.columns.map(col => (
                          <option key={col} value={col}>
                            {col}
                          </option>
                        ))}
                  </select>
                  <p className="text-xs text-slate-500 mt-1">Select the numeric column representing electrical demand.</p>
                </div>

                <div className="flex flex-col justify-end">
                  <button
                    onClick={handlePreprocess}
                    disabled={isPreprocessing}
                    className="w-full inline-flex items-center justify-center px-4 py-2 text-sm font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors shadow-sm disabled:opacity-50"
                  >
                    {isPreprocessing ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Processing & Splitting...
                      </>
                    ) : (
                      <>
                        <Sliders className="w-4 h-4 mr-2" />
                        Run Preprocessing Pipeline
                      </>
                    )}
                  </button>
                </div>
              </div>

              {preprocessError && (
                <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-center space-x-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{preprocessError}</span>
                </div>
              )}

              {/* Preprocessing Summary Card */}
              {preprocessingSummary && (
                <div className="border border-emerald-200 bg-emerald-50/40 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="inline-flex items-center text-xs font-bold uppercase tracking-wider text-emerald-800">
                      <CheckCircle2 className="w-4 h-4 mr-1.5 text-emerald-600" />
                      Preprocessing Summary
                    </span>
                    <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
                      No Data Leakage (Scaler on Train Split Only)
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-white p-3 rounded-lg border border-slate-200 shadow-2xs">
                      <p className="text-xs text-slate-500">Hourly Records</p>
                      <p className="text-base font-bold text-slate-900">{preprocessingSummary.hourly_records.toLocaleString()}</p>
                      <p className="text-2xs text-slate-400">1-hour resampled mean</p>
                    </div>
                    <div className="bg-white p-3 rounded-lg border border-slate-200 shadow-2xs">
                      <p className="text-xs text-slate-500">Split (Train / Val / Test)</p>
                      <p className="text-base font-bold text-slate-900">
                        {preprocessingSummary.train_samples} / {preprocessingSummary.val_samples} / {preprocessingSummary.test_samples}
                      </p>
                      <p className="text-2xs text-slate-400">70% / 15% / 15% chronological</p>
                    </div>
                    <div className="bg-white p-3 rounded-lg border border-slate-200 shadow-2xs">
                      <p className="text-xs text-slate-500">Input Sequence</p>
                      <p className="text-base font-bold text-slate-900">{preprocessingSummary.input_sequence_length} Hours</p>
                      <p className="text-2xs text-slate-400">Sliding historical lag window</p>
                    </div>
                    <div className="bg-white p-3 rounded-lg border border-slate-200 shadow-2xs">
                      <p className="text-xs text-slate-500">Mean Consumption</p>
                      <p className="text-base font-bold text-slate-900">{preprocessingSummary.mean_kw} kW</p>
                      <p className="text-2xs text-slate-400">Min: {preprocessingSummary.min_kw} | Max: {preprocessingSummary.max_kw}</p>
                    </div>
                  </div>

                  <p className="text-xs text-emerald-900 leading-relaxed">
                    Data sanitized from <strong>{preprocessingSummary.initial_raw_rows.toLocaleString()}</strong> raw records.
                    Handled <strong>{preprocessingSummary.missing_values_handled}</strong> missing/gap points (conservative short-gap interpolation &le; 2 hours).
                    MinMaxScaler min ({preprocessingSummary.scaler_train_min} kW) and max ({preprocessingSummary.scaler_train_max} kW) calibrated strictly on the training partition.
                  </p>
                </div>
              )}
            </div>
          )}
        </section>

        {/* ── D. MODEL TRAINING ────────────────────────────────────────────── */}
        <section id="training" className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
          <div className="pb-4 mb-4 border-b border-slate-100">
            <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
              <Cpu className="w-5 h-5 text-blue-600" />
              <span>Model Training (TensorFlow / Keras)</span>
            </h2>
            <p className="text-sm text-slate-500">
              Train real LSTM or GRU recurrent networks. Real deep learning execution with EarlyStopping on validation loss.
            </p>
          </div>

          {/* Model Architecture Specification */}
          <div className="mb-6 p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-2">
            <div className="font-semibold text-slate-700 flex items-center space-x-1.5">
              <Layers className="w-4 h-4 text-blue-600" />
              <span>Neural Network Architecture Specification:</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 font-mono text-slate-600">
              <div className="p-2 bg-white rounded border border-slate-200">
                <span className="font-bold text-blue-800">Input:</span> (24, 1) lag steps
              </div>
              <div className="p-2 bg-white rounded border border-slate-200">
                <span className="font-bold text-blue-800">Recurrent:</span> LSTM(32) or GRU(32)
              </div>
              <div className="p-2 bg-white rounded border border-slate-200">
                <span className="font-bold text-blue-800">Hidden:</span> Dense(16, ReLU)
              </div>
              <div className="p-2 bg-white rounded border border-slate-200">
                <span className="font-bold text-blue-800">Output:</span> Dense(1, Linear)
              </div>
            </div>
            <p className="text-slate-500">Optimizer: Adam (learning_rate=0.001) • Loss: Mean Squared Error (MSE) • Early Stopping (patience=3)</p>
          </div>

          {/* Training Action Buttons */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <button
              onClick={() => handleTrainModel('lstm')}
              disabled={isTraining || (!datasetPreview && !preprocessingSummary)}
              className="p-4 rounded-xl border-2 border-blue-200 hover:border-blue-500 bg-white hover:bg-blue-50/50 transition-all text-left group disabled:opacity-50 shadow-2xs"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-sm text-slate-900 group-hover:text-blue-700">Train LSTM</span>
                <span className="text-2xs font-semibold px-2 py-0.5 rounded bg-blue-100 text-blue-800">32 Units</span>
              </div>
              <p className="text-xs text-slate-500">Long Short-Term Memory network with separate cell and hidden state vectors.</p>
              <div className="mt-3 flex items-center text-xs font-semibold text-blue-600">
                Execute Training <ArrowRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>

            <button
              onClick={() => handleTrainModel('gru')}
              disabled={isTraining || (!datasetPreview && !preprocessingSummary)}
              className="p-4 rounded-xl border-2 border-emerald-200 hover:border-emerald-500 bg-white hover:bg-emerald-50/50 transition-all text-left group disabled:opacity-50 shadow-2xs"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-sm text-slate-900 group-hover:text-emerald-700">Train GRU</span>
                <span className="text-2xs font-semibold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">32 Units</span>
              </div>
              <p className="text-xs text-slate-500">Gated Recurrent Unit with reset and update gates for faster convergence.</p>
              <div className="mt-3 flex items-center text-xs font-semibold text-emerald-600">
                Execute Training <ArrowRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>

            <button
              onClick={() => handleTrainModel('compare')}
              disabled={isTraining || (!datasetPreview && !preprocessingSummary)}
              className="p-4 rounded-xl border-2 border-purple-200 hover:border-purple-500 bg-purple-50/30 hover:bg-purple-50 transition-all text-left group disabled:opacity-50 shadow-2xs"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-sm text-slate-900 group-hover:text-purple-700">Train & Compare Both</span>
                <span className="text-2xs font-semibold px-2 py-0.5 rounded bg-purple-100 text-purple-800">LSTM vs GRU</span>
              </div>
              <p className="text-xs text-slate-500">Train both architectures sequentially on identical splits to evaluate performance.</p>
              <div className="mt-3 flex items-center text-xs font-semibold text-purple-600">
                Run Comparison <ArrowRight className="w-3.5 h-3.5 ml-1 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>
          </div>

          {/* Training In-Progress Spinner */}
          {isTraining && (
            <div className="mt-6 p-4 rounded-xl bg-blue-50 border border-blue-200 flex items-center space-x-3 text-blue-900">
              <RefreshCw className="w-5 h-5 animate-spin text-blue-600 shrink-0" />
              <div>
                <p className="text-sm font-semibold">{trainingMessage}</p>
                <p className="text-xs text-blue-700">Executing Keras Sequential model on CPU/GPU. Early stopping will restore optimal weights.</p>
              </div>
            </div>
          )}

          {trainError && (
            <div className="mt-4 p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{trainError}</span>
            </div>
          )}
        </section>

        {/* ── E. RESULTS & ANOMALY DETECTION ─────────────────────────────────── */}
        {trainResult && (
          <div id="results" className="space-y-8">
            
            {/* 1. Model Performance & Comparison Table */}
            <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <div className="flex flex-wrap items-center justify-between pb-4 mb-4 border-b border-slate-100 gap-2">
                <div>
                  <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
                    <TrendingUp className="w-5 h-5 text-blue-600" />
                    <span>Model Performance Metrics</span>
                  </h2>
                  <p className="text-sm text-slate-500">
                    Evaluated in real kilowatt (kW) units on the held-out test split (15%).
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-semibold px-3 py-1 rounded-full bg-blue-100 text-blue-800">
                    Active: {trainResult.model_name || trainResult.best_model}
                  </span>
                </div>
              </div>

              {/* Single Model Metric Cards (or Winning Model) */}
              {trainResult.metrics && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold text-slate-500 uppercase">Test MAE</span>
                    <p className="text-2xl font-black text-slate-900 mt-1">{trainResult.metrics.mae.toFixed(4)} <span className="text-xs font-medium text-slate-500">kW</span></p>
                    <p className="text-2xs text-slate-400 mt-0.5">Mean Absolute Error</p>
                  </div>
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold text-slate-500 uppercase">Test RMSE</span>
                    <p className="text-2xl font-black text-slate-900 mt-1">{trainResult.metrics.rmse.toFixed(4)} <span className="text-xs font-medium text-slate-500">kW</span></p>
                    <p className="text-2xs text-slate-400 mt-0.5">Root Mean Squared Error</p>
                  </div>
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold text-slate-500 uppercase">Validation MAE</span>
                    <p className="text-2xl font-black text-slate-900 mt-1">{trainResult.metrics.val_mae.toFixed(4)} <span className="text-xs font-medium text-slate-500">kW</span></p>
                    <p className="text-2xs text-slate-400 mt-0.5">Validation Baseline</p>
                  </div>
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold text-slate-500 uppercase">Training Duration</span>
                    <p className="text-2xl font-black text-slate-900 mt-1">{trainResult.metrics.training_time_seconds.toFixed(2)} <span className="text-xs font-medium text-slate-500">sec</span></p>
                    <p className="text-2xs text-slate-400 mt-0.5">{trainResult.metrics.epochs_trained} epochs ({trainResult.metrics.parameters_count.toLocaleString()} params)</p>
                  </div>
                </div>
              )}

              {/* LSTM vs GRU Comparison Table */}
              {trainResult.comparison_table && trainResult.comparison_table.length > 0 && (
                <div className="mt-6 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-slate-800 flex items-center space-x-1.5">
                      <TableIcon className="w-4 h-4 text-purple-600" />
                      <span>LSTM vs GRU Comparative Evaluation Table</span>
                    </h3>
                    <span className="text-xs text-slate-500">Model selection strictly based on lower Validation MAE</span>
                  </div>

                  <div className="overflow-x-auto border border-slate-200 rounded-lg">
                    <table className="min-w-full divide-y divide-slate-200 text-xs">
                      <thead className="bg-slate-100 text-slate-700 font-semibold">
                        <tr>
                          <th className="px-4 py-2.5 text-left">Architecture</th>
                          <th className="px-4 py-2.5 text-right">Val MAE (kW)</th>
                          <th className="px-4 py-2.5 text-right">Val RMSE (kW)</th>
                          <th className="px-4 py-2.5 text-right">Test MAE (kW)</th>
                          <th className="px-4 py-2.5 text-right">Test RMSE (kW)</th>
                          <th className="px-4 py-2.5 text-right">Train Time (s)</th>
                          <th className="px-4 py-2.5 text-right">Parameters</th>
                          <th className="px-4 py-2.5 text-center">Selection</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white">
                        {trainResult.comparison_table.map((row: ComparisonRow) => (
                          <tr key={row.model} className={row.is_best ? 'bg-emerald-50/50 font-semibold' : 'hover:bg-slate-50'}>
                            <td className="px-4 py-3 flex items-center space-x-2">
                              <span className="font-bold text-slate-900">{row.model}</span>
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-slate-800">{row.val_mae.toFixed(4)}</td>
                            <td className="px-4 py-3 text-right font-mono text-slate-800">{row.val_rmse.toFixed(4)}</td>
                            <td className="px-4 py-3 text-right font-mono text-slate-800">{row.test_mae.toFixed(4)}</td>
                            <td className="px-4 py-3 text-right font-mono text-slate-800">{row.test_rmse.toFixed(4)}</td>
                            <td className="px-4 py-3 text-right font-mono text-slate-800">{row.training_time_seconds.toFixed(2)}</td>
                            <td className="px-4 py-3 text-right font-mono text-slate-800">{row.parameters.toLocaleString()}</td>
                            <td className="px-4 py-3 text-center">
                              {row.is_best ? (
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-2xs font-bold bg-emerald-100 text-emerald-800">
                                  ★ Best Model
                                </span>
                              ) : (
                                <span className="text-slate-400 text-2xs">Alternative</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {trainResult.best_model_reason && (
                    <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-xs text-emerald-900 flex items-start space-x-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                      <div>
                        <strong>Scientific Selection Rationale:</strong> {trainResult.best_model_reason} Evaluated in true kilowatt units without post-selection bias.
                      </div>
                    </div>
                  )}
                </div>
              )}
            </section>

            {/* 2. Prediction Results: Graph & Sample Predictions Table */}
            <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <div className="flex flex-wrap items-center justify-between pb-4 mb-4 border-b border-slate-100 gap-2">
                <div>
                  <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
                    <Activity className="w-5 h-5 text-blue-600" />
                    <span>Actual vs Predicted Electricity Consumption</span>
                  </h2>
                  <p className="text-sm text-slate-500">
                    Comparing continuous ground-truth consumption against recurrent predictions across held-out test hours.
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-slate-500">Show points:</span>
                  {[50, 100, 200].map(cnt => (
                    <button
                      key={cnt}
                      onClick={() => setChartSlice(cnt)}
                      className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
                        chartSlice === cnt ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                      }`}
                    >
                      {cnt}h
                    </button>
                  ))}
                </div>
              </div>

              {/* Line Chart */}
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="timestamp" tick={{ fontSize: 11, fill: '#64748b' }} />
                    <YAxis tick={{ fontSize: 11, fill: '#64748b' }} unit=" kW" />
                    <Tooltip
                      formatter={(value: any, name: any) => [`${Number(value).toFixed(3)} kW`, name]}
                      labelFormatter={(label, items) => {
                        const item = items && items[0]?.payload;
                        return item ? `Timestamp: ${item.fullTimestamp}` : `Time: ${label}`;
                      }}
                      contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                    <Line type="monotone" dataKey="Actual" stroke="#0f172a" strokeWidth={2} dot={false} name="Actual (kW)" />
                    {activeModelName === 'COMPARE' ? (
                      <>
                        <Line type="monotone" dataKey="LSTM" stroke="#2563eb" strokeWidth={1.8} strokeDasharray="4 4" dot={false} name="LSTM Prediction" />
                        <Line type="monotone" dataKey="GRU" stroke="#059669" strokeWidth={1.8} strokeDasharray="2 2" dot={false} name="GRU Prediction" />
                      </>
                    ) : (
                      <Line type="monotone" dataKey="Predicted" stroke="#2563eb" strokeWidth={2} dot={false} name="Predicted (kW)" />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Sample Predictions Table */}
              <div className="mt-6">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
                  Sample Predictions on Held-Out Test Split
                </h3>
                <div className="overflow-x-auto border border-slate-200 rounded-lg">
                  <table className="min-w-full divide-y divide-slate-200 text-xs">
                    <thead className="bg-slate-100 text-slate-700 font-semibold">
                      <tr>
                        <th className="px-3 py-2 text-left">Timestamp</th>
                        <th className="px-3 py-2 text-right">Actual (kW)</th>
                        <th className="px-3 py-2 text-right">Predicted (kW)</th>
                        <th className="px-3 py-2 text-right">Error (Actual - Pred)</th>
                        <th className="px-3 py-2 text-center">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 bg-white font-mono">
                      {trainResult.sample_predictions.slice(0, 10).map((p: PredictionPoint, idx: number) => (
                        <tr key={idx} className={p.is_anomaly ? 'bg-amber-50/70' : 'hover:bg-slate-50'}>
                          <td className="px-3 py-2 text-slate-700">{p.timestamp}</td>
                          <td className="px-3 py-2 text-right font-bold text-slate-900">{p.actual.toFixed(3)}</td>
                          <td className="px-3 py-2 text-right text-blue-700">{p.predicted.toFixed(3)}</td>
                          <td className="px-3 py-2 text-right text-slate-600">{p.residual.toFixed(3)}</td>
                          <td className="px-3 py-2 text-center">
                            {p.is_anomaly ? (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-2xs font-bold bg-amber-100 text-amber-800">
                                Anomaly
                              </span>
                            ) : (
                              <span className="text-slate-400 text-2xs font-sans">Normal</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* 3. Anomaly Detection Section */}
            <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <div className="pb-4 mb-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
                    <AlertTriangle className="w-5 h-5 text-amber-600" />
                    <span>Statistical Anomaly Detection</span>
                  </h2>
                  <p className="text-sm text-slate-500">
                    Threshold derived strictly from the 95th percentile of positive validation residuals.
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-amber-100 text-amber-900 border border-amber-200">
                    {trainResult.total_anomalies} Anomalies Flagged
                  </span>
                </div>
              </div>

              {/* Anomaly Metrics */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <div className="bg-amber-50/50 border border-amber-200 rounded-xl p-4">
                  <span className="text-xs font-semibold text-amber-900 uppercase">Anomaly Threshold</span>
                  <p className="text-2xl font-black text-amber-900 mt-1">
                    +{trainResult.anomaly_threshold_kw.toFixed(4)} <span className="text-xs font-medium text-amber-700">kW</span>
                  </p>
                  <p className="text-2xs text-amber-700 mt-0.5">95th percentile of positive validation residuals</p>
                </div>

                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                  <span className="text-xs font-semibold text-slate-500 uppercase">Detection Criteria</span>
                  <p className="text-sm font-semibold text-slate-800 mt-2">
                    Residual = Actual - Predicted &gt; Threshold
                  </p>
                  <p className="text-2xs text-slate-500 mt-0.5">Evaluated across test split without test tuning</p>
                </div>
              </div>

              {/* Academic Disclaimer Callout */}
              <div className="p-4 rounded-xl bg-blue-50 border border-blue-200 text-xs text-blue-900 flex items-start space-x-3 mb-6">
                <Info className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
                <div className="leading-relaxed">
                  <strong>Academic Presentation Note:</strong> An anomaly indicates an unusually high consumption reading relative to historical patterns (Actual &gt; Predicted + Threshold). It represents a statistically atypical surge, not necessarily equipment defect or intentional energy wastage.
                </div>
              </div>

              {/* Anomalies Table */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
                  Detected Anomalous Consumption Events ({trainResult.anomalies.length})
                </h3>
                {trainResult.anomalies.length === 0 ? (
                  <div className="text-center py-6 bg-slate-50 rounded-lg text-xs text-slate-500">
                    No anomalies exceeded the 95th percentile residual threshold on this test split.
                  </div>
                ) : (
                  <div className="overflow-x-auto border border-slate-200 rounded-lg max-h-72 overflow-y-auto">
                    <table className="min-w-full divide-y divide-slate-200 text-xs">
                      <thead className="bg-slate-100 text-slate-700 font-semibold sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-left">Timestamp</th>
                          <th className="px-3 py-2 text-right">Actual (kW)</th>
                          <th className="px-3 py-2 text-right">Predicted (kW)</th>
                          <th className="px-3 py-2 text-right">Residual Spike (kW)</th>
                          <th className="px-3 py-2 text-center">Severity</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white font-mono">
                        {trainResult.anomalies.map((anom: AnomalyPoint) => (
                          <tr key={anom.id} className="hover:bg-slate-50">
                            <td className="px-3 py-2 text-slate-800">{anom.timestamp}</td>
                            <td className="px-3 py-2 text-right font-bold text-slate-900">{anom.actual.toFixed(3)}</td>
                            <td className="px-3 py-2 text-right text-slate-600">{anom.predicted.toFixed(3)}</td>
                            <td className="px-3 py-2 text-right font-semibold text-rose-600">+{anom.residual.toFixed(3)}</td>
                            <td className="px-3 py-2 text-center">
                              {anom.severity === 'Critical' ? (
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-2xs font-bold bg-rose-100 text-rose-800">
                                  Critical (&ge;99%)
                                </span>
                              ) : anom.severity === 'High' ? (
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-2xs font-bold bg-amber-100 text-amber-800">
                                  High (&ge;95%)
                                </span>
                              ) : (
                                <span className="inline-flex items-center px-2 py-0.5 rounded text-2xs font-bold bg-blue-100 text-blue-800">
                                  Moderate
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </section>

            {/* ── 4. NEXT 24 HOURS ELECTRICITY CONSUMPTION FORECAST ─────────────── */}
            {forecast24h && (
              <section id="forecast24h" className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
                <div className="flex flex-wrap items-center justify-between pb-4 border-b border-slate-100 gap-2">
                  <div>
                    <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
                      <Sparkles className="w-5 h-5 text-blue-600" />
                      <span>Next 24 Hours Electricity Consumption Forecast</span>
                    </h2>
                    <p className="text-sm text-slate-500">
                      Roll-forward recursive multi-step forecasting using the optimal architecture ({forecast24h.model_used}) projected from the latest 24-hour readings.
                    </p>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800">
                      <Zap className="w-3.5 h-3.5 mr-1 text-blue-600" />
                      Model: {forecast24h.model_used} {forecast24h.is_best_model ? '(Optimal Model)' : ''}
                    </span>
                    <button
                      onClick={() => handleReforecast()}
                      disabled={isForecasting}
                      className="inline-flex items-center px-3 py-1.5 rounded-lg border border-slate-300 hover:bg-slate-50 text-slate-700 text-xs font-semibold transition-colors disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${isForecasting ? 'animate-spin' : ''}`} />
                      Re-forecast
                    </button>
                  </div>
                </div>

                {/* 4 Forecast Metric Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="bg-blue-50/60 border border-blue-200 rounded-xl p-4">
                    <span className="text-xs font-semibold text-blue-900 uppercase">Total 24h Projected Energy</span>
                    <p className="text-2xl font-black text-blue-950 mt-1">
                      {forecast24h.total_projected_kwh.toFixed(2)} <span className="text-xs font-medium text-blue-700">kWh</span>
                    </p>
                    <p className="text-2xs text-blue-700 mt-0.5">Cumulative load over next 24 hours</p>
                  </div>

                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold text-slate-500 uppercase">Projected Peak Demand</span>
                    <p className="text-2xl font-black text-rose-600 mt-1">
                      {forecast24h.peak_forecast_kw.toFixed(3)} <span className="text-xs font-medium text-slate-500">kW</span>
                    </p>
                    <p className="text-2xs text-slate-500 mt-0.5 truncate" title={forecast24h.peak_hour}>
                      Peak hour: {forecast24h.peak_hour.slice(11, 16) || forecast24h.peak_hour}
                    </p>
                  </div>

                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold text-slate-500 uppercase">Projected Base Load</span>
                    <p className="text-2xl font-black text-emerald-600 mt-1">
                      {forecast24h.min_forecast_kw.toFixed(3)} <span className="text-xs font-medium text-slate-500">kW</span>
                    </p>
                    <p className="text-2xs text-slate-500 mt-0.5 truncate" title={forecast24h.min_hour}>
                      Minimum hour: {forecast24h.min_hour.slice(11, 16) || forecast24h.min_hour}
                    </p>
                  </div>

                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <span className="text-xs font-semibold text-slate-500 uppercase">Average Forecast Load</span>
                    <p className="text-2xl font-black text-slate-900 mt-1">
                      {forecast24h.average_forecast_kw.toFixed(3)} <span className="text-xs font-medium text-slate-500">kW</span>
                    </p>
                    <p className="text-2xs text-slate-500 mt-0.5">Last reading: {forecast24h.last_known_reading_kw} kW</p>
                  </div>
                </div>

                {/* Model Selection & Roll-forward Rationale Callout */}
                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 flex items-start space-x-3">
                  <Info className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
                  <div className="leading-relaxed">
                    <strong>Forecasting Methodology:</strong> Initialized with the most recent 24-hour historical window ending at <strong>{forecast24h.base_timestamp}</strong>. The <strong>{forecast24h.model_used}</strong> model ({forecast24h.selection_rationale}) recursively rolls forward each prediction y&#770;(t+k) to predict future step t+k+1. The light blue dashed lines show the expanding 95% confidence interval envelope accounting for recursive uncertainty accumulation over 24 hours.
                  </div>
                </div>

                {/* Continuous Trajectory Chart: Past 12h History -> Next 24h Forecast */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                      <TrendingUp className="w-4 h-4 text-blue-600" />
                      <span>Continuous Load Trajectory (Past 12h Ground Truth &rarr; Next 24h Projected Load)</span>
                    </h3>
                    <span className="text-xs text-slate-400">Solid: Actual History • Bold Blue: 24h Forecast • Dashed: 95% CI</span>
                  </div>

                  <div className="h-80 w-full border border-slate-200 rounded-xl p-3 bg-white">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={forecastChartData} margin={{ top: 10, right: 25, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#64748b' }} interval="preserveStartEnd" />
                        <YAxis tick={{ fontSize: 11, fill: '#64748b' }} unit=" kW" domain={['auto', 'auto']} />
                        <Tooltip
                          formatter={(value: any, name: any) => [`${Number(value).toFixed(3)} kW`, name]}
                          labelFormatter={(label, items) => {
                            const item = items && items[0]?.payload;
                            return item ? `${item.isHistorical ? 'Historical Ground Truth' : 'Forecast'}: ${item.fullTimestamp}` : String(label);
                          }}
                          contentStyle={{ backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '12px' }}
                        />
                        <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                        <Line
                          type="monotone"
                          dataKey="HistoricalActual"
                          stroke="#475569"
                          strokeWidth={2}
                          dot={{ r: 2, fill: '#475569' }}
                          name="Historical Actual (Past 12h)"
                          connectNulls={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="Forecast24h"
                          stroke="#2563eb"
                          strokeWidth={2.5}
                          dot={{ r: 3, fill: '#2563eb' }}
                          name="24h Projected Forecast (kW)"
                          connectNulls={true}
                        />
                        <Line
                          type="monotone"
                          dataKey="UpperBound"
                          stroke="#93c5fd"
                          strokeDasharray="3 3"
                          strokeWidth={1.5}
                          dot={false}
                          name="Upper 95% Confidence Bound"
                          connectNulls={true}
                        />
                        <Line
                          type="monotone"
                          dataKey="LowerBound"
                          stroke="#93c5fd"
                          strokeDasharray="3 3"
                          strokeWidth={1.5}
                          dot={false}
                          name="Lower 95% Confidence Bound"
                          connectNulls={true}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* 24-Hour Breakdown Table */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                      24-Hour Projected Consumption Breakdown
                    </h3>
                    <a
                      href={api.getForecastCsvUrl()}
                      download="powerpulse_24h_forecast.csv"
                      className="inline-flex items-center text-xs font-semibold text-blue-600 hover:text-blue-800"
                    >
                      <Download className="w-3.5 h-3.5 mr-1" />
                      Download 24h Forecast CSV
                    </a>
                  </div>

                  <div className="overflow-x-auto border border-slate-200 rounded-lg max-h-72 overflow-y-auto">
                    <table className="min-w-full divide-y divide-slate-200 text-xs">
                      <thead className="bg-slate-100 text-slate-700 font-semibold sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-center w-16">Horizon</th>
                          <th className="px-3 py-2 text-left">Projected Timestamp</th>
                          <th className="px-3 py-2 text-right">Forecast (kW)</th>
                          <th className="px-3 py-2 text-right">95% Confidence Range</th>
                          <th className="px-3 py-2 text-center">Projected Profile</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white font-mono">
                        {forecast24h.forecast_points.map((pt: ForecastPoint24h) => {
                          const isPeak = pt.timestamp === forecast24h.peak_hour;
                          const isMin = pt.timestamp === forecast24h.min_hour;
                          return (
                            <tr key={pt.hour_ahead} className={isPeak ? 'bg-rose-50/60 font-semibold' : isMin ? 'bg-emerald-50/60 font-semibold' : 'hover:bg-slate-50'}>
                              <td className="px-3 py-2 text-center text-slate-500 font-bold">+{pt.hour_ahead}h</td>
                              <td className="px-3 py-2 text-slate-800">{pt.timestamp}</td>
                              <td className="px-3 py-2 text-right text-blue-900 font-bold">{pt.forecast_kw.toFixed(3)} kW</td>
                              <td className="px-3 py-2 text-right text-slate-600">
                                [{pt.lower_bound_kw.toFixed(3)} – {pt.upper_bound_kw.toFixed(3)}] kW
                              </td>
                              <td className="px-3 py-2 text-center font-sans">
                                {isPeak ? (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded text-2xs font-bold bg-rose-100 text-rose-800">
                                    Peak Demand
                                  </span>
                                ) : isMin ? (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded text-2xs font-bold bg-emerald-100 text-emerald-800">
                                    Base Load
                                  </span>
                                ) : (
                                  <span className="text-slate-400 text-2xs">Normal Load</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </section>
            )}

            {/* ── F. DOWNLOAD RESULTS ────────────────────────────────────────── */}
            <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <div className="pb-4 mb-4 border-b border-slate-100">
                <h2 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
                  <Download className="w-5 h-5 text-blue-600" />
                  <span>Download Evaluation &amp; Forecast Results</span>
                </h2>
                <p className="text-sm text-slate-500">
                  Export test predictions, 24-hour future forecast, detected anomalies, and comparative model matrices directly to CSV.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <a
                  href={api.getForecastCsvUrl()}
                  download="powerpulse_24h_forecast.csv"
                  className="inline-flex items-center justify-center p-3.5 rounded-lg border border-blue-200 bg-blue-50/50 hover:bg-blue-100 text-blue-900 text-xs font-semibold transition-colors shadow-2xs"
                >
                  <Sparkles className="w-4 h-4 mr-2 text-blue-600" />
                  Download 24h Forecast (CSV)
                </a>

                <a
                  href={api.getPredictionsCsvUrl()}
                  download="powerpulse_predictions.csv"
                  className="inline-flex items-center justify-center p-3.5 rounded-lg border border-slate-300 hover:border-blue-500 hover:bg-blue-50 text-slate-800 text-xs font-semibold transition-colors shadow-2xs"
                >
                  <Download className="w-4 h-4 mr-2 text-blue-600" />
                  Download Predictions (CSV)
                </a>

                <a
                  href={api.getAnomaliesCsvUrl()}
                  download="powerpulse_anomalies.csv"
                  className="inline-flex items-center justify-center p-3.5 rounded-lg border border-slate-300 hover:border-amber-500 hover:bg-amber-50 text-slate-800 text-xs font-semibold transition-colors shadow-2xs"
                >
                  <Download className="w-4 h-4 mr-2 text-amber-600" />
                  Download Anomalies (CSV)
                </a>

                <a
                  href={api.getComparisonCsvUrl()}
                  download="powerpulse_model_comparison.csv"
                  className="inline-flex items-center justify-center p-3.5 rounded-lg border border-slate-300 hover:border-purple-500 hover:bg-purple-50 text-slate-800 text-xs font-semibold transition-colors shadow-2xs"
                >
                  <Download className="w-4 h-4 mr-2 text-purple-600" />
                  Download Model Comparison (CSV)
                </a>
              </div>
            </section>

          </div>
        )}

      </main>

      {/* ── FOOTER ──────────────────────────────────────────────────────────── */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
        <div className="max-w-6xl mx-auto px-4 space-y-1">
          <p className="font-semibold text-slate-700">PowerPulse • College Deep Learning Academic Project</p>
          <p>
            Implemented with TensorFlow/Keras (Sequential LSTM &amp; GRU), FastAPI, Pandas, and React + Vite.
          </p>
          <p className="text-slate-400 text-2xs">
            Dataset Reference: UCI Individual Household Electric Power Consumption (Hébrail &amp; Bérard, EDF R&amp;D).
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
