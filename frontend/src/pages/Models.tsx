import React, { useState, useEffect } from 'react';
import {
  Cpu, Play, Award, CheckCircle2, AlertCircle, RefreshCw,
  Clock, Zap, Activity, Layers, ArrowRight, ShieldCheck
} from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend
} from 'recharts';
import { api } from '../services/api';
import { ModelComparison, TrainStatus, DatasetStatus, ModelMetrics } from '../types';
import { EmptyState, LoadingState } from '../components/EmptyState';

interface ModelsProps {
  datasetStatus: DatasetStatus | null;
  onRefresh: () => void;
}

export const Models: React.FC<ModelsProps> = ({ datasetStatus, onRefresh }) => {
  const [comparison, setComparison] = useState<ModelComparison | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [epochs, setEpochs] = useState<number>(15);
  const [batchSize, setBatchSize] = useState<number>(64);
  const [error, setError] = useState<string | null>(null);

  // Poll training status if training
  useEffect(() => {
    loadComparison();
    checkTrainingStatus();
  }, [datasetStatus?.models_trained]);

  useEffect(() => {
    let interval: any = null;
    if (trainStatus?.status === 'training') {
      interval = setInterval(async () => {
        try {
          const status = await api.getTrainingStatus();
          setTrainStatus(status);
          if (status.status === 'completed') {
            await loadComparison();
            onRefresh();
            clearInterval(interval);
          } else if (status.status === 'failed') {
            setError(status.error_message || 'Training failed.');
            clearInterval(interval);
          }
        } catch (e) {
          console.error('Error polling status:', e);
        }
      }, 1500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [trainStatus?.status]);

  const loadComparison = async () => {
    try {
      const data = await api.getModelComparison();
      setComparison(data);
    } catch (e) {
      console.error('Error loading comparison:', e);
    }
  };

  const checkTrainingStatus = async () => {
    try {
      const status = await api.getTrainingStatus();
      setTrainStatus(status);
    } catch (e) {
      console.error('Error checking status:', e);
    }
  };

  const handleStartTraining = async () => {
    if (!datasetStatus?.is_loaded) {
      setError("Please load a dataset before initiating training.");
      return;
    }

    setError(null);
    setLoading(true);
    try {
      const status = await api.triggerTraining(epochs, batchSize);
      setTrainStatus(status);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to trigger training.");
    } finally {
      setLoading(false);
    }
  };

  if (!datasetStatus?.is_loaded) {
    return (
      <EmptyState
        icon={Cpu}
        title="Dataset Required for Deep Learning"
        description="Neural networks require preprocessed sequential time-series data. Initialize the sample dataset to begin training LSTM and GRU models."
        actionText="Load Sample Dataset"
        onAction={() => api.loadSampleDataset().then(() => onRefresh())}
      />
    );
  }

  const lstm = comparison?.models?.['LSTM'];
  const gru = comparison?.models?.['GRU'];
  const isTraining = trainStatus?.status === 'training';

  return (
    <div className="space-y-6">
      {/* Top Banner and Training Launcher */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <h1 className="text-xl font-bold text-navy-950 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-500" />
            <span>Deep Learning Recurrent Model Lab</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Comparative training of LSTM vs GRU architectures with strict chronological held-out validation
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Epoch Selector */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg text-xs">
            <span className="text-slate-500 px-2 font-medium">Epochs:</span>
            {[5, 10, 15, 20].map(ep => (
              <button
                key={ep}
                onClick={() => setEpochs(ep)}
                disabled={isTraining}
                className={`px-2.5 py-1.5 rounded font-semibold transition-all ${
                  epochs === ep ? 'bg-white text-navy-950 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {ep}
              </button>
            ))}
          </div>

          {/* Batch Size Selector */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg text-xs">
            <span className="text-slate-500 px-2 font-medium">Batch:</span>
            {[32, 64, 128].map(b => (
              <button
                key={b}
                onClick={() => setBatchSize(b)}
                disabled={isTraining}
                className={`px-2.5 py-1.5 rounded font-semibold transition-all ${
                  batchSize === b ? 'bg-navy-900 text-white shadow-sm' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {b}
              </button>
            ))}
          </div>

          {/* Trigger Training Button */}
          <button
            onClick={handleStartTraining}
            disabled={isTraining || loading}
            className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-navy-950 bg-gradient-to-r from-emerald-400 to-cyan-400 hover:from-emerald-300 hover:to-cyan-300 rounded-lg shadow-sm transition-all disabled:opacity-50"
          >
            {isTraining ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )}
            <span>{isTraining ? 'Training in Progress...' : 'Train Models (LSTM + GRU)'}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-xl text-sm flex items-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Live Training Progress Bar (shows during training) */}
      {isTraining && trainStatus && (
        <div className="bg-navy-950 text-white p-6 rounded-xl border border-navy-800 shadow-xl">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-sm font-bold text-white">
                Training Neural Networks ({trainStatus.current_model || 'Initializing...'})
              </span>
            </div>
            <span className="text-xs font-mono font-bold text-emerald-400">
              {trainStatus.progress_percentage}% Completed
            </span>
          </div>

          {/* Progress track */}
          <div className="w-full bg-navy-800 rounded-full h-3 overflow-hidden p-0.5">
            <div
              className="bg-gradient-to-r from-emerald-400 to-cyan-400 h-full rounded-full transition-all duration-500 ease-out"
              style={{ width: `${Math.max(5, trainStatus.progress_percentage)}%` }}
            />
          </div>

          <div className="mt-3 flex justify-between text-xs text-slate-400">
            <span>
              {trainStatus.current_epoch && trainStatus.total_epochs
                ? `Epoch ${trainStatus.current_epoch} of ${trainStatus.total_epochs}`
                : 'Compiling TensorFlow computation graph...'}
            </span>
            <span>EarlyStopping active &bull; Adam lr=0.001</span>
          </div>
        </div>
      )}

      {/* Best Model Banner */}
      {comparison?.is_trained && comparison.best_model && (
        <div className="bg-gradient-to-r from-emerald-950 via-navy-900 to-navy-950 text-white p-6 rounded-2xl border border-emerald-500/30 shadow-lg relative overflow-hidden">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-start gap-3.5">
              <div className="w-11 h-11 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shrink-0">
                <Award className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">
                    Production Model Selected
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-400 text-navy-950">
                    {comparison.best_model}
                  </span>
                </div>
                <h2 className="text-xl font-extrabold text-white mt-1">
                  Optimal Architecture: {comparison.best_model} Model
                </h2>
                <p className="text-xs text-slate-300 mt-1 leading-relaxed max-w-2xl">
                  {comparison.selection_reason}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4 border-t md:border-t-0 md:border-l border-navy-800 pt-3 md:pt-0 md:pl-6 text-xs">
              <div>
                <span className="text-slate-400 block">Selection Rule:</span>
                <span className="font-semibold text-emerald-300">Lowest Validation MAE</span>
              </div>
              <div>
                <span className="text-slate-400 block">Unit:</span>
                <span className="font-semibold text-cyan-300">Original kW</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Side-by-Side Model Comparison Cards */}
      {comparison?.is_trained ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* LSTM Card */}
          <div className={`
            bg-white rounded-xl border p-6 shadow-sm relative transition-all
            ${comparison.best_model === 'LSTM' ? 'border-emerald-500 ring-1 ring-emerald-500/20' : 'border-slate-200/80'}
          `}>
            {comparison.best_model === 'LSTM' && (
              <span className="absolute top-4 right-4 px-2.5 py-0.5 rounded-full text-[10px] font-extrabold bg-emerald-100 text-emerald-800 border border-emerald-300 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> BEST MODEL
              </span>
            )}

            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600 font-bold font-mono">
                LSTM
              </div>
              <div>
                <h3 className="text-base font-bold text-navy-950">Long Short-Term Memory</h3>
                <p className="text-xs text-slate-500">Input(24,1) &rarr; LSTM(32) &rarr; Dense(16) &rarr; Dense(1)</p>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 my-4">
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Validation MAE</span>
                <span className="text-lg font-bold text-emerald-600 font-mono">{lstm?.val_mae} kW</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Validation RMSE</span>
                <span className="text-lg font-bold text-slate-800 font-mono">{lstm?.val_rmse} kW</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Test MAE</span>
                <span className="text-lg font-bold text-slate-800 font-mono">{lstm?.test_mae} kW</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Test RMSE</span>
                <span className="text-lg font-bold text-slate-800 font-mono">{lstm?.test_rmse} kW</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Test R&sup2; Score</span>
                <span className="text-lg font-bold text-cyan-600 font-mono">{lstm?.test_r2}</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Training Time</span>
                <span className="text-lg font-bold text-slate-800 font-mono">{lstm?.training_duration_seconds}s</span>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 flex justify-between text-xs text-slate-500">
              <span>Inference Latency: <strong>{lstm?.inference_latency_ms} ms/sample</strong></span>
              <span>Epochs: <strong>{lstm?.epochs_trained}</strong></span>
              <span>Params: <strong>{lstm?.architecture?.total_parameters?.toLocaleString() || '4,849'}</strong></span>
            </div>
          </div>

          {/* GRU Card */}
          <div className={`
            bg-white rounded-xl border p-6 shadow-sm relative transition-all
            ${comparison.best_model === 'GRU' ? 'border-cyan-500 ring-1 ring-cyan-500/20' : 'border-slate-200/80'}
          `}>
            {comparison.best_model === 'GRU' && (
              <span className="absolute top-4 right-4 px-2.5 py-0.5 rounded-full text-[10px] font-extrabold bg-cyan-100 text-cyan-800 border border-cyan-300 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> BEST MODEL
              </span>
            )}

            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-cyan-50 border border-cyan-200 flex items-center justify-center text-cyan-600 font-bold font-mono">
                GRU
              </div>
              <div>
                <h3 className="text-base font-bold text-navy-950">Gated Recurrent Unit</h3>
                <p className="text-xs text-slate-500">Input(24,1) &rarr; GRU(32) &rarr; Dense(16) &rarr; Dense(1)</p>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 my-4">
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Validation MAE</span>
                <span className="text-lg font-bold text-cyan-600 font-mono">{gru?.val_mae} kW</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Validation RMSE</span>
                <span className="text-lg font-bold text-slate-800 font-mono">{gru?.val_rmse} kW</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Test MAE</span>
                <span className="text-lg font-bold text-slate-800 font-mono">{gru?.test_mae} kW</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Test RMSE</span>
                <span className="text-lg font-bold text-slate-800 font-mono">{gru?.test_rmse} kW</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Test R&sup2; Score</span>
                <span className="text-lg font-bold text-cyan-600 font-mono">{gru?.test_r2}</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Training Time</span>
                <span className="text-lg font-bold text-slate-800 font-mono">{gru?.training_duration_seconds}s</span>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 flex justify-between text-xs text-slate-500">
              <span>Inference Latency: <strong>{gru?.inference_latency_ms} ms/sample</strong></span>
              <span>Epochs: <strong>{gru?.epochs_trained}</strong></span>
              <span>Params: <strong>{gru?.architecture?.total_parameters?.toLocaleString() || '3,809'}</strong></span>
            </div>
          </div>
        </div>
      ) : (
        <EmptyState
          icon={Cpu}
          title="Neural Network Models Not Yet Trained"
          description="Click 'Train Models' above to train both LSTM and GRU networks on the current dataset. The best model will be selected based on lowest validation MAE."
          actionText="Start Training"
          onAction={handleStartTraining}
        />
      )}

      {/* Loss Curves & Convergence */}
      {comparison?.is_trained && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* LSTM Loss Curve */}
          <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
            <h3 className="text-sm font-bold text-navy-950 mb-1 flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" />
              <span>LSTM Training & Validation Loss History</span>
            </h3>
            <p className="text-xs text-slate-500 mb-4">MSE Loss per epoch</p>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lstm?.loss_history || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis dataKey="epoch" tickLine={false} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="train_loss" name="Train MSE" stroke="#10B981" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="val_loss" name="Val MSE" stroke="#047857" strokeWidth={2} strokeDasharray="3 3" dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* GRU Loss Curve */}
          <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
            <h3 className="text-sm font-bold text-navy-950 mb-1 flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-500" />
              <span>GRU Training & Validation Loss History</span>
            </h3>
            <p className="text-xs text-slate-500 mb-4">MSE Loss per epoch</p>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={gru?.loss_history || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                  <XAxis dataKey="epoch" tickLine={false} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="train_loss" name="Train MSE" stroke="#06B6D4" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="val_loss" name="Val MSE" stroke="#0891B2" strokeWidth={2} strokeDasharray="3 3" dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Model Architecture Specifications Inspector */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
        <h3 className="text-base font-bold text-navy-950 flex items-center gap-2 mb-2">
          <Layers className="w-4 h-4 text-slate-700" />
          <span>Deep Learning Architecture Specifications</span>
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          Strict implementation adhering to the college deep-learning assignment specification
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <h4 className="font-bold text-navy-950 mb-2 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span>LSTM Architecture Blueprint</span>
            </h4>
            <ul className="space-y-1.5 text-slate-600 font-mono text-[11px]">
              <li>Input Layer: (None, 24, 1) &mdash; 24-hour sequence</li>
              <li>LSTM Hidden Layer: 32 units, tanh activation, sigmoid gates</li>
              <li>Dense Feature Layer: 16 units, ReLU activation</li>
              <li>Output Regression Layer: 1 unit (predicted next-hour kW)</li>
              <li>Optimizer: Adam (learning_rate = 0.001, beta_1=0.9)</li>
              <li>Loss Function: Mean Squared Error (MSE)</li>
            </ul>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <h4 className="font-bold text-navy-950 mb-2 flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-500" />
              <span>GRU Architecture Blueprint</span>
            </h4>
            <ul className="space-y-1.5 text-slate-600 font-mono text-[11px]">
              <li>Input Layer: (None, 24, 1) &mdash; 24-hour sequence</li>
              <li>GRU Hidden Layer: 32 units (reset and update gates)</li>
              <li>Dense Feature Layer: 16 units, ReLU activation</li>
              <li>Output Regression Layer: 1 unit (predicted next-hour kW)</li>
              <li>Optimizer: Adam (learning_rate = 0.001, beta_1=0.9)</li>
              <li>Loss Function: Mean Squared Error (MSE)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
