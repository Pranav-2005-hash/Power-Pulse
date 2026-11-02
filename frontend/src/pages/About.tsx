import React from 'react';
import {
  Info, Cpu, Database, CheckCircle2, AlertTriangle,
  Layers, GitBranch, ShieldCheck, BookOpen, Sparkles
} from 'lucide-react';

export const About: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      {/* Header */}
      <div className="bg-gradient-to-r from-navy-950 via-navy-900 to-navy-850 rounded-2xl p-8 text-white border border-navy-800 shadow-xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold mb-3">
          <BookOpen className="w-3.5 h-3.5" />
          Academic Deep Learning Project Specification
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight">
          PowerPulse: AI-Powered Energy Forecasting & Anomaly Detection
        </h1>
        <p className="text-sm text-slate-300 mt-2 max-w-3xl leading-relaxed">
          A full-stack deep learning solution for household electricity load prediction and residual-based anomaly detection using recurrent neural network architectures (LSTM &amp; GRU).
        </p>
      </div>

      {/* 1. Problem Statement & Objectives */}
      <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-sm space-y-6">
        <div>
          <h2 className="text-lg font-bold text-navy-950 flex items-center gap-2 mb-3">
            <span className="w-2.5 h-6 rounded-full bg-emerald-500 inline-block" />
            <span>1. Problem Statement &amp; Motivation</span>
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            Modern smart electrical grids and household energy management systems require accurate, real-time load forecasting to balance grid supply, optimize decentralized storage, and identify atypical consumption surges. Unpredicted peaks cause grid stress, inflated consumer tariffs, and undetected equipment anomalies. This project evaluates recurrent deep neural network architectures to predict hourly electrical load (in kilowatt units) and detect statistical surges using rigorous chronological evaluation.
          </p>
        </div>

        <div>
          <h3 className="text-sm font-bold text-navy-950 uppercase tracking-wider text-slate-500 mb-3">
            Core Project Objectives
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold text-navy-950">Multi-Step Recursive Forecasting</h4>
                <p className="text-xs text-slate-500 mt-1">
                  Predict 1, 6, 12, 24, and 48 hours into the future using 24-hour historical windows without fabricating future readings.
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold text-navy-950">Zero Data Leakage Pipeline</h4>
                <p className="text-xs text-slate-500 mt-1">
                  Strict 70% Train, 15% Validation, 15% Test chronological split. MinMaxScaler is fitted strictly on the training partition only.
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold text-navy-950">Validation-Derived Anomaly Detection</h4>
                <p className="text-xs text-slate-500 mt-1">
                  Thresholds (90th, 95th, 99th percentiles) derived exclusively from positive validation residuals to avoid overfitting the test set.
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold text-navy-950">Objective Model Selection</h4>
                <p className="text-xs text-slate-500 mt-1">
                  Best model selected strictly using validation MAE in real kW units rather than biased test metrics or hardcoded claims.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Deep Learning Recurrent Architectures (LSTM vs GRU) */}
      <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-sm space-y-6">
        <h2 className="text-lg font-bold text-navy-950 flex items-center gap-2">
          <span className="w-2.5 h-6 rounded-full bg-cyan-500 inline-block" />
          <span>2. Deep Learning Architectures: LSTM vs GRU</span>
        </h2>
        <p className="text-sm text-slate-600 leading-relaxed">
          Standard Feedforward Neural Networks cannot maintain temporal state across sequential time intervals. Recurrent Neural Networks (RNNs) address this through internal hidden state feedback loops, but suffer from vanishing and exploding gradients over extended sequences. PowerPulse implements two gated architectures to overcome this limitation:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* LSTM Blueprint */}
          <div className="p-5 rounded-xl border border-slate-200 bg-slate-50/50">
            <div className="flex items-center gap-2 mb-3">
              <span className="px-2.5 py-1 rounded-md bg-emerald-100 text-emerald-800 font-mono font-bold text-xs">
                LSTM
              </span>
              <h3 className="text-sm font-bold text-navy-950">Long Short-Term Memory</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed mb-4">
              Introduces an explicit cell state vector <span className="font-mono text-slate-800">C_t</span> regulated by three distinct gating mechanisms:
            </p>
            <ul className="space-y-2 text-xs text-slate-600 list-disc list-inside">
              <li><strong>Forget Gate (f_t):</strong> Decides what information from previous cell state to discard via sigmoid function.</li>
              <li><strong>Input Gate (i_t):</strong> Regulates which new candidate values are written into the cell state.</li>
              <li><strong>Output Gate (o_t):</strong> Computes the next hidden state <span className="font-mono">h_t</span> based on filtered cell state memory.</li>
            </ul>
            <div className="mt-4 pt-3 border-t border-slate-200 text-xs font-mono text-slate-500">
              Param Count: 4 * [(input_dim + 1) * units + units^2] &asymp; 4,849 params
            </div>
          </div>

          {/* GRU Blueprint */}
          <div className="p-5 rounded-xl border border-slate-200 bg-slate-50/50">
            <div className="flex items-center gap-2 mb-3">
              <span className="px-2.5 py-1 rounded-md bg-cyan-100 text-cyan-800 font-mono font-bold text-xs">
                GRU
              </span>
              <h3 className="text-sm font-bold text-navy-950">Gated Recurrent Unit</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed mb-4">
              A streamlined recurrent variant proposed by Cho et al. (2014) merging cell state and hidden state:
            </p>
            <ul className="space-y-2 text-xs text-slate-600 list-disc list-inside">
              <li><strong>Reset Gate (r_t):</strong> Controls how much of the previous hidden state memory to forget when computing new candidate state.</li>
              <li><strong>Update Gate (z_t):</strong> Acts simultaneously as both forget and input gates, balancing old vs new state.</li>
              <li><strong>Efficiency:</strong> Faster training execution and lower latency due to having 25% fewer weight matrices than LSTM.</li>
            </ul>
            <div className="mt-4 pt-3 border-t border-slate-200 text-xs font-mono text-slate-500">
              Param Count: 3 * [(input_dim + 1) * units + units^2] &asymp; 3,809 params
            </div>
          </div>
        </div>
      </div>

      {/* 3. System Architecture & End-to-End Flow */}
      <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-navy-950 flex items-center gap-2">
          <span className="w-2.5 h-6 rounded-full bg-navy-900 inline-block" />
          <span>3. End-to-End System Pipeline</span>
        </h2>
        
        <div className="bg-navy-950 text-slate-300 p-6 rounded-xl font-mono text-xs overflow-x-auto leading-relaxed border border-navy-800">
          <pre>{`[Raw Data Ingestion]
        │ (UCI Household Electric Power CSV: Date, Time, Global_active_power)
        ▼
[Preprocessing & Cleaning]
        │ Parse timestamps, convert '?' to NaN, conservative gap check (<=2h)
        ▼
[Hourly Mean Resampling]
        │ Aggregates minute recordings to continuous 1-hour interval mean
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
[FastAPI REST Backend] <──── HTTP/JSON ────> [React + Vite + Tailwind UI]`}</pre>
        </div>
      </div>

      {/* 4. Dataset Citation */}
      <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-sm space-y-3">
        <h2 className="text-lg font-bold text-navy-950 flex items-center gap-2">
          <span className="w-2.5 h-6 rounded-full bg-amber-500 inline-block" />
          <span>4. Dataset Citation &amp; Benchmark Reference</span>
        </h2>
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed font-mono">
          <p className="font-bold text-navy-950 mb-1">
            UCI Machine Learning Repository: Individual Household Electric Power Consumption Dataset
          </p>
          <p>
            Creators: Georges Hebrail (EDF R&amp;D), Alice Berard (EDF R&amp;D).
          </p>
          <p className="text-slate-500 mt-1">
            Source URL: <a href="https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption" target="_blank" rel="noreferrer" className="text-emerald-600 underline">https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption</a>
          </p>
          <p className="text-slate-500 mt-1">
            Target Feature: <strong>Global_active_power</strong> (household global minute-averaged active power in kilowatt kW).
          </p>
        </div>
      </div>

      {/* 5. Honest Limitations & Future Work */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-sm space-y-3">
          <h3 className="text-base font-bold text-navy-950 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            <span>Honest Project Limitations</span>
          </h3>
          <ul className="space-y-2 text-xs text-slate-600 leading-relaxed list-disc list-inside">
            <li>
              <strong>Absence of Exogenous Weather Variables:</strong> The current model relies purely on autoregressive lag windows without ambient temperature, solar irradiation, or humidity features.
            </li>
            <li>
              <strong>Recursive Multi-Step Error Accumulation:</strong> Over long horizons (e.g. 48h), feeding predicted outputs recursively can cause errors to compound over time.
            </li>
            <li>
              <strong>Stationarity Assumptions:</strong> Long-term behavioral changes (such as extended resident vacations or new HVAC installations) can alter underlying consumption baselines.
            </li>
          </ul>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-sm space-y-3">
          <h3 className="text-base font-bold text-navy-950 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-500" />
            <span>Future Research Directions</span>
          </h3>
          <ul className="space-y-2 text-xs text-slate-600 leading-relaxed list-disc list-inside">
            <li>
              <strong>Multi-Variate Exogenous Fusion:</strong> Integrating weather API forecasts and calendar embeddings (holidays, daylight saving changes).
            </li>
            <li>
              <strong>Transformer / Informer Models:</strong> Evaluating Temporal Fusion Transformers (TFT) with self-attention for direct multi-step quantile forecasting.
            </li>
            <li>
              <strong>Edge Microcontroller Deployment:</strong> Quantizing TensorFlow models into TensorFlow Lite (TFLite) for on-device inference on smart meter hardware.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
