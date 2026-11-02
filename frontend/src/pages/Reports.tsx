import React, { useState } from 'react';
import {
  FileText, Download, Eye, ExternalLink, CheckCircle2,
  AlertCircle, Table, FileSpreadsheet, Code2, Globe
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetStatus } from '../types';

interface ReportsProps {
  datasetStatus: DatasetStatus | null;
}

export const Reports: React.FC<ReportsProps> = ({ datasetStatus }) => {
  const [showPreviewModal, setShowPreviewModal] = useState(false);

  const reportItems = [
    {
      title: "Consolidated Executive HTML Report",
      filename: "powerpulse_summary.html",
      format: "HTML / Printable PDF",
      icon: Globe,
      color: "emerald",
      endpoint: api.getHtmlReportUrl(),
      description: "Complete formal evaluation document featuring model architectures, validation thresholds, comparative metrics, and academic citations.",
      requiresModels: true,
      canPreview: true,
    },
    {
      title: "Model Performance Comparison Matrix",
      filename: "powerpulse_model_comparison.csv",
      format: "CSV",
      icon: FileSpreadsheet,
      color: "cyan",
      endpoint: api.getModelComparisonCsvUrl(),
      description: "Tabular evaluation metrics: validation MAE, validation RMSE, test MAE, test RMSE, R² score, parameter counts, and training durations.",
      requiresModels: true,
      canPreview: false,
    },
    {
      title: "Recursive Load Forecast Series",
      filename: "powerpulse_forecast.csv",
      format: "CSV",
      icon: Table,
      color: "emerald",
      endpoint: api.getForecastCsvUrl(),
      description: "Detailed multi-step future consumption predictions with timestamps, step indices, and 95% confidence intervals.",
      requiresModels: true,
      canPreview: false,
    },
    {
      title: "Flagged Anomaly Incident Register",
      filename: "powerpulse_anomalies.csv",
      format: "CSV",
      icon: FileSpreadsheet,
      color: "amber",
      endpoint: api.getAnomaliesCsvUrl(),
      description: "Historical anomaly occurrences where positive residuals exceeded validation thresholds, including severity rankings and review states.",
      requiresModels: true,
      canPreview: false,
    },
    {
      title: "Cleaned Hourly Consumption Time-Series",
      filename: "powerpulse_hourly_consumption.csv",
      format: "CSV",
      icon: Table,
      color: "blue",
      endpoint: api.getProcessedDataCsvUrl(),
      description: "Resampled hourly mean dataset ready for external econometric, statistical, or machine learning pipelines.",
      requiresModels: false,
      canPreview: false,
    },
    {
      title: "Full Machine Learning Evaluation Payload",
      filename: "powerpulse_evaluation.json",
      format: "JSON",
      icon: Code2,
      color: "purple",
      endpoint: api.getEvaluationJsonUrl(),
      description: "Programmatic JSON structure containing model hyperparameters, layer definitions, data split metadata, and residual distributions.",
      requiresModels: true,
      canPreview: false,
    },
  ];

  const modelsReady = !!datasetStatus?.models_trained;

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-navy-950 flex items-center gap-2">
            <FileText className="w-5 h-5 text-emerald-500" />
            <span>Reports & Artifact Export Center</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Download verified machine learning artifacts, CSV matrices, JSON telemetry, and executive reports
          </p>
        </div>

        <button
          onClick={() => window.open(api.getHtmlReportUrl(), '_blank')}
          disabled={!modelsReady}
          className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-white bg-navy-900 hover:bg-navy-800 rounded-lg shadow-sm transition-all disabled:opacity-50"
        >
          <Globe className="w-3.5 h-3.5 text-cyan-400" />
          <span>Open Executive Summary (HTML)</span>
        </button>
      </div>

      {!modelsReady && (
        <div className="bg-amber-50 border border-amber-200 text-amber-800 p-4 rounded-xl text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-amber-600" />
          <span>
            Some analytical reports require trained deep learning models. Train LSTM/GRU models on the Models page to unlock all files.
          </span>
        </div>
      )}

      {/* Reports Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {reportItems.map((item) => {
          const isDisabled = item.requiresModels && !modelsReady;
          return (
            <div
              key={item.filename}
              className={`
                bg-white rounded-xl border p-5 shadow-sm flex flex-col justify-between transition-all
                ${isDisabled ? 'opacity-60 border-slate-200' : 'border-slate-200/80 hover:shadow-md'}
              `}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-slate-100 text-slate-700">
                    {item.format}
                  </span>
                  <div className="w-8 h-8 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-center text-slate-600">
                    <item.icon className="w-4 h-4" />
                  </div>
                </div>

                <h3 className="text-sm font-bold text-navy-950 mb-1.5">{item.title}</h3>
                <p className="text-xs text-slate-500 leading-relaxed mb-4">{item.description}</p>
              </div>

              <div className="pt-4 border-t border-slate-100 flex items-center gap-2">
                <a
                  href={isDisabled ? '#' : item.endpoint}
                  download={item.filename}
                  onClick={(e) => {
                    if (isDisabled) e.preventDefault();
                  }}
                  className={`
                    flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all
                    ${isDisabled 
                      ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
                      : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm'}
                  `}
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download {item.format}</span>
                </a>

                {item.canPreview && (
                  <button
                    onClick={() => setShowPreviewModal(true)}
                    disabled={isDisabled}
                    className="p-2 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-600 disabled:opacity-40"
                    title="Live Preview"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* HTML Report In-Page Preview Modal */}
      {showPreviewModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-navy-950/70 backdrop-blur-sm p-4">
          <div className="bg-white rounded-2xl w-full max-w-4xl h-[85vh] flex flex-col shadow-2xl border border-slate-200 overflow-hidden">
            <div className="h-14 px-6 border-b border-slate-200 flex items-center justify-between bg-slate-50">
              <span className="text-sm font-bold text-navy-950">Executive Report Live Preview</span>
              <div className="flex items-center gap-3">
                <a
                  href={api.getHtmlReportUrl()}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1.5 text-xs text-emerald-600 font-semibold hover:underline"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  <span>Open in Tab</span>
                </a>
                <button
                  onClick={() => setShowPreviewModal(false)}
                  className="px-3 py-1 text-xs font-bold text-slate-600 hover:text-slate-900 bg-white border border-slate-200 rounded-lg"
                >
                  Close
                </button>
              </div>
            </div>

            <iframe
              src={api.getHtmlReportUrl()}
              title="Report Preview"
              className="w-full flex-1 border-0"
            />
          </div>
        </div>
      )}
    </div>
  );
};
