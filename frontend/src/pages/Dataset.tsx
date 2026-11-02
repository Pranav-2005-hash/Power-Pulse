import React, { useState, useEffect } from 'react';
import {
  Database, Upload, RefreshCw, CheckCircle2, AlertCircle,
  FileText, Clock, Zap, ArrowRight, ChevronLeft, ChevronRight, ShieldCheck
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetStatus, DatasetPreview } from '../types';
import { EmptyState, LoadingState } from '../components/EmptyState';

interface DatasetProps {
  datasetStatus: DatasetStatus | null;
  onRefresh: () => void;
}

export const Dataset: React.FC<DatasetProps> = ({ datasetStatus, onRefresh }) => {
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [page, setPage] = useState<number>(1);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Upload fields
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [timestampCol, setTimestampCol] = useState('');
  const [consumptionCol, setConsumptionCol] = useState('');

  useEffect(() => {
    if (datasetStatus?.is_loaded) {
      loadPreview(page);
    }
  }, [datasetStatus?.is_loaded, page]);

  const loadPreview = async (p: number) => {
    setLoading(true);
    try {
      const data = await api.getDatasetPreview(p, 50);
      setPreview(data);
    } catch (e: any) {
      console.error('Error fetching preview:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    setError(null);
    setSuccessMsg(null);
    setLoading(true);
    try {
      await api.loadSampleDataset();
      setSuccessMsg("UCI Individual Household Power Consumption sample dataset loaded successfully.");
      onRefresh();
      loadPreview(1);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to load sample dataset.");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError("Please select a valid CSV or TXT file to upload.");
      return;
    }

    setError(null);
    setSuccessMsg(null);
    setUploading(true);

    try {
      const res = await api.uploadDataset(
        selectedFile,
        timestampCol || undefined,
        consumptionCol || undefined
      );
      setSuccessMsg(
        `Uploaded '${res.filename}': ${res.raw_rows.toLocaleString()} raw rows parsed, ` +
        `${res.hourly_rows.toLocaleString()} hourly records generated. ` +
        `Duplicate timestamps handled: ${res.duplicate_timestamps}.`
      );
      setSelectedFile(null);
      onRefresh();
      loadPreview(1);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Validation failed on uploaded CSV file.");
    } finally {
      setUploading(false);
    }
  };

  const totalPages = preview ? Math.ceil(preview.total_rows / preview.page_size) : 1;

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-navy-950 flex items-center gap-2">
            <Database className="w-5 h-5 text-emerald-500" />
            <span>Dataset Ingestion & Data Hygiene Lab</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Conservative minute-to-hourly resampling, gap interpolation, and strict chronological partition
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleLoadSample}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg shadow-sm transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Load / Reset Sample Dataset</span>
          </button>
        </div>
      </div>

      {/* Notifications */}
      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-xl text-sm flex items-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-600" />
          <span>{error}</span>
        </div>
      )}

      {successMsg && (
        <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 p-4 rounded-xl text-sm flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 shrink-0 text-emerald-600" />
          <span>{successMsg}</span>
        </div>
      )}

      {/* Dataset Status Summary Grid */}
      {datasetStatus?.is_loaded && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">Active Source</span>
            <span className="text-sm font-bold text-navy-950 truncate block mt-1 capitalize">
              {datasetStatus.source_type}
            </span>
            <span className="text-[10px] text-slate-400 truncate block">{datasetStatus.filename}</span>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">Raw Records</span>
            <span className="text-sm font-bold text-navy-950 block mt-1 font-mono">
              {datasetStatus.total_raw_rows.toLocaleString()}
            </span>
            <span className="text-[10px] text-slate-400 block">Sub-hourly readings</span>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">Hourly Rows</span>
            <span className="text-sm font-bold text-emerald-600 block mt-1 font-mono">
              {datasetStatus.total_hourly_rows.toLocaleString()}
            </span>
            <span className="text-[10px] text-slate-400 block">Resampled mean</span>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">Observation Span</span>
            <span className="text-xs font-bold text-navy-950 block mt-1">
              {datasetStatus.start_date?.slice(0, 10)}
            </span>
            <span className="text-[10px] text-slate-400 block">to {datasetStatus.end_date?.slice(0, 10)}</span>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">Missing Handled</span>
            <span className="text-sm font-bold text-navy-950 block mt-1 font-mono">
              {datasetStatus.missing_values_handled}
            </span>
            <span className="text-[10px] text-slate-400 block">Interpolated &le; 2h</span>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">Model State</span>
            <span className={`text-xs font-bold block mt-1 flex items-center gap-1 ${
              datasetStatus.models_trained ? 'text-emerald-600' : 'text-amber-600'
            }`}>
              {datasetStatus.models_trained ? '✓ Compatible' : 'Re-train Needed'}
            </span>
            <span className="text-[10px] text-slate-400 block">
              {datasetStatus.models_trained ? 'Weights valid' : 'New data pending'}
            </span>
          </div>
        </div>
      )}

      {/* Upload Custom CSV Form */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
        <h2 className="text-base font-bold text-navy-950 flex items-center gap-2 mb-1">
          <Upload className="w-4 h-4 text-slate-700" />
          <span>Upload Custom Household Consumption CSV</span>
        </h2>
        <p className="text-xs text-slate-500 mb-6">
          Upload your own power consumption dataset. The pipeline validates timestamp sequences, identifies gaps, converts invalid markers (&apos;?&apos;), and resamples to hourly mean.
        </p>

        <form onSubmit={handleUpload} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* File Input */}
            <div className="md:col-span-1">
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                CSV or TXT Data File <span className="text-rose-500">*</span>
              </label>
              <input
                type="file"
                accept=".csv,.txt"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                className="w-full text-xs text-slate-500 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-100 file:text-slate-700 hover:file:bg-slate-200 border border-slate-200 rounded-lg p-1 focus:outline-none"
              />
            </div>

            {/* Custom Timestamp Column */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Timestamp Column <span className="text-slate-400 font-normal">(Optional)</span>
              </label>
              <input
                type="text"
                placeholder="e.g., Date, timestamp, datetime"
                value={timestampCol}
                onChange={(e) => setTimestampCol(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
            </div>

            {/* Custom Consumption Column */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Consumption Column <span className="text-slate-400 font-normal">(Optional)</span>
              </label>
              <input
                type="text"
                placeholder="e.g., Global_active_power, power, kw"
                value={consumptionCol}
                onChange={(e) => setConsumptionCol(e.target.value)}
                className="w-full px-3 py-2 text-xs border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
            </div>
          </div>

          <div className="flex items-center justify-between pt-2">
            <span className="text-[11px] text-slate-400">
              * Note: Uploading a new dataset invalidates previously trained models to prevent shape or scale mismatch.
            </span>
            <button
              type="submit"
              disabled={uploading || !selectedFile}
              className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-white bg-navy-900 hover:bg-navy-800 rounded-lg shadow-sm transition-all disabled:opacity-50"
            >
              {uploading ? (
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Upload className="w-3.5 h-3.5" />
              )}
              <span>{uploading ? 'Validating & Resampling...' : 'Upload & Process Dataset'}</span>
            </button>
          </div>
        </form>
      </div>

      {/* Dataset Preview Table */}
      {preview && (
        <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm overflow-hidden">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
            <div>
              <h2 className="text-base font-bold text-navy-950">Cleaned Hourly Dataset Preview</h2>
              <p className="text-xs text-slate-500">
                Displaying page {page} of {totalPages} ({preview.total_rows.toLocaleString()} total hourly records)
              </p>
            </div>

            {/* Pagination Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page <= 1 || loading}
                className="p-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40"
              >
                <ChevronLeft className="w-4 h-4 text-slate-600" />
              </button>
              <span className="text-xs font-medium text-slate-700 font-mono px-2">
                Page {page} / {totalPages}
              </span>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages || loading}
                className="p-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40"
              >
                <ChevronRight className="w-4 h-4 text-slate-600" />
              </button>
            </div>
          </div>

          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-slate-50 border-b border-slate-200 sticky top-0 z-10">
                <tr>
                  <th className="py-2.5 px-4 font-semibold text-slate-600">Row #</th>
                  <th className="py-2.5 px-4 font-semibold text-slate-600">Timestamp (1h interval)</th>
                  <th className="py-2.5 px-4 font-semibold text-slate-600">Global_active_power (kW)</th>
                  <th className="py-2.5 px-4 font-semibold text-slate-600">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {preview.rows.map((row, idx) => {
                  const globalIdx = (page - 1) * preview.page_size + idx + 1;
                  return (
                    <tr key={row.timestamp} className="hover:bg-slate-50">
                      <td className="py-2 px-4 text-slate-400">{globalIdx}</td>
                      <td className="py-2 px-4 text-slate-900 font-sans">{row.timestamp}</td>
                      <td className="py-2 px-4 font-bold text-emerald-600">
                        {row.Global_active_power} kW
                      </td>
                      <td className="py-2 px-4 text-[11px] font-sans text-emerald-700">
                        <span className="inline-flex items-center gap-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                          Valid
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
