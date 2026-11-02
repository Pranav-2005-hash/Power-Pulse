import React from 'react';
import { Menu, Zap, RefreshCw, Cpu, Database } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface NavbarProps {
  onOpenMobile: () => void;
  backendOnline: boolean;
  onRefreshAll: () => void;
  isRefreshing: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  onOpenMobile,
  backendOnline,
  onRefreshAll,
  isRefreshing,
}) => {
  const navigate = useNavigate();

  return (
    <header className="sticky top-0 z-30 h-16 bg-white/90 backdrop-blur-md border-b border-slate-200/80 px-4 sm:px-6 lg:px-8 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobile}
          className="p-2 -ml-2 text-slate-600 hover:text-slate-900 rounded-lg lg:hidden hover:bg-slate-100"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2">
          <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
            <Zap className="w-3.5 h-3.5 text-emerald-500 fill-emerald-500" />
            UCI Household Power Pipeline
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        {/* Backend health pill */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
          backendOnline 
            ? 'bg-slate-50 text-slate-700 border-slate-200' 
            : 'bg-rose-50 text-rose-700 border-rose-200'
        }`}>
          <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          <span className="hidden md:inline">API:</span>
          <span>{backendOnline ? '127.0.0.1:8000' : 'Disconnected'}</span>
        </div>

        {/* Global Refresh Button */}
        <button
          onClick={onRefreshAll}
          disabled={isRefreshing}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg shadow-sm transition-all disabled:opacity-50"
          title="Refresh real data from backend"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-emerald-600' : ''}`} />
          <span className="hidden sm:inline">Sync</span>
        </button>

        {/* Quick action: Dataset */}
        <button
          onClick={() => navigate('/dataset')}
          className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200/80 rounded-lg transition-all"
        >
          <Database className="w-3.5 h-3.5 text-slate-500" />
          <span>Data Ingestion</span>
        </button>

        {/* Quick action: Models Lab */}
        <button
          onClick={() => navigate('/models')}
          className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-white bg-navy-900 hover:bg-navy-800 rounded-lg shadow-sm hover:shadow transition-all"
        >
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>Train Models</span>
        </button>
      </div>
    </header>
  );
};
