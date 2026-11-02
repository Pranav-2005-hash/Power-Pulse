import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, LineChart, TrendingUp, AlertTriangle,
  Cpu, Database, FileText, Info, Zap, CheckCircle2, AlertCircle
} from 'lucide-react';
import { DatasetStatus } from '../types';

interface SidebarProps {
  datasetStatus: DatasetStatus | null;
  backendOnline: boolean;
  isOpen: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  datasetStatus,
  backendOnline,
  isOpen,
  onCloseMobile
}) => {
  const navItems = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/analytics', label: 'Analytics', icon: LineChart },
    { to: '/forecast', label: 'DL Forecast', icon: TrendingUp },
    { to: '/anomalies', label: 'Anomalies', icon: AlertTriangle },
    { to: '/models', label: 'Models Lab', icon: Cpu },
    { to: '/dataset', label: 'Dataset', icon: Database },
    { to: '/reports', label: 'Reports', icon: FileText },
    { to: '/about', label: 'Documentation', icon: Info },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-navy-950/70 backdrop-blur-sm z-40 lg:hidden"
          onClick={onCloseMobile}
        />
      )}

      <aside className={`
        fixed top-0 bottom-0 left-0 z-50 w-64 bg-navy-950 border-r border-navy-800/80 text-slate-300
        flex flex-col transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        {/* Brand Header */}
        <div className="h-16 flex items-center px-6 border-b border-navy-800/80 bg-navy-900/60">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-emerald-500 to-cyan-400 flex items-center justify-center shadow-glow-green text-navy-950">
              <Zap className="w-5 h-5 fill-current" />
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight text-white flex items-center">
                Power<span className="text-emerald-400 font-extrabold">Pulse</span>
              </span>
              <p className="text-[10px] uppercase tracking-wider text-slate-400 font-medium">
                Deep Learning Energy AI
              </p>
            </div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onCloseMobile}
              className={({ isActive }) => `
                flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all
                ${isActive 
                  ? 'bg-gradient-to-r from-emerald-500/20 to-cyan-500/10 text-emerald-400 border border-emerald-500/30 shadow-sm' 
                  : 'text-slate-400 hover:text-slate-100 hover:bg-navy-900/70'}
              `}
            >
              <item.icon className="w-4 h-4 shrink-0" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Pipeline & System Status Card */}
        <div className="p-4 border-t border-navy-800/80 bg-navy-900/50 m-3 rounded-xl border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] uppercase tracking-wider font-semibold text-slate-400">
              Pipeline Status
            </span>
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
              <span className="text-xs text-slate-300">
                {backendOnline ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>

          <div className="space-y-1.5 text-xs text-slate-400">
            <div className="flex justify-between items-center py-0.5">
              <span>Dataset:</span>
              <span className={`font-medium flex items-center gap-1 ${datasetStatus?.is_loaded ? 'text-emerald-400' : 'text-amber-400'}`}>
                {datasetStatus?.is_loaded ? (
                  <><CheckCircle2 className="w-3 h-3" /> Ready</>
                ) : (
                  <><AlertCircle className="w-3 h-3" /> None</>
                )}
              </span>
            </div>

            <div className="flex justify-between items-center py-0.5">
              <span>DL Models:</span>
              <span className={`font-medium flex items-center gap-1 ${datasetStatus?.models_trained ? 'text-cyan-400' : 'text-amber-400'}`}>
                {datasetStatus?.models_trained ? (
                  <><CheckCircle2 className="w-3 h-3" /> Trained</>
                ) : (
                  <><AlertCircle className="w-3 h-3" /> Untrained</>
                )}
              </span>
            </div>

            <div className="flex justify-between items-center py-0.5">
              <span>Target:</span>
              <span className="font-mono text-[11px] text-slate-200">Global_active_power</span>
            </div>
          </div>
        </div>

        {/* Footer info */}
        <div className="px-6 py-3 border-t border-navy-800/60 text-[11px] text-slate-500 flex justify-between items-center">
          <span>College DL Project</span>
          <span className="font-mono text-[10px]">v1.0.0</span>
        </div>
      </aside>
    </>
  );
};
