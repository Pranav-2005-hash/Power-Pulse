import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { Analytics } from './pages/Analytics';
import { Forecast } from './pages/Forecast';
import { Anomalies } from './pages/Anomalies';
import { Models } from './pages/Models';
import { Dataset } from './pages/Dataset';
import { Reports } from './pages/Reports';
import { About } from './pages/About';
import { api } from './services/api';
import { DatasetStatus } from './types';

export const App: React.FC = () => {
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState<boolean>(false);

  useEffect(() => {
    checkHealthAndStatus();
    // Heartbeat check every 10 seconds
    const interval = setInterval(checkHealthAndStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const checkHealthAndStatus = async () => {
    try {
      await api.checkHealth();
      setBackendOnline(true);
      const status = await api.getDatasetStatus();
      setDatasetStatus(status);
    } catch (err) {
      setBackendOnline(false);
    }
  };

  const handleGlobalRefresh = async () => {
    setIsRefreshing(true);
    await checkHealthAndStatus();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 flex">
        {/* Navy Sidebar */}
        <Sidebar
          datasetStatus={datasetStatus}
          backendOnline={backendOnline}
          isOpen={mobileSidebarOpen}
          onCloseMobile={() => setMobileSidebarOpen(false)}
        />

        {/* Main Content Area */}
        <div className="flex-1 lg:pl-64 flex flex-col min-w-0">
          <Navbar
            onOpenMobile={() => setMobileSidebarOpen(true)}
            backendOnline={backendOnline}
            onRefreshAll={handleGlobalRefresh}
            isRefreshing={isRefreshing}
          />

          <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
            <Routes>
              <Route path="/" element={<Dashboard datasetStatus={datasetStatus} onRefresh={handleGlobalRefresh} />} />
              <Route path="/analytics" element={<Analytics datasetStatus={datasetStatus} />} />
              <Route path="/forecast" element={<Forecast datasetStatus={datasetStatus} />} />
              <Route path="/anomalies" element={<Anomalies datasetStatus={datasetStatus} />} />
              <Route path="/models" element={<Models datasetStatus={datasetStatus} onRefresh={handleGlobalRefresh} />} />
              <Route path="/dataset" element={<Dataset datasetStatus={datasetStatus} onRefresh={handleGlobalRefresh} />} />
              <Route path="/reports" element={<Reports datasetStatus={datasetStatus} />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
};

export default App;
