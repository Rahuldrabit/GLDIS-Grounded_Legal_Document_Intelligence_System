import React, { useState } from 'react';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { DocumentDetail } from './pages/DocumentDetail';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [params, setParams] = useState<any>({});

  const handleNavigate = (page: string, newParams?: any) => {
    setCurrentPage(page);
    setParams(newParams || {});
  };

  return (
    <Layout onNavigate={handleNavigate}>
      {currentPage === 'dashboard' && <Dashboard onNavigate={handleNavigate} />}
      {currentPage === 'document' && params.id && (
        <DocumentDetail documentId={params.id} onNavigate={handleNavigate} />
      )}
    </Layout>
  );
}

export default App;
