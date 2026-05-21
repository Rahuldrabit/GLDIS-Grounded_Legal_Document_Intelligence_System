import React, { useEffect, useState } from 'react';
import { api, DocumentListResponse } from '../api/client';
import './Dashboard.css';

interface DashboardProps {
  onNavigate: (page: string, params?: any) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const [documents, setDocuments] = useState<DocumentListResponse[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  const fetchDocuments = async () => {
    try {
      const docs = await api.listDocuments();
      setDocuments(docs);
    } catch (err: any) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchDocuments();
    // Poll every 5 seconds for status updates
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError('');
    try {
      await api.uploadDocument(file);
      fetchDocuments();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleProcess = async (e: React.MouseEvent, docId: string) => {
    e.stopPropagation();
    try {
      await api.processDocumentSync(docId);
      fetchDocuments();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleDelete = async (e: React.MouseEvent, docId: string) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this document?')) return;
    try {
      await api.deleteDocument(docId);
      fetchDocuments();
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Your Documents</h2>
        <div className="upload-container">
          <input
            type="file"
            id="file-upload"
            className="file-input"
            onChange={handleFileUpload}
            disabled={uploading}
            accept=".pdf,.png,.jpg,.jpeg,.tiff,.txt"
          />
          <label htmlFor="file-upload" className="btn btn-primary hover-lift">
            {uploading ? 'Uploading...' : 'Upload Document'}
          </label>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="document-grid">
        {documents.map((doc) => (
          <div 
            key={doc.document_id} 
            className="document-card glass-panel hover-lift"
            onClick={() => onNavigate('document', { id: doc.document_id })}
          >
            <div className="doc-icon">📄</div>
            <div className="doc-info">
              <h3>{doc.filename}</h3>
              <div className="doc-meta">
                <span className={`badge badge-${doc.status.toLowerCase()}`}>
                  {doc.status}
                </span>
                <span>{new Date(doc.upload_time).toLocaleDateString()}</span>
              </div>
            </div>
            <div className="doc-actions">
              {doc.status === 'uploaded' && (
                <button 
                  className="btn btn-secondary btn-sm"
                  onClick={(e) => handleProcess(e, doc.document_id)}
                >
                  Process
                </button>
              )}
              <button 
                className="btn btn-secondary btn-sm delete-btn"
                onClick={(e) => handleDelete(e, doc.document_id)}
              >
                🗑️
              </button>
            </div>
          </div>
        ))}
        {documents.length === 0 && !uploading && (
          <div className="empty-state">
            <p>No documents uploaded yet. Upload a document to get started.</p>
          </div>
        )}
      </div>
    </div>
  );
};
