import React, { useEffect, useState } from 'react';
import { api, DocumentDetailResponse, DraftResponse } from '../api/client';
import './DocumentDetail.css';

interface DocumentDetailProps {
  documentId: string;
  onNavigate: (page: string, params?: any) => void;
}

export const DocumentDetail: React.FC<DocumentDetailProps> = ({ documentId, onNavigate }) => {
  const [doc, setDoc] = useState<DocumentDetailResponse | null>(null);
  const [draft, setDraft] = useState<DraftResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchDetail = async () => {
      try {
        const data = await api.getDocument(documentId);
        setDoc(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
  }, [documentId]);

  const handleGenerateDraft = async () => {
    setGenerating(true);
    setError('');
    try {
      const data = await api.generateDraft(documentId);
      setDraft(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setGenerating(false);
    }
  };

  if (loading) return <div className="loading">Loading document...</div>;
  if (!doc) return <div className="error-message">Document not found {error}</div>;

  return (
    <div className="document-detail animate-fade-in">
      <button className="back-btn" onClick={() => onNavigate('dashboard')}>
        ← Back to Dashboard
      </button>

      <header className="detail-header glass-panel">
        <div className="header-info">
          <h2>{doc.filename}</h2>
          <span className={`badge badge-${doc.status.toLowerCase()}`}>{doc.status}</span>
        </div>
        <div className="meta-stats">
          <div className="stat"><span>Pages:</span> {doc.page_count}</div>
          <div className="stat"><span>Chunks:</span> {doc.chunks_count}</div>
        </div>
      </header>

      {error && <div className="error-message">{error}</div>}

      <div className="content-grid">
        <section className="extracted-fields glass-panel">
          <h3>Extracted Fields</h3>
          {doc.structured_fields.length === 0 ? (
            <p className="empty-text">No fields extracted yet. Is the document processed?</p>
          ) : (
            <div className="fields-list">
              {doc.structured_fields.map((field, idx) => (
                <div key={idx} className="field-item">
                  <div className="field-name">{field.field}</div>
                  <div className="field-value">{String(field.value)}</div>
                  <div className="field-conf">{(field.confidence * 100).toFixed(0)}%</div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="draft-generation glass-panel">
          <h3>Draft Generation</h3>
          {doc.status !== 'ready' ? (
            <p className="empty-text">Document must be "ready" to generate a draft.</p>
          ) : (
            <div className="draft-actions">
              <button 
                className="btn btn-primary" 
                onClick={handleGenerateDraft}
                disabled={generating}
              >
                {generating ? 'Generating Draft...' : 'Generate Case Fact Summary'}
              </button>
            </div>
          )}

          {draft && (
            <div className="draft-result animate-fade-in">
              <h4>Generated Draft</h4>
              <div className="score">Grounding Score: {(draft.grounding_score * 100).toFixed(1)}%</div>
              <div className="draft-text">
                {draft.generated_text}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};
