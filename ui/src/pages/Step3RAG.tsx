import React, { useState } from 'react';
import { api } from '../api/client';
import './Step3RAG.css';

interface EvidenceChunk {
  chunk_id: string;
  page: number;
  text: string;
  score: number;
}

interface Step3RAGProps {
  documentText: string;
  onBack: () => void;
}

export const Step3RAG: React.FC<Step3RAGProps> = ({ documentText, onBack }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [drafting, setDrafting] = useState(false);
  const [chatHistory, setChatHistory] = useState<Array<{role: string, content: string, evidence?: EvidenceChunk[]}>>([]);
  
  const handleQuery = async (presetQuery?: string) => {
    const q = presetQuery || query;
    if (!q) return;

    setLoading(true);
    setChatHistory(prev => [...prev, { role: 'user', content: q }]);
    
    try {
      const res = await api.statelessQuery(documentText, q);
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: res.generated_text,
        evidence: res.evidence_chunks
      }]);
      setQuery('');
    } catch (err: any) {
      setChatHistory(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateDraft = async () => {
    setDrafting(true);
    try {
      const res = await api.statelessGenerateDraft(documentText);
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: `# Case Fact Summary\n\n${res.generated_text}`,
        evidence: res.evidence_chunks
      }]);
    } catch (err: any) {
      setChatHistory(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setDrafting(false);
    }
  };

  return (
    <div className="step3-container animate-fade-in">
      <div className="workspace-header">
        <button className="btn btn-secondary" onClick={onBack}>← Back</button>
        <div className="workspace-title">
          <h3>Grounded Q&A Workspace</h3>
        </div>
        <button className="btn btn-primary" onClick={handleGenerateDraft} disabled={drafting}>
          {drafting ? 'Generating...' : '📝 Generate Case Draft'}
        </button>
      </div>

      <div className="chat-layout glass-panel">
        <div className="preset-prompts">
          <h4>Suggested Actions</h4>
          <button className="preset-btn" onClick={() => handleQuery('Summarize the key obligations of both parties.')}>Summarize Obligations</button>
          <button className="preset-btn" onClick={() => handleQuery('Identify any risk or liability clauses.')}>Risk Analysis</button>
          <button className="preset-btn" onClick={() => handleQuery('What are the termination conditions?')}>Termination Terms</button>
        </div>

        <div className="chat-main">
          <div className="chat-messages">
            {chatHistory.length === 0 && (
              <div className="empty-chat">
                Ask a question about your document or use a preset prompt.
              </div>
            )}
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`message-bubble ${msg.role}`}>
                <div className="msg-content">{msg.content}</div>
                {msg.evidence && msg.evidence.length > 0 && (
                  <div className="evidence-section">
                    <div className="evidence-title">Grounded Sources</div>
                    {msg.evidence.map((ev, i) => (
                      <div key={i} className="evidence-card">
                        <span className="ev-badge">Page {ev.page}</span>
                        <span className="ev-text">{ev.text.substring(0, 150)}...</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {loading && <div className="message-bubble assistant typing">Thinking...</div>}
          </div>

          <div className="chat-input-area">
            <input 
              type="text" 
              className="form-control" 
              placeholder="Ask anything about the document..." 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
              disabled={loading}
            />
            <button className="btn btn-primary" onClick={() => handleQuery()} disabled={loading || !query}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
