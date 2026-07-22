import React, { useState, useEffect } from 'react';
import './ApiKeyModal.css';

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const PROVIDERS = [
  { id: 'google', name: 'Google Gemini (Native or OpenAI-compatible)' },
  { id: 'openrouter', name: 'OpenRouter' },
  { id: 'openai', name: 'OpenAI' },
  { id: 'lmstudio', name: 'LM Studio (Local)' }
];

const PRESET_MODELS: Record<string, string[]> = {
  google: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash-exp'],
  openrouter: ['google/gemma-2-9b-it:free', 'anthropic/claude-3.5-sonnet', 'openai/gpt-4o-mini', 'meta-llama/llama-3.3-70b-instruct'],
  openai: ['gpt-4o-mini', 'gpt-4o'],
  lmstudio: ['qwen2.5-vl-7b-instruct', 'llama-3.2-1b-instruct']
};

export const ApiKeyModal: React.FC<ApiKeyModalProps> = ({ isOpen, onClose }) => {
  const [provider, setProvider] = useState('google');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gemini-2.5-flash');
  const [baseUrl, setBaseUrl] = useState('');
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setProvider(localStorage.getItem('llm_provider') || 'google');
      setApiKey(localStorage.getItem('llm_api_key') || '');
      setModel(localStorage.getItem('llm_model') || 'gemini-2.5-flash');
      setBaseUrl(localStorage.getItem('llm_base_url') || '');
    }
  }, [isOpen]);

  const handleProviderChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newProv = e.target.value;
    setProvider(newProv);
    setModel(PRESET_MODELS[newProv]?.[0] || '');
  };

  const handleSave = () => {
    localStorage.setItem('llm_provider', provider);
    localStorage.setItem('llm_api_key', apiKey);
    localStorage.setItem('llm_model', model);
    localStorage.setItem('llm_base_url', baseUrl);
    onClose();
  };

  const handleClear = () => {
    localStorage.removeItem('llm_provider');
    localStorage.removeItem('llm_api_key');
    localStorage.removeItem('llm_model');
    localStorage.removeItem('llm_base_url');
    setApiKey('');
    setBaseUrl('');
    setProvider('google');
    setModel('gemini-2.5-flash');
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content glass-panel animate-fade-in">
        <header className="modal-header">
          <h2>API Settings (BYOK)</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </header>

        <div className="modal-body">
          <p className="help-text">Give your API key. Keys are stored locally in your browser and sent securely via headers.</p>
          
          <div className="form-group">
            <label>Provider</label>
            <select value={provider} onChange={handleProviderChange} className="form-control">
              {PROVIDERS.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>API Key</label>
            <div className="input-with-icon">
              <input 
                type={showKey ? 'text' : 'password'} 
                value={apiKey} 
                onChange={(e) => setApiKey(e.target.value)} 
                placeholder={`Enter ${provider} API Key...`}
                className="form-control"
              />
              <button 
                type="button" 
                className="toggle-visibility" 
                onClick={() => setShowKey(!showKey)}
              >
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>

          <div className="form-group">
            <label>Model</label>
            <input 
              type="text" 
              value={model} 
              onChange={(e) => setModel(e.target.value)} 
              list="preset-models"
              className="form-control"
              placeholder="e.g. gemini-2.5-flash"
            />
            <datalist id="preset-models">
              {PRESET_MODELS[provider]?.map(m => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>

          <div className="form-group">
            <label>Custom Base URL (Optional)</label>
            <input 
              type="text" 
              value={baseUrl} 
              onChange={(e) => setBaseUrl(e.target.value)} 
              className="form-control"
              placeholder={provider === 'lmstudio' ? 'http://localhost:1234/v1' : 'Leave empty for default'}
            />
          </div>
        </div>

        <footer className="modal-footer">
          <button className="btn btn-secondary" onClick={handleClear}>Clear Keys</button>
          <div className="action-buttons">
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSave}>Save Settings</button>
          </div>
        </footer>
      </div>
    </div>
  );
};
