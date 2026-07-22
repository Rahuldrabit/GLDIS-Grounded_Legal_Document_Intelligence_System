import React, { useState, useRef } from 'react';
import { api } from '../api/client';
import './Step1Upload.css';

interface Step1UploadProps {
  onExtractSuccess: (data: any) => void;
  onApiKeyClick: () => void;
}

export const Step1Upload: React.FC<Step1UploadProps> = ({ onExtractSuccess, onApiKeyClick }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setError('');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError('');
    }
  };

  const handleExtract = async () => {
    if (!file) return;
    
    setExtracting(true);
    setError('');
    
    try {
      // In stateless mode, we do not poll, we just call the stateless endpoint
      const result = await api.statelessVlmExtract(file);
      onExtractSuccess({
        fileName: file.name,
        fileUrl: URL.createObjectURL(file),
        extractedText: result.text || '',
        structuredFields: result.entities ? Object.entries(result.entities).map(([k, v]) => ({
          field: k,
          value: v,
          confidence: result.confidence || 0.95
        })) : []
      });
    } catch (err: any) {
      setError(err.message || 'Extraction failed');
    } finally {
      setExtracting(false);
    }
  };

  return (
    <div className="step1-container animate-fade-in">
      <div className="upload-header">
        <h2>Document Ingestion</h2>
        <button className="btn btn-secondary" onClick={onApiKeyClick}>
          ⚙️ BYOK Settings
        </button>
      </div>

      <div 
        className={`drop-zone glass-panel ${isDragging ? 'drag-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          style={{ display: 'none' }} 
          onChange={handleFileSelect}
          accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp"
        />
        <div className="upload-icon">📄</div>
        {file ? (
          <div className="file-selected">
            <p className="file-name">{file.name}</p>
            <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        ) : (
          <div className="drop-text">
            <p>Drag and drop your document here</p>
            <p className="sub-text">or click to browse files (PDF, PNG, JPG)</p>
          </div>
        )}
      </div>

      {error && <div className="error-alert">{error}</div>}

      <div className="extraction-controls">
        <button 
          className="btn btn-primary extract-btn hover-lift" 
          onClick={handleExtract}
          disabled={!file || extracting}
        >
          {extracting ? 'Extracting with VLM...' : 'Start VLM Extraction'}
        </button>
      </div>
    </div>
  );
};
