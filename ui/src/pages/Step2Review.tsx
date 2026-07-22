import React, { useState } from 'react';
import './Step2Review.css';

interface Field {
  field: string;
  value: any;
  confidence: number;
}

interface Step2ReviewProps {
  extractedData: {
    fileName: string;
    fileUrl: string;
    extractedText: string;
    structuredFields: Field[];
  };
  onProceed: (data: any) => void;
  onBack: () => void;
}

export const Step2Review: React.FC<Step2ReviewProps> = ({ extractedData, onProceed, onBack }) => {
  const [text, setText] = useState(extractedData.extractedText);
  const [fields, setFields] = useState<Field[]>(extractedData.structuredFields);
  const [activeTab, setActiveTab] = useState<'text' | 'fields'>('text');

  const handleFieldChange = (index: number, key: keyof Field, val: any) => {
    const newFields = [...fields];
    newFields[index] = { ...newFields[index], [key]: val };
    setFields(newFields);
  };

  const handleAddField = () => {
    setFields([...fields, { field: 'New Field', value: '', confidence: 1.0 }]);
  };

  const handleRemoveField = (index: number) => {
    const newFields = [...fields];
    newFields.splice(index, 1);
    setFields(newFields);
  };

  const handleSaveAndProceed = () => {
    onProceed({
      ...extractedData,
      extractedText: text,
      structuredFields: fields,
    });
  };

  return (
    <div className="step2-container animate-fade-in">
      <div className="workspace-header">
        <button className="btn btn-secondary" onClick={onBack}>← Back</button>
        <div className="workspace-title">
          <h3>Review & Edit</h3>
          <span className="file-name">{extractedData.fileName}</span>
        </div>
        <button className="btn btn-primary" onClick={handleSaveAndProceed}>Proceed to RAG Q&A →</button>
      </div>

      <div className="workspace-grid">
        {/* Left Pane: Visual Inspector */}
        <div className="workspace-pane image-pane glass-panel">
          <div className="pane-header">
            <h4>Document Viewer</h4>
          </div>
          <div className="image-container">
            {extractedData.fileUrl ? (
              <img src={extractedData.fileUrl} alt="Document Preview" />
            ) : (
              <div className="no-image">No preview available</div>
            )}
          </div>
        </div>

        {/* Right Pane: Editor */}
        <div className="workspace-pane editor-pane glass-panel">
          <div className="pane-tabs">
            <button 
              className={`tab-btn ${activeTab === 'text' ? 'active' : ''}`}
              onClick={() => setActiveTab('text')}
            >
              Extracted Text
            </button>
            <button 
              className={`tab-btn ${activeTab === 'fields' ? 'active' : ''}`}
              onClick={() => setActiveTab('fields')}
            >
              Structured Fields
            </button>
          </div>

          <div className="pane-content">
            {activeTab === 'text' ? (
              <textarea 
                className="text-editor" 
                value={text} 
                onChange={(e) => setText(e.target.value)}
                placeholder="Extracted document text will appear here..."
              />
            ) : (
              <div className="fields-editor">
                <div className="fields-list">
                  {fields.map((f, i) => (
                    <div key={i} className="field-row">
                      <input 
                        type="text" 
                        className="form-control field-key" 
                        value={f.field} 
                        onChange={(e) => handleFieldChange(i, 'field', e.target.value)}
                      />
                      <input 
                        type="text" 
                        className="form-control field-value" 
                        value={typeof f.value === 'string' ? f.value : JSON.stringify(f.value)} 
                        onChange={(e) => handleFieldChange(i, 'value', e.target.value)}
                      />
                      <div className={`conf-badge ${f.confidence < 0.7 ? 'low-conf' : ''}`}>
                        {(f.confidence * 100).toFixed(0)}%
                      </div>
                      <button className="remove-btn" onClick={() => handleRemoveField(i)}>×</button>
                    </div>
                  ))}
                </div>
                <button className="btn btn-secondary add-field-btn" onClick={handleAddField}>
                  + Add Field
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
