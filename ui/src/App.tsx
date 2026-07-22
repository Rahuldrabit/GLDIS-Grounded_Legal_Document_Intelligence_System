import React, { useState } from 'react';
import { StepBar } from './components/StepBar';
import { ApiKeyModal } from './components/ApiKeyModal';
import { Step1Upload } from './pages/Step1Upload';
import { Step2Review } from './pages/Step2Review';
import { Step3RAG } from './pages/Step3RAG';
import './App.css';

function App() {
  const [currentStep, setCurrentStep] = useState(1);
  const [maxStepAllowed, setMaxStepAllowed] = useState(1);
  const [isApiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  
  // Document state (stateless, kept in client memory)
  const [extractedData, setExtractedData] = useState<any>(null);
  const [finalText, setFinalText] = useState('');

  const handleStepClick = (step: number) => {
    if (step <= maxStepAllowed) {
      setCurrentStep(step);
    }
  };

  const handleExtractSuccess = (data: any) => {
    setExtractedData(data);
    setMaxStepAllowed(Math.max(maxStepAllowed, 2));
    setCurrentStep(2);
  };

  const handleReviewProceed = (data: any) => {
    // Generate a single string of text for the RAG step
    const textContext = `${data.extractedText}\n\n---\nKey Fields:\n${data.structuredFields.map((f: any) => `${f.field}: ${f.value}`).join('\n')}`;
    setFinalText(textContext);
    setMaxStepAllowed(Math.max(maxStepAllowed, 3));
    setCurrentStep(3);
  };

  return (
    <div className="app-container">
      <header className="app-topbar">
        <h1>GLDIS Workspace</h1>
        <button className="btn btn-secondary" onClick={() => setApiKeyModalOpen(true)}>
          ⚙️ BYOK Settings
        </button>
      </header>

      <StepBar 
        currentStep={currentStep} 
        onStepClick={handleStepClick} 
        maxStepAllowed={maxStepAllowed} 
      />

      <main className="app-main">
        {currentStep === 1 && (
          <Step1Upload 
            onExtractSuccess={handleExtractSuccess} 
            onApiKeyClick={() => setApiKeyModalOpen(true)}
          />
        )}
        
        {currentStep === 2 && extractedData && (
          <Step2Review 
            extractedData={extractedData}
            onProceed={handleReviewProceed}
            onBack={() => setCurrentStep(1)}
          />
        )}

        {currentStep === 3 && finalText && (
          <Step3RAG 
            documentText={finalText}
            onBack={() => setCurrentStep(2)}
          />
        )}
      </main>

      <ApiKeyModal 
        isOpen={isApiKeyModalOpen} 
        onClose={() => setApiKeyModalOpen(false)} 
      />
    </div>
  );
}

export default App;
