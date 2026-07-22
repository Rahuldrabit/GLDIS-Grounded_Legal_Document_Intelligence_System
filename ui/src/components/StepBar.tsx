import React from 'react';
import './StepBar.css';

interface StepBarProps {
  currentStep: number;
  onStepClick: (step: number) => void;
  maxStepAllowed: number;
}

export const StepBar: React.FC<StepBarProps> = ({ currentStep, onStepClick, maxStepAllowed }) => {
  const steps = [
    { num: 1, label: 'Upload & Extract' },
    { num: 2, label: 'Review & Edit' },
    { num: 3, label: 'RAG Q&A' }
  ];

  return (
    <div className="step-bar">
      {steps.map((step, idx) => {
        const isPast = step.num < currentStep;
        const isActive = step.num === currentStep;
        const isLocked = step.num > maxStepAllowed;

        return (
          <React.Fragment key={step.num}>
            <div 
              className={`step-item ${isActive ? 'active' : ''} ${isPast ? 'past' : ''} ${isLocked ? 'locked' : ''}`}
              onClick={() => !isLocked && onStepClick(step.num)}
            >
              <div className="step-circle">{step.num}</div>
              <div className="step-label">{step.label}</div>
            </div>
            {idx < steps.length - 1 && (
              <div className={`step-connector ${isPast ? 'filled' : ''}`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
