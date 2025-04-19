import React from 'react';
import PropTypes from 'prop-types';
import Card from './Card';
import Button from './Button';
import './ModuleInfo.css';

const ModuleInfo = ({
  title,
  description,
  detail,
  required,
  estimatedTime,
  steps,
  onStart,
  status,
}) => {
  return (
    <Card className="module-card">
      <div className="module-info">
        <h3>{title}</h3>
        <h4>{description}</h4>
        <p>{detail}</p>
        {required && <p className="required">(Required component)</p>}
        <p className="estimated-time">{estimatedTime}</p>
        
        <div className="steps-info">
          {steps?.map(step => (
            <div key={step.id} className="step-item">
              <span className="step-title">{step.title}</span>
              <span className="step-description">{step.description}</span>
            </div>
          ))}
        </div>
      </div>
      
      <Button 
        variant="start"
        onClick={onStart}
        className="start-button"
      >
        Start the Module
      </Button>
      {status && <span className="status-badge">{status}</span>}
    </Card>
  );
};

ModuleInfo.propTypes = {
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  detail: PropTypes.string,
  required: PropTypes.bool,
  estimatedTime: PropTypes.string,
  steps: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.string.isRequired,
    title: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
  })),
  onStart: PropTypes.func.isRequired,
  status: PropTypes.string,
};

export default ModuleInfo; 