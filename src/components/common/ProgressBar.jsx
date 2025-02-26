import PropTypes from 'prop-types';
import './ProgressBar.css';

const ProgressBar = ({ progress = 50 }) => {
  return (
    <div className="progress-container">
      <div className="progress-bar">
        <div 
          className="progress-fill"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="progress-number">
        {progress}%
      </div>
    </div>
  );
};

ProgressBar.propTypes = {
  progress: PropTypes.number
};

export default ProgressBar; 