import { useNavigate } from 'react-router-dom';
import PropTypes from 'prop-types';
import ProgressBar from './ProgressBar';
import './PageHeader.css';

const PageHeader = ({ title, children, variant = 'default' }) => {
  const navigate = useNavigate();

  return (
    <div className={`page-header ${variant}`}>
      <div className="header-content">
        <div className="header-title">
          <div className="header-title-content">
            <div className="back-button" onClick={() => navigate('/')}>
              ← Back to Home
            </div>
            <h1>{title}</h1>
          </div>
          <ProgressBar />
        </div>
        {children}
      </div>
    </div>
  );
};

PageHeader.propTypes = {
  title: PropTypes.string.isRequired,
  children: PropTypes.node,
  variant: PropTypes.oneOf(['default', 'causal'])
};

export default PageHeader; 