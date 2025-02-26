import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import PropTypes from 'prop-types';
import './App.css';
import moduleConfig from './config/modules.json';
import aboutConfig from './config/about.json';
import StructuredSurvey from './components/StructuredSurvey';
import GeographicProfile from './components/GeographicProfile';
import CausalSurvey from './components/CausalSurvey';
import ModuleInfo from './components/common/ModuleInfo';
import Button from './components/common/Button';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2196F3',
    },
    background: {
      default: '#000000',
    },
  },
});

// Main layout component that includes header and sidebar
function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();

  const handleEmailClick = () => {
    window.location.href = 'mailto:cli@mit.edu';
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="user-info">
          <div className="user-icon">AI</div>
          <span className="user-email" onClick={handleEmailClick} style={{ cursor: 'pointer' }}>
            cli@mit.edu
          </span>
        </div>
      </header>
      <div className="main-content">
        <nav className="sidebar">
          <div 
            className={`nav-item ${location.pathname === '/' ? 'active' : ''}`}
            onClick={() => navigate('/')}
          >
            Home
          </div>
          <div 
            className={`nav-item ${location.pathname === '/about' ? 'active' : ''}`}
            onClick={() => navigate('/about')}
          >
            About
          </div>
        </nav>
        <main className="content">
          {children}
        </main>
      </div>
    </div>
  );
}

Layout.propTypes = {
  children: PropTypes.node.isRequired,
};

// Home page component
function Home() {
  const navigate = useNavigate();

  const handleStartModule = (moduleId) => {
    if (moduleId === 2) {
      navigate('/geographic');
    } else if (moduleId === 3) {
      navigate('/survey');
    } else if (moduleId === 4) {
      navigate('/causal');
    }
  };

  return (
    <>
      <h1>{moduleConfig.title}</h1>
      <div className="modules-container">
        <div className="modules-list">
          {moduleConfig.modules.map(module => (
            <ModuleInfo
              key={module.id}
              {...module}
              onStart={() => handleStartModule(module.id)}
            />
          ))}
        </div>

        <div className="instructions-panel">
          <h2>Study Instructions</h2>
          <p>Thank you again for participating in our study!</p>
          <p>Your task is to complete the modules shown on the left (or above, depending on your screen size). The modules will become available one by one. Please start with the module that is marked with the "DO THIS NEXT!" sign.</p>
        </div>
      </div>
    </>
  );
}

// About page component
function About() {
  return (
    <div className="about-content">
      <h1>{aboutConfig.title}</h1>
      <div className="about-text">
        {aboutConfig.content.map((item, index) => (
          <p key={index}>{item.paragraph}</p>
        ))}
      </div>
    </div>
  );
}

const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<Layout><Home /></Layout>} />
          <Route path="/about" element={<Layout><About /></Layout>} />
          <Route path="/geographic" element={<Layout><GeographicProfile /></Layout>} />
          <Route path="/survey" element={<Layout><StructuredSurvey /></Layout>} />
          <Route path="/causal" element={<Layout><CausalSurvey /></Layout>} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
};

export default App;
