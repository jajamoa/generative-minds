import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import './App.css'
import moduleConfig from './config/modules.json'
import aboutConfig from './config/about.json'
import ChatInterface from './components/ChatInterface'
import StructuredSurvey from './components/StructuredSurvey'
import GeographicProfile from './components/GeographicProfile'

// Main layout component that includes header and sidebar
function Layout({ children }) {
  const navigate = useNavigate()
  const location = useLocation()

  const handleEmailClick = () => {
    window.location.href = 'mailto:cli@mit.edu'
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="user-info">
          <span className="user-icon">G</span>
          <span className="user-email">test_user@test.com</span>
        </div>
      </header>

      <div className="main-content">
        <nav className="sidebar">
          <div 
            className={`nav-item ${location.pathname === '/' ? 'active' : ''}`}
            onClick={() => navigate('/')}
          >
            <span>Home</span>
          </div>
          <div 
            className={`nav-item ${location.pathname === '/about' ? 'active' : ''}`}
            onClick={() => navigate('/about')}
          >
            <span>About the study</span>
          </div>
          <div 
            className="nav-item"
            onClick={handleEmailClick}
          >
            <span>Email the admin</span>
          </div>
        </nav>

        <div className="content">
          {children}
        </div>
      </div>
    </div>
  )
}

// Home page component
function Home() {
  const navigate = useNavigate()

  const handleStartModule = (moduleId) => {
    if (moduleId === 2) { // Preparation Pt. 2 - Geographic Profile
      navigate('/geographic')
    } else if (moduleId === 3) { // Structured Survey
      navigate('/survey')
    } else if (moduleId === 4) { // Chatbot
      navigate('/chat')
    }
  }

  return (
    <>
      <h1>{moduleConfig.title}</h1>
      <div className="modules-container">
        <div className="modules-list">
          {moduleConfig.modules.map(module => (
            <div key={module.id} className="module-card">
              <div className="module-info">
                <h3>{module.title}</h3>
                <h4>{module.description}</h4>
                <p>{module.detail}</p>
                {module.required && <p className="required">(Required component)</p>}
                <p className="estimated-time">{module.estimatedTime}</p>
                
                {/* 显示步骤信息 */}
                {module.steps && (
                  <div className="steps-info">
                    {module.steps.map(step => (
                      <div key={step.id} className="step-item">
                        <span className="step-title">{step.title}</span>
                        <span className="step-description">{step.description}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              <button 
                className="start-button"
                onClick={() => handleStartModule(module.id)}
              >
                Start the Module
              </button>
              {module.status && <span className="status-badge">{module.status}</span>}
            </div>
          ))}
        </div>

        <div className="instructions-panel">
          <h2>Study Instructions</h2>
          <p>Thank you again for participating in our study!</p>
          <p>Your task is to complete the modules shown on the left (or above, depending on your screen size). The modules will become available one by one. Please start with the module that is marked with the "DO THIS NEXT!" sign.</p>
        </div>
      </div>
    </>
  )
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
  )
}

// Main App component
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout><Home /></Layout>} />
        <Route path="/about" element={<Layout><About /></Layout>} />
        <Route path="/geographic" element={<Layout><GeographicProfile /></Layout>} />
        <Route path="/survey" element={<Layout><StructuredSurvey /></Layout>} />
        <Route path="/chat" element={<Layout><ChatInterface /></Layout>} />
      </Routes>
    </Router>
  )
}

export default App
