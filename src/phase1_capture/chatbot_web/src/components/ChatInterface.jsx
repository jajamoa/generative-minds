import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import '../styles/ChatInterface.css'
import scenarioConfig from '../config/chatScenarios.json'

function ChatInterface() {
  const navigate = useNavigate()
  const [currentScenario, setCurrentScenario] = useState(0)
  const [messages, setMessages] = useState([])
  const [isTyping, setIsTyping] = useState(false)
  const [showSubtitles, setShowSubtitles] = useState(true)
  const [isPaused, setIsPaused] = useState(false)

  useEffect(() => {
    if (currentScenario === 0) {
      // Start with first question after a delay
      setTimeout(() => {
        addMessage(scenarioConfig.scenarios[0])
      }, 1000)
    }
  }, [])

  const addMessage = (scenario) => {
    setIsTyping(true)
    setTimeout(() => {
      setMessages([...messages, {
        role: 'assistant',
        content: scenario.question,
        type: scenario.type
      }])
      setIsTyping(false)
      setCurrentScenario(currentScenario + 1)
    }, 1500)
  }

  const handleNext = () => {
    if (currentScenario < scenarioConfig.scenarios.length) {
      addMessage(scenarioConfig.scenarios[currentScenario])
    }
  }

  const progress = (currentScenario / scenarioConfig.scenarios.length) * 100

  return (
    <div className="chat-interface">
      <div className="back-button" onClick={() => navigate('/')}>
        ← Back to Home
      </div>
      <div className="chat-container">
        <div className="chat-circle">
          {isTyping ? (
            <div className="microphone-icon">
              <span className="typing-indicator"></span>
            </div>
          ) : (
            <div className="agent-avatar">AI</div>
          )}
        </div>

        <div className="progress-container">
          <div className="progress-bar">
            <div className="progress-track">
              <div 
                className="progress-fill"
                style={{ width: `${progress}%` }}
              ></div>
              <div 
                className="progress-indicator"
                style={{ left: `${progress}%` }}
              ></div>
            </div>
          </div>
          <div className="progress-labels">
            <span>Start</span>
            <span>End</span>
          </div>
        </div>

        <div className="controls">
          <button 
            className="control-button"
            onClick={() => setShowSubtitles(!showSubtitles)}
          >
            {showSubtitles ? 'Hide' : 'Show'} Subtitles
          </button>
          <button 
            className="control-button"
            onClick={() => setIsPaused(!isPaused)}
          >
            {isPaused ? '▷' : '||'}
          </button>
          <button 
            className="control-button"
            onClick={handleNext}
          >
            Next →
          </button>
        </div>

        {showSubtitles && messages.length > 0 && (
          <div className="subtitles">
            {messages[messages.length - 1].content}
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatInterface 