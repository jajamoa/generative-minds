import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import '../styles/StructuredSurvey.css'
import surveyConfig from '../config/surveyQuestions.json'

function StructuredSurvey() {
  const navigate = useNavigate()
  const [currentSection, setCurrentSection] = useState(0)
  const [answers, setAnswers] = useState({})

  const handleAnswer = (questionId, value) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value
    }))
  }

  const handleNext = () => {
    if (currentSection < surveyConfig.sections.length - 1) {
      setCurrentSection(currentSection + 1)
    }
  }

  const handlePrevious = () => {
    if (currentSection > 0) {
      setCurrentSection(currentSection - 1)
    }
  }

  const currentSectionData = surveyConfig.sections[currentSection]

  return (
    <div className="survey-container">
      <div className="back-button" onClick={() => navigate('/')}>
        ← Back to Home
      </div>

      <div className="survey-content">
        <div className="survey-header">
          <h1>{surveyConfig.title}</h1>
          <div className="section-navigation">
            {surveyConfig.sections.map((section, index) => (
              <div 
                key={section.id}
                className={`section-indicator ${index === currentSection ? 'active' : ''} 
                  ${index < currentSection ? 'completed' : ''}`}
              >
                {index + 1}
              </div>
            ))}
          </div>
          <h2>{currentSectionData.title}</h2>
        </div>

        <div className="questions-container">
          {currentSectionData.questions.map(question => (
            <div key={question.id} className="question-item">
              <label>{question.text}</label>
              {question.type === 'radio' && (
                <div className="radio-options">
                  {question.options.map(option => (
                    <label key={option.value} className="radio-label">
                      <input
                        type="radio"
                        name={question.id}
                        value={option.value}
                        checked={answers[question.id] === option.value}
                        onChange={(e) => handleAnswer(question.id, e.target.value)}
                      />
                      <span>{option.label}</span>
                    </label>
                  ))}
                </div>
              )}
              {question.type === 'text' && (
                <input
                  type="text"
                  value={answers[question.id] || ''}
                  onChange={(e) => handleAnswer(question.id, e.target.value)}
                  placeholder={question.placeholder}
                />
              )}
              {question.type === 'textarea' && (
                <textarea
                  value={answers[question.id] || ''}
                  onChange={(e) => handleAnswer(question.id, e.target.value)}
                  placeholder={question.placeholder}
                  rows={4}
                />
              )}
            </div>
          ))}
        </div>

        <div className="survey-navigation">
          <button 
            className="nav-button"
            onClick={handlePrevious}
            disabled={currentSection === 0}
          >
            ← Previous
          </button>
          <div className="section-counter">
            Section {currentSection + 1} of {surveyConfig.sections.length}
          </div>
          <button 
            className="nav-button"
            onClick={handleNext}
            disabled={currentSection === surveyConfig.sections.length - 1}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}

export default StructuredSurvey 