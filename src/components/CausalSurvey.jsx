import { useState } from 'react';
import CausalGraph from './CausalGraph';
import Button from './common/Button';
import PageHeader from './common/PageHeader';
import './CausalSurvey.css';

const CausalSurvey = () => {
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: `Welcome to the AI Causal Reasoning System!

Please describe a recent decision you made. For example:
- "I decided to buy an electric car because it's environmentally friendly and cost-effective in the long run"
- "I chose to work from home because it offers better work-life balance"
- "I started learning programming because I want to switch careers to development"

The more details you provide, the better I can analyze the causal relationships.`
  }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [currentScenarioId, setCurrentScenarioId] = useState(null);
  const [showGraph, setShowGraph] = useState(false);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    setLoading(true);
    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    try {
      const endpoint = !currentScenarioId ? '/api/start_scenario' : '/api/answer_question';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          description: userMessage,
          answer: userMessage,
          question: messages[messages.length - 1]?.content
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.error }]);
        return;
      }

      if (data.scenario_id) {
        setCurrentScenarioId(data.scenario_id);
      }

      if (data.graph_data) {
        setGraphData(data.graph_data);
      }

      if (data.causal_relations?.length > 0) {
        const relationsText = data.causal_relations
          .map(r => `- ${r.cause} → ${r.effect}`)
          .join('\n');
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `I found the following causal relationships:\n${relationsText}`
        }]);
      }

      if (data.follow_up_question) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.follow_up_question
        }]);
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'The information for this decision scenario is complete. You can:\n1. Click "New Scenario" to start a new analysis\n2. Check the causal graph on the right to understand your decision pattern'
        }]);
        setCurrentScenarioId(null);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${error.message}. Please try again later.`
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleNewScenario = () => {
    setCurrentScenarioId(null);
    setGraphData({ nodes: [], edges: [] });
    setMessages([{
      role: 'assistant',
      content: 'Please describe a new decision scenario.'
    }]);
  };

  const handleTextareaInput = (e) => {
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    setInput(e.target.value);
  };

  return (
    <PageHeader title="SF Zoning Chatbot" variant="causal">
      <div className="causal-survey">
        <main className="chat-container">
          <div className="messages-container">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message-wrapper ${message.role}`}
              >
                <div className="message-content">
                  <div className="message-icon">
                    {message.role === 'assistant' ? '🤖' : '👤'}
                  </div>
                  <div className="message-text">
                    {message.content}
                  </div>
                </div>
              </div>
            ))}
            {loading && (
              <div className="message-wrapper assistant">
                <div className="message-content">
                  <div className="message-icon">
                    🤖
                  </div>
                  <div className="message-text">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                value={input}
                onChange={handleTextareaInput}
                placeholder={currentScenarioId 
                  ? "Please answer the question..."
                  : "Describe a decision scenario, e.g., 'I recently bought an electric car because...'"}
                disabled={loading}
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
              />
              <Button
                variant="primary"
                onClick={handleSubmit}
                disabled={loading || !input.trim()}
                className="send-button"
              >
                <svg viewBox="0 0 24 24" className="send-icon">
                  <path fill="currentColor" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
              </Button>
            </div>
            <div className="input-help">
              Press Enter to send, Shift + Enter for new line
            </div>
          </div>
        </main>

        <div className={`causal-sidebar right ${showGraph ? 'expanded' : ''}`}>
          <div className="graph-container">
            <h2>Causal Graph</h2>
            <p className="graph-description">
              This graph shows the causal relationships in your decision process.
            </p>
            <CausalGraph
              nodes={graphData.nodes}
              edges={graphData.edges}
            />
          </div>
        </div>
        
        <button 
          className="graph-toggle"
          onClick={() => setShowGraph(!showGraph)}
        >
          {showGraph ? '→' : '←'} Graph
        </button>
      </div>
    </PageHeader>
  );
};

export default CausalSurvey; 