import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from dialogue.dialogue_manager import DialogueManager, DecisionScenario
from inference.causal_model import CausalModel
from visualization.graph_viz import CausalGraphVisualizer

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "max_age": 3600
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Load environment variables (from project root)
print("\n=== Starting API server... ===")
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in the .env file at project root")
print("OpenAI API key loaded successfully")

# Create output directory
os.makedirs("logs", exist_ok=True)
print("Logs directory created/verified")

# Global dialogue manager
dialogue_manager = DialogueManager(api_key)
print("Dialogue manager initialized")
print("=== Server startup complete ===\n")

@app.route("/api/start_scenario", methods=["POST"])
def start_scenario():
    """Start a new decision scenario"""
    print("\n=== Received start_scenario request ===")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Request data: {request.get_data(as_text=True)}")
    
    description = request.json.get("description", "")
    print(f"\nExtracted description: {description}")
    
    if not description:
        print("Error: No description provided")
        return jsonify({"error": "Please provide a decision description"}), 400
    
    # Process user input
    result = dialogue_manager.process_user_input(description)
    
    # Get graph data
    graph_data = dialogue_manager.get_current_graph_data()
    
    response_data = {
        "scenario_id": result["scenario_id"],
        "causal_relations": result["causal_relations"],
        "graph_data": graph_data,
        "follow_up_question": result["follow_up_question"]
    }
    
    print("\n=== Sending response ===")
    print(f"Response data: {response_data}")
    print("=== Request complete ===\n")
    return jsonify(response_data)

@app.route("/api/answer_question", methods=["POST"])
def answer_question():
    """Handle user's answer"""
    print("\n=== Received answer_question request ===")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Request data: {request.get_data(as_text=True)}")
    
    answer = request.json.get("answer", "")
    question = request.json.get("question", "")
    print(f"\nExtracted answer: {answer}")
    print(f"Question being answered: {question}")
    
    if not answer:
        print("Error: No answer provided")
        return jsonify({"error": "Please provide an answer"}), 400
    
    # Process user input
    result = dialogue_manager.process_user_input(answer, question)
    
    # Get graph data
    graph_data = dialogue_manager.get_current_graph_data()
    
    response_data = {
        "causal_relations": result["causal_relations"],
        "graph_data": graph_data,
        "follow_up_question": result["follow_up_question"]
    }
    
    print("\n=== Sending response ===")
    print(f"Response data: {response_data}")
    print("=== Request complete ===\n")
    return jsonify(response_data)

@app.route("/api/get_scenarios", methods=["GET"])
def get_scenarios():
    """Get all scenarios"""
    print("\n=== Received get_scenarios request ===")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    
    print(f"\nTotal scenarios: {len(dialogue_manager.scenarios)}")
    scenarios_data = []
    
    for i, scenario in enumerate(dialogue_manager.scenarios, 1):
        print(f"\nProcessing scenario {i}:")
        print(f"- Description: {scenario.description}")
        print(f"- Timestamp: {scenario.timestamp}")
        print(f"- Conversation history: {len(scenario.conversation_history)} messages")
        print(f"- Causal relations: {len(scenario.causal_relations)} relations")
        
        # Create causal model for each scenario
        causal_model = CausalModel()
        marginals = causal_model.update_from_dialogue(scenario.causal_relations)
        print(f"- Model created with {len(causal_model.graph.nodes())} nodes and {len(causal_model.graph.edges())} edges")
        
        # Generate graph data
        graph_data = {
            "nodes": [{"id": node, "label": node} for node in causal_model.graph.nodes()],
            "edges": [{"from": edge[0], "to": edge[1]} for edge in causal_model.graph.edges()]
        }
        
        scenarios_data.append({
            "id": scenario.timestamp,
            "description": scenario.description,
            "causal_relations": scenario.causal_relations,
            "conversation_history": scenario.conversation_history,
            "graph_data": graph_data
        })
    
    print("\n=== Sending response ===")
    print(f"Returning {len(scenarios_data)} scenarios")
    print("=== Request complete ===\n")
    return jsonify(scenarios_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True) 