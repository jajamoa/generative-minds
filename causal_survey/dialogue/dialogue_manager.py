import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import openai

class DecisionScenario:
    def __init__(self, description: str):
        """Initialize decision scenario"""
        self.description = description
        self.causal_relations = []
        self.conversation_history = []
        self.timestamp = datetime.now().isoformat()

class DialogueManager:
    def __init__(self, api_key: str):
        """Initialize dialogue manager"""
        self.client = openai.OpenAI(api_key=api_key)
        self.chat_id = str(uuid.uuid4())
        self.scenarios: List[DecisionScenario] = []
        self.current_scenario: Optional[DecisionScenario] = None
        
    def process_user_input(self, description: str, question: str = None) -> Dict[str, Any]:
        """Process user input and return response with graph data"""
        if not self.current_scenario:
            # Start new scenario
            self.current_scenario = DecisionScenario(description)
            self.scenarios.append(self.current_scenario)
        else:
            # Record answer to previous question
            self.current_scenario.conversation_history.append({
                "role": "user",
                "content": description,
                "question": question,
                "timestamp": datetime.now().isoformat()
            })
        
        # Extract causal relations
        relations = self.extract_causal_relations(description)
        
        # Generate follow-up question
        follow_up = self.generate_follow_up_question(description)
        
        # Save log
        self.save_log()
        
        # If no follow-up question, scenario is complete
        if not follow_up:
            self.current_scenario = None
        
        return {
            "scenario_id": self.current_scenario.timestamp if self.current_scenario else None,
            "causal_relations": relations,
            "follow_up_question": follow_up
        }
    
    def extract_causal_relations(self, text: str) -> List[Dict[str, str]]:
        """Use GPT-4 to extract causal relations"""
        try:
            # Build complete dialogue history text
            full_context = ""
            if self.current_scenario:
                full_context = self.current_scenario.description + "\n\n"
                for msg in self.current_scenario.conversation_history:
                    if msg["role"] == "user":
                        full_context += f"Q: {msg['question']}\nA: {msg['content']}\n"
            
            prompt = f"""Please analyze all causal relationships in the following complete dialogue.
            Return the results in JSON format as:
            {{"relations": [
                {{"cause": "cause", "effect": "effect"}}
            ]}}
            If there are multiple causal relationships, add multiple objects in the array.
            Make sure not to miss any causal relationships mentioned before.
            Return only JSON, no other explanation.
            
            Complete dialogue:
            {full_context}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in analyzing causal relationships. Please analyze the causal relationships in the entire dialogue context and return only the JSON format analysis results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            relations = result.get("relations", [])
            if isinstance(relations, dict):
                relations = [relations]
            if self.current_scenario:
                # Merge old and new causal relations, remove duplicates
                existing_relations = self.current_scenario.causal_relations
                merged_relations = existing_relations.copy()
                for new_relation in relations:
                    if new_relation not in existing_relations:
                        merged_relations.append(new_relation)
                self.current_scenario.causal_relations = merged_relations
                # Save log after extracting new causal relations
                self.save_log()
            return relations
        except Exception as e:
            print(f"Error extracting causal relations: {e}")
            return []
            
    def generate_follow_up_question(self, previous_answer: str) -> Optional[str]:
        """Generate follow-up question"""
        try:
            # Build prompt with dialogue history
            conversation_context = ""
            if self.current_scenario:
                for msg in self.current_scenario.conversation_history:
                    if msg["role"] == "user":
                        conversation_context += f"Q: {msg['question']}\nA: {msg['content']}\n"
            
            # Truncate previous answer if too long
            max_length = 4000  # Maximum tokens for context
            if len(previous_answer) > max_length:
                previous_answer = previous_answer[:max_length] + "..."
            
            prompt = f"""Based on the following dialogue history and the user's current answer, generate a follow-up question to better understand their decision process.
            The question should be specific, relevant to the context, and help uncover additional causal relationships.
            If enough information has been collected to understand the decision process and causal relationships, return an empty string.
            Return only the question itself, no other explanation.
            
            Previous dialogue history:
            {conversation_context}
            
            Current answer: {previous_answer}
            
            Requirements for the follow-up question:
            1. Must be specific and focused
            2. Should relate to previously mentioned factors
            3. Should help uncover new causal relationships
            4. Should be a complete, well-formed question
            5. Should not repeat previously asked questions"""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in generating follow-up questions. Based on the complete dialogue history, determine if a follow-up is needed. If needed, return a specific, well-formed question; if not, return an empty string."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=150  # Limit response length
            )
            
            follow_up = response.choices[0].message.content.strip()
            if not follow_up or len(follow_up) < 10:  # Ensure question is not too short
                return None
            return follow_up
        except Exception as e:
            print(f"Error generating follow-up question: {e}")
            return None
        
    def save_log(self):
        """Save dialogue log"""
        scenarios_data = []
        for scenario in self.scenarios:
            scenarios_data.append({
                "description": scenario.description,
                "conversation_history": scenario.conversation_history,
                "causal_relations": scenario.causal_relations,
                "timestamp": scenario.timestamp
            })
            
        log_data = {
            "chat_id": self.chat_id,
            "scenarios": scenarios_data,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"logs/{self.chat_id}.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
    
    def get_current_graph_data(self) -> Dict[str, Any]:
        """Get current graph data for visualization"""
        if not self.current_scenario:
            return {"nodes": [], "edges": []}
            
        # Convert causal relations to graph data
        nodes = set()
        edges = []
        for relation in self.current_scenario.causal_relations:
            nodes.add(relation["cause"])
            nodes.add(relation["effect"])
            edges.append({"from": relation["cause"], "to": relation["effect"]})
            
        return {
            "nodes": [{"id": node, "label": node} for node in nodes],
            "edges": edges
        }