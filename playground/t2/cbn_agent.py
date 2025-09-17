"""CBN Agent for processing belief attribution tasks"""

import json
from typing import Dict, Optional, Tuple, Any
from llm_utils import QwenLLM

class CBNAgent:
    """Agent that uses causal Bayesian networks for belief attribution"""
    
    def __init__(self, model: str = "qwen-plus", cbn_path: str = "sample_cbn.json"):
        """Initialize CBN agent
        
        Args:
            model: Name of LLM model to use
            cbn_path: Path to causal Bayesian network definition
        """
        self.llm = QwenLLM(model=model)
        with open(cbn_path) as f:
            cbn_data = json.load(f)
            
        # Store all CBNs indexed by prolific_id
        self.cbns_by_id = {}
        self.default_cbn = None
        
        if isinstance(cbn_data, list) and len(cbn_data) > 0:
            for session in cbn_data:
                if "graphs" in session and len(session["graphs"]) > 0:
                    prolific_id = session.get("prolificId", "unknown")
                    first_graph = session["graphs"][0]
                    if "graphData" in first_graph:
                        graph_data = first_graph["graphData"]
                        self.cbns_by_id[prolific_id] = graph_data
                        # Set first valid CBN as default
                        if self.default_cbn is None:
                            self.default_cbn = graph_data
            
            if self.default_cbn is None:
                raise ValueError("No valid CBN found in data")
        else:
            # Fallback: assume it's already in the correct format
            self.default_cbn = cbn_data
            self.cbns_by_id["default"] = cbn_data
            
    def select_cbn(self, prolific_id: Optional[str] = None) -> Dict[str, Any]:
        """Select CBN based on prolific_id
        
        Args:
            prolific_id: Participant's prolific ID to find specific CBN
            
        Returns:
            CBN graphData for the specified participant, or default if not found
        """
        if prolific_id and prolific_id in self.cbns_by_id:
            return self.cbns_by_id[prolific_id]
        
        # Fallback to default CBN
        return self.default_cbn
    
    def translate_to_do_operation(
        self,
        question: str,
        demographics: Dict[str, Any],
        context_qas: list,
        include_demographics: bool,
        include_context: bool,
        temperature: float = 0.1,
        debug: bool = False,
        prolific_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Translate question to do() operation based on CBN variables"""
        
        # Build context
        context_parts = []
        if include_demographics and demographics:
            demo_text = "Demographics: " + ", ".join([f"{k}: {v}" for k, v in demographics.items()])
            context_parts.append(demo_text)
        
        if include_context and context_qas:
            context_text = "Conversation: " + " ".join([f"Q: {qa['question']} A: {qa['answer']}" for qa in context_qas])
            context_parts.append(context_text)
        
        context = " | ".join(context_parts)
        
        # Get the appropriate CBN for this participant
        current_cbn = self.select_cbn(prolific_id)
        
        # Get CBN variables
        cbn_variables = list(current_cbn.get('nodes', {}).keys())
        
        prompt = f"""Context: {context}
Question: {question}
Available CBN Variables: {cbn_variables}

Translate this question into a causal intervention (do operation).
Which variable should be intervened on and what value should it be set to?
Format: {{"variable": "variable_name", "value": 0.8}}"""
        
        if debug:
            print(f"\n=== CBN Agent: Translating to do() operation ===")
        
        response = self.llm.generate_response(
            prompt,
            system_message="You are translating questions to causal interventions. Return only JSON.",
            temperature=temperature,
            return_json=True,
            debug=debug
        )
        
        if debug:
            print(f"Do operation: {response}")
        
        return response or {"variable": cbn_variables[0] if cbn_variables else "unknown", "value": 0.5}
    
    def run_cbn_inference(
        self,
        do_operation: Dict[str, Any],
        debug: bool = False,
        prolific_id: Optional[str] = None
    ) -> Dict[str, float]:
        """Run CBN inference with do() operation"""
        
        # Get the appropriate CBN for this participant
        current_cbn = self.select_cbn(prolific_id)
        
        # Initialize beliefs
        beliefs = {}
        nodes = current_cbn.get('nodes', {})
        edges = current_cbn.get('edges', {})
        
        # Set all nodes to default values first
        for node in nodes:
            beliefs[node] = 0.5  # neutral belief
        
        # Apply do() operation - fix the intervened variable
        intervention_var = do_operation.get('variable')
        intervention_value = float(do_operation.get('value', 0.5))
        
        if intervention_var in nodes:
            beliefs[intervention_var] = intervention_value
        
        # Forward propagation through the network
        # Simple approach: iterate through edges and update beliefs
        for _ in range(3):  # Multiple passes for convergence
            updated = False
            for edge_id, edge in edges.items():
                source = edge.get('source')
                target = edge.get('target')
                modifier = edge.get('modifier', 1.0)
                
                # Skip if target is the intervention variable (it's fixed)
                if target == intervention_var:
                    continue
                
                if source in beliefs and target in beliefs:
                    # Calculate influence
                    source_prob = beliefs[source]
                    # Simple causal influence: target influenced by source
                    influence = source_prob * modifier
                    new_value = min(0.95, max(0.05, 0.5 + (influence - 0.5) * 0.7))
                    
                    if abs(new_value - beliefs[target]) > 0.01:
                        beliefs[target] = new_value
                        updated = True
            
            if not updated:
                break
        
        if debug:
            print(f"\n=== CBN Agent: Running CBN inference ===")
            print(f"Intervention: {intervention_var} = {intervention_value}")
            print(f"CBN inference result: {beliefs}")
        
        return beliefs
    
    def select_answer(
        self,
        vqa: Dict[str, Any],
        cbn_state: Dict[str, float],
        temperature: float = 0.1,
        debug: bool = False
    ) -> str:
        """Select answer based on updated CBN state"""
        
        question = vqa.get("task_question", "")
        task_type = vqa.get("task_type", "belief_attribution")
        
        # Format CBN state for LLM
        cbn_summary = ", ".join([f"{var}: {val:.2f}" for var, val in cbn_state.items()])
        
        if task_type == "belief_update":
            # Handle scale prediction for belief_update
            scale = vqa.get("scale", [1, 10])
            
            prompt = f"""Question: {question}
Current CBN State: {cbn_summary}
Scale: {scale[0]} to {scale[1]}

Based on the causal network state, predict a number on the scale from {scale[0]} to {scale[1]}.
Return only the number."""
            
            if debug:
                print(f"\n=== CBN Agent: Predicting scale value (belief_update) ===")
            
            response = self.llm.generate_response(
                prompt,
                system_message="You are predicting scale values based on causal inference results. Return only a number.",
                temperature=temperature,
                debug=debug
            )
            
            # Extract number from response
            if response:
                import re
                numbers = re.findall(r'\b\d+\b', response.strip())
                if numbers:
                    try:
                        predicted_value = int(numbers[0])
                        # Clamp to scale range
                        predicted_value = max(scale[0], min(scale[1], predicted_value))
                        return str(predicted_value)
                    except ValueError:
                        pass
            
            # Fallback: return middle of scale
            return str((scale[0] + scale[1]) // 2)
            
        else:
            # Handle multiple choice for belief_attribution
            answer_options = vqa.get("answer_options", {})
            
            prompt = f"""Question: {question}
Current CBN State: {cbn_summary}
Answer Options: {answer_options}

Based on the causal network state after intervention, which answer best reflects the likely outcome?
Return only the letter (A, B, C, etc.)."""
            
            if debug:
                print(f"\n=== CBN Agent: Selecting answer (belief_attribution) ===")
            
            response = self.llm.generate_response(
                prompt,
                system_message="You are selecting answers based on causal inference results.",
                temperature=temperature,
                debug=debug
            )
            
            # Extract just the letter
            if response:
                response = response.strip().upper()
                for option in answer_options.keys():
                    if option.upper() in response:
                        return option.upper()
            
            return list(answer_options.keys())[0] if answer_options else "A"
    
    def process_query(
        self,
        vqa: Dict[str, Any],
        demographics: Dict[str, Any],
        context_qas: list,
        include_demographics: bool,
        include_context: bool,
        temperature: float = 0.1,
        debug: bool = False
    ) -> str:
        """Process query using CBN causal inference pipeline"""
        
        if debug:
            print(f"\n{'='*80}")
            print(f"CBN AGENT: Starting causal inference pipeline")
            print(f"Task Type: {vqa.get('task_type', 'unknown')}")
            print(f"Question: {vqa.get('task_question', '')}")
            print(f"{'='*80}")
        
        # Get prolific_id from vqa data for CBN selection
        prolific_id = vqa.get("prolific_id")
        
        # Step 1: Select CBN based on prolific_id
        selected_cbn = self.select_cbn(prolific_id)
        if debug:
            print(f"\n=== Step 1: CBN Selected ===")
            print(f"Prolific ID: {prolific_id}")
            print(f"CBN nodes: {len(selected_cbn.get('nodes', {}))}")
            print(f"CBN edges: {len(selected_cbn.get('edges', {}))}")
        
        # Step 2: Translate question to do() operation
        task_question = vqa.get("task_question", "")
        do_operation = self.translate_to_do_operation(
            task_question, demographics, context_qas, 
            include_demographics, include_context, temperature, debug, prolific_id
        )
        
        # Step 3: Run CBN inference with do() operation
        cbn_state = self.run_cbn_inference(do_operation, debug, prolific_id)
        
        # Step 4: Select answer based on CBN state
        response = self.select_answer(vqa, cbn_state, temperature, debug)
        
        if debug:
            print(f"\n=== CBN Agent: Final Result ===")
            print(f"Final Answer: {response}")
            print(f"{'='*80}")
        
        return response
