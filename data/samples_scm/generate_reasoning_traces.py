import os
import json
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Required packages
# pip install openai tqdm python-dotenv

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ReasoningTraceGenerator:
    def __init__(self, 
                 output_dir: str = "data/samples_scm", 
                 model_provider: str = "dashscope",
                 api_key: str = None,
                 max_workers: int = 4):
        """Initialize the generator
        
        Args:
            output_dir: Output directory
            model_provider: Model provider ("dashscope" or "openai")
            api_key: API key
            max_workers: Maximum number of parallel worker threads
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_provider = model_provider
        self.max_workers = max_workers
        
        # Set up API client
        if model_provider == "dashscope":
            # Try to get API key from parameter, then .env, then environment variable
            self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            if not self.api_key:
                raise ValueError("DASHSCOPE_API_KEY not found in parameters, .env file, or environment variables")
                
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif model_provider == "openai":
            # Try to get API key from parameter, then .env, then environment variable
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in parameters, .env file, or environment variables")
                
            self.client = OpenAI(
                api_key=self.api_key,
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

    def load_demographics(self, input_file: str) -> List[Dict[str, Any]]:
        """Load demographics data from input file
        
        Args:
            input_file: Input file path
            
        Returns:
            List of demographics data
        """
        # Process based on input file format
        # Example: Assume input is a JSON array with demographics field
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check data format
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'demographics' in data:
            return [data['demographics']]
        else:
            raise ValueError("Unsupported input format. Please provide a JSON array with demographics field")

    def generate_prompt_for_json(self, demographics: Dict[str, Any], sample_id: int) -> str:
        """Create prompt for generating JSON file
        
        Args:
            demographics: Demographics data
            sample_id: Sample ID
            
        Returns:
            Prompt string
        """
        prompt = f"""Based on the demographic information below, create a detailed and nuanced causal model representing how this person would think about urban density changes (upzoning). Model their cognitive processes realistically, including biases, emotional reasoning, and competing motivations while strictly adhering to the specified schema format.

Demographics:
- Age: {demographics.get('age', 'N/A')}
- Income: {demographics.get('income', 'N/A')}
- Education: {demographics.get('education', 'N/A')}
- Occupation: {demographics.get('occupation', 'N/A')}
- Housing: {demographics.get('housing', 'N/A')}

Consider the following dimensions of human reasoning:
1. **Multiple cognitive layers**:
   - Core values and identity (deep-seated beliefs about property, community, fairness)
   - Prior experiences and personal history related to housing/neighborhoods
   - Factual knowledge and misconceptions about urban planning
   - Emotional responses (fear, hope, anxiety, excitement)
   - Social influences (peer groups, community norms, media consumption)
   - Economic interests (property values, rent concerns, investment opportunities)
   - Short vs. long-term thinking patterns

2. **Competing motivations and tensions**:
   - Self-interest vs. community benefit
   - Economic gain vs. neighborhood character
   - Environmental values vs. convenience preferences
   - Change resistance vs. openness to new opportunities
   - Abstract policy positions vs. personal impact anticipation

3. **Cognitive biases to incorporate**:
   - Status quo bias
   - Loss aversion
   - Confirmation bias
   - Availability heuristic (based on personal anecdotes)
   - Proximity bias (NIMBY/YIMBY tendencies)
   - Tribalism/in-group preferences
   - Temporal discounting
   - Authority bias

4. **Reasoning complexity levels**:
   - First-order effects (direct impacts)
   - Second-order effects (indirect consequences)
   - Feedback loops and cyclic reasoning
   - Uncertainty handling and probability estimation
   - Trade-off analysis and prioritization

Generate a JSON file following this schema EXACTLY:
```
{{
  "agent_id": "sample_{sample_id}",
  "demographics": {{
    "age": {demographics.get('age', 'N/A')},
    "income": "{demographics.get('income', 'N/A')}",
    "education": "{demographics.get('education', 'N/A')}",
    "occupation": "{demographics.get('occupation', 'N/A')}",
    "housing": "{demographics.get('housing', 'N/A')}"
  }},
  "nodes": {{
    "n1": {{
      "label": "node_name",
      "type": "binary|continuous",
      "range": [0.0, 1.0], // if type is continuous
      "values": [true, false], // if type is binary
      "semantic_role": "external_state|internal_affect|behavioral_intention",
      "appearance": {{
        "qa_ids": ["qa_01"],
        "frequency": 1
      }},
      "incoming_edges": [],
      "outgoing_edges": ["e1", "e2"]
    }},
    // Additional nodes...
  }},
  "edges": {{
    "e1": {{
      "from": "n1",
      "to": "n2",
      "function": {{
        "target": "n2",
        "inputs": ["n1"],
        "function_type": "sigmoid|threshold",
        "parameters": {{
          // For sigmoid function:
          "weights": [0.8],
          "bias": -0.2
          // For threshold function:
          "threshold": 0.6,
          "direction": "greater|less|equal"
        }},
        "noise_std": 0.1,
        "support_qas": ["qa_01"],
        "confidence": 0.9
      }},
      "support_qas": ["qa_01"]
    }},
    // Additional edges...
  }},
  "qas": [
    {{
      "qa_id": "qa_01",
      "question": "Ask about urban development impacts",
      "answer": "Detailed response explaining perspective",
      "parsed_belief": {{
        "belief_structure": {{
          "from": "n1",
          "to": "n2",
          "direction": "positive|negative"
        }},
        "belief_strength": {{
          "estimated_probability": 0.8,
          "confidence_rating": 0.9
        }},
        "counterfactual": "If [condition], then [alternative outcome]."
      }}
    }},
    // Additional QAs...
  ]
}}
```

Important guidelines:
1. Create appropriately complex models based on demographics:
   - Adjust complexity based on education level, life experience, and occupational complexity
   - Models should have 4-20 nodes, with more educated individuals having more complex models
   - Balance between complexity and interpretability
   - Always include a final node related to "upzoning_stance" with semantic_role "behavioral_intention"

2. Node types must be strictly limited to the following:
   - Use only "binary" or "continuous" types (no other types allowed)
   - Use only the three semantic_roles: "external_state", "internal_affect", "behavioral_intention"
   - Ensure all required fields are present and properly formatted

3. Edge relationships must follow these rules:
   - Use only "sigmoid" or "threshold" function types
   - For sigmoid function, include weights and bias parameters
   - For threshold function, include threshold and direction parameters

4. Make QA pairs deep and realistic:
   - 5-10 detailed QA pairs showing the person's thought process
   - Use realistic language matched to education level and background
   - Include counterfactuals that illustrate causal reasoning

5. Make connections properly:
   - Check that incoming_edges and outgoing_edges are consistent
   - Create multiple paths to the final stance node
   - Include feedback loops where appropriate

Output only valid JSON without any additional explanation.
"""
        return prompt

    def generate_prompt_for_mmd(self, json_content: Dict[str, Any]) -> str:
        """Create prompt for generating MMD file
        
        Args:
            json_content: JSON content
            
        Returns:
            Prompt string
        """
        # Collect all node and edge information for reference
        nodes = json_content.get('nodes', {})
        edges = json_content.get('edges', {})
        
        prompt = f"""Create a Mermaid flowchart for the JSON data below, strictly following the format I provide.

The JSON represents how a person thinks about urban density changes (upzoning):
agent_id: "{json_content.get('agent_id', '')}"

Here are the nodes:
"""
        # Add node information
        for node_id, node in nodes.items():
            label = node.get('label', '')
            prompt += f"- {node_id}: {label}\n"
        
        prompt += "\nHere are the relationships (edges):\n"
        # Add edge information
        for edge_id, edge in edges.items():
            from_node = edge.get('from', '')
            to_node = edge.get('to', '')
            modifier = edge.get('modifier', 0)
            relation_type = "positive" if modifier > 0 else "negative"
            prompt += f"- {edge_id}: from {from_node} to {to_node}, relationship: {relation_type}\n"
        
        # Add a specific format example
        prompt += """
IMPORTANT: Use EXACTLY this format without any changes:

```mermaid
flowchart TD
    n1[node1_label]
    n2[node2_label]
    n3[node3_label]
    ... (continue for all nodes)
    
    n1 --> n2
    n1 --x n3    (use --x for negative relationships)
    ... (continue for all connections)
    
    linkStyle 0 stroke:#00AA00,stroke-width:2px
    linkStyle 1 stroke:#FF0000,stroke-dasharray:3,stroke-width:2px
    ... (continue for all links, green solid for positive, red dashed for negative)
```

Rules to follow precisely:
1. Define all nodes first, then all connections, then all linkStyles
2. Use exactly 4 spaces for indentation
3. Format nodes as: n1[label_text]
4. For positive relationships: n1 --> n2
5. For negative relationships: n1 --x n2
6. Link styles:
   - Positive: stroke:#00AA00,stroke-width:2px (green solid)
   - Negative: stroke:#FF0000,stroke-dasharray:3,stroke-width:2px (red dashed)
7. Number all linkStyles in sequential order starting from 0

Output only the complete Mermaid code with no explanation.
"""
        return prompt

    def call_large_model(self, prompt: str, max_tokens: int = 4000) -> str:
        """Call the large language model
        
        Args:
            prompt: Prompt text
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Model response
        """
        if self.model_provider == "dashscope":
            model_name = "qwen-max"  # Can be changed to other Qwen models
        else:
            model_name = "gpt-4"
            
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling model: {e}")
            return ""

    def generate_json_file(self, demographics: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
        """Generate JSON file
        
        Args:
            demographics: Demographics data
            sample_id: Sample ID
            
        Returns:
            Generated JSON content
        """
        prompt = self.generate_prompt_for_json(demographics, sample_id)
        response = self.call_large_model(prompt, max_tokens=4000)
        
        # Try to parse JSON response
        try:
            # Handle possible markdown wrapping
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            json_content = json.loads(response)
            return json_content
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON, sample ID {sample_id}: {e}")
            print(f"Original response: {response[:500]}...")
            return None

    def generate_mmd_file(self, json_content: Dict[str, Any]) -> str:
        """Generate MMD file
        
        Args:
            json_content: JSON content
            
        Returns:
            Generated MMD content
        """
        prompt = self.generate_prompt_for_mmd(json_content)
        response = self.call_large_model(prompt, max_tokens=2000)
        
        # Handle possible markdown wrapping
        if "```mermaid" in response:
            response = response.split("```mermaid")[1].split("```")[0].strip()
            response = "```mermaid\n" + response + "\n```"
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
            response = "```mermaid\n" + response + "\n```"
        else:
            response = "```mermaid\n" + response + "\n```"
            
        return response

    def process_demographic(self, demographic: Dict[str, Any], sample_id: int) -> Tuple[bool, bool]:
        """Process a single demographic entry
        
        Args:
            demographic: Demographics data
            sample_id: Sample ID
            
        Returns:
            Tuple of (JSON generation success, MMD generation success)
        """
        print(f"Processing sample {sample_id}...")
        json_success, mmd_success = False, False
        
        # Generate JSON file
        json_content = self.generate_json_file(demographic, sample_id)
        if json_content:
            json_path = self.output_dir / f"sample_{sample_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=4, ensure_ascii=False)
            print(f"Generated JSON file: {json_path}")
            json_success = True
            
            # Generate MMD file
            mmd_content = self.generate_mmd_file(json_content)
            if mmd_content:
                mmd_path = self.output_dir / f"sample_{sample_id}.mmd"
                with open(mmd_path, 'w', encoding='utf-8') as f:
                    f.write(mmd_content)
                print(f"Generated MMD file: {mmd_path}")
                mmd_success = True
        
        return json_success, mmd_success

    def process_all_demographics(self, demographics: List[Dict[str, Any]], start_id: int = 1) -> Dict[str, int]:
        """Process all demographics data
        
        Args:
            demographics: List of demographics data
            start_id: Starting sample ID
            
        Returns:
            Dictionary containing processing result statistics
        """
        results = {
            "total": len(demographics),
            "json_success": 0,
            "mmd_success": 0,
            "failed": 0
        }
        
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_id = {
                executor.submit(self.process_demographic, demographic, i + start_id): i + start_id
                for i, demographic in enumerate(demographics)
            }
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(demographics), desc="Processing progress"):
                sample_id = future_to_id[future]
                try:
                    json_success, mmd_success = future.result()
                    if json_success:
                        results["json_success"] += 1
                    if mmd_success:
                        results["mmd_success"] += 1
                    if not (json_success and mmd_success):
                        results["failed"] += 1
                except Exception as e:
                    print(f"Error processing sample {sample_id}: {e}")
                    results["failed"] += 1
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Batch generate Structural Causal Models (SCMs) and visualizations")
    parser.add_argument("--input", required=True, help="Input file path containing demographics data")
    parser.add_argument("--output_dir", default="data/samples_scm", help="Output directory")
    parser.add_argument("--provider", choices=["dashscope", "openai"], default="dashscope", help="Model provider to use")
    parser.add_argument("--api_key", help="Model API key (optional, can be read from .env file)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel worker threads")
    parser.add_argument("--start_id", type=int, default=1, help="Starting sample ID")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ReasoningTraceGenerator(
        output_dir=args.output_dir,
        model_provider=args.provider,
        api_key=args.api_key,
        max_workers=args.max_workers
    )
    
    # Load data
    demographics = generator.load_demographics(args.input)
    
    # Process all data
    results = generator.process_all_demographics(demographics, start_id=args.start_id)
    
    # Print results
    print("\nProcessing complete!")
    print(f"Total: {results['total']}")
    print(f"Successfully generated JSON: {results['json_success']}")
    print(f"Successfully generated MMD: {results['mmd_success']}")
    print(f"Failed: {results['failed']}")

if __name__ == "__main__":
    main() 