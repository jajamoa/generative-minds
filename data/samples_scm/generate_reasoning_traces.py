import os
import json
import argparse
import random
import concurrent.futures
import re
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

    def load_agents(self, input_file: str) -> List[Dict[str, Any]]:
        """Load agent data from input file
        
        Args:
            input_file: Input file path
            
        Returns:
            List of agent data
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            agents = json.load(f)
        
        return agents

    def generate_demographics(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demographics data based on agent role
        
        Args:
            agent: Agent data
            
        Returns:
            Demographics dictionary
        """
        # Create reasonable demographics based on agent role
        role = agent["role"]
        description = agent["description"]
        
        # Generate appropriate age
        if "elder" in role.lower() or "retiree" in role.lower():
            age = random.randint(65, 85)
        elif "young" in role.lower() or "student" in role.lower() or "millennial" in role.lower():
            age = random.randint(18, 35)
        elif "child" in role.lower():
            age = random.randint(8, 12)
        elif "adolescent" in role.lower():
            age = random.randint(13, 17)
        else:
            age = random.randint(30, 60)
        
        # Generate appropriate income
        if "low-income" in role.lower() or "tenant" in role.lower() or "homeless" in role.lower():
            income = "$15,000-$40,000"
        elif "professional" in role.lower() or "tech worker" in role.lower() or "architect" in role.lower():
            income = "$80,000-$120,000"
        else:
            income = "$40,000-$80,000"
        
        # Generate appropriate education
        if "professional" in role.lower() or "expert" in role.lower() or "architect" in role.lower():
            education = "master's degree or higher"
        elif "worker" in role.lower() or "artist" in role.lower():
            education = "bachelor's degree"
        else:
            education = "high school graduate"
        
        # Extract occupation from role if possible
        occupation = role.replace("Perspective", "").strip()
        
        # Housing situation
        if "homeowner" in role.lower():
            housing = "homeowner"
        elif "tenant" in role.lower() or "renter" in role.lower():
            housing = "renter"
        elif "homeless" in role.lower():
            housing = "unhoused"
        else:
            housing = "mixed housing situation"
            
        return {
            "age": age,
            "income": income,
            "education": education,
            "occupation": occupation,
            "housing": housing
        }

    def generate_prompt_for_json(self, agent: Dict[str, Any], sample_id: int) -> str:
        """Create prompt for generating JSON file
        
        Args:
            agent: Agent data
            sample_id: Sample ID
            
        Returns:
            Prompt string
        """
        demographics = self.generate_demographics(agent)
        
        # Extract complexity information from decision structure
        decision_structure = agent.get('decision_structure', '')
        node_count = 10  # Default node count
        
        # Try to extract node and edge counts from decision structure
        node_edge_match = re.search(r'(\d+)\s*nodes,\s*(\d+)\s*edges', decision_structure)
        if node_edge_match:
            extracted_node_count = int(node_edge_match.group(1))
            # Use the extracted count, but ensure some variability
            node_count = max(4, min(25, extracted_node_count + random.randint(-2, 2)))
        
        # Analyze the decision structure for key themes
        key_themes = []
        for line in decision_structure.split('\n'):
            if line.strip().startswith('- '):
                theme = line.strip()[2:].split(':')[0] if ':' in line else line.strip()[2:]
                if len(theme) > 5 and not theme.startswith('Complex') and not theme.startswith('Focus on'):
                    key_themes.append(theme)
        
        prompt = f"""Create a causal model representing how a person with the perspective below would think about urban density changes (upzoning). Model their cognitive processes based on their decision structure characteristics, strictly following the schema format.

Agent Role: {agent.get('role', '')}
Agent Description: {agent.get('description', '')}
Decision Structure: {decision_structure}

Demographics:
- Age: {demographics.get('age', 'N/A')}
- Income: {demographics.get('income', 'N/A')}
- Education: {demographics.get('education', 'N/A')}
- Occupation: {demographics.get('occupation', 'N/A')}
- Housing: {demographics.get('housing', 'N/A')}

CRITICAL: YOUR REASONING MUST DIRECTLY REFLECT THE DECISION STRUCTURE DETAILS:
- The decision structure describes exactly how this person thinks - follow it closely
- Pay careful attention to each line in the decision structure, especially biases and priorities
- If the decision structure mentions specific concerns (e.g., "focus on economic impact"), make these central
- Match the complexity level described (e.g., "simple linear path" vs "complex network")
- Implement any mentioned cognitive biases (e.g., "loss aversion") in the causal relationships

CAUSAL GRAPH STRUCTURE REQUIREMENTS:
- START your causal graph with a policy node (e.g., "upzoning_policy" or "building_height_increase")
- END with "upzoning_stance" as the only terminal node
- Create logical paths from policy → impacts → perceptions → stance
- Include direct and indirect effects of the policy change

IMPORTANT: CREATE REALISTIC CAUSAL RELATIONSHIPS:
- Ensure all causal relationships follow real-world logic
- Example of GOOD reasoning chain:
  * Upzoning policy → Building height increases (positive)
  * Building height increases → Traffic congestion increases (positive)
  * Building height increases → Sunlight access decreases (negative)
  * Traffic congestion → Air quality decreases (negative)
  * Sunlight access → Public space quality increases (positive)
  * Air quality → Health impact improves (positive)
  * Health impact → Quality of life improves (positive)
  * Quality of life → Support for upzoning increases (positive)

- Example of POOR reasoning (avoid this):
  * Traffic congestion → More likely to drive (positive) - this is illogical
  * Short commute time → Less likely to use transit (negative) - this is illogical

IMPORTANT MODELING REQUIREMENTS:
1. Node structure:
   - Create approximately {node_count} nodes based on the decision structure
   - Mix binary and continuous node types, with appropriate semantic roles
   - "upzoning_stance" must be the ONLY terminal node (no outgoing edges)
   - All paths must eventually lead to "upzoning_stance"

2. Causal relationships:
   - Use a balanced mix of positive and negative relationships based on real-world logic
   - Include both "sigmoid" and "threshold" functions for edges
   - Create relationships that reflect the agent's biases and priorities from decision structure
   - Implement both direct and indirect paths to upzoning stance

3. QA format:
   - Create 15-20 detailed QAs that support the nodes and edges
   - Make QAs clearly express causal beliefs with specified directions
   - Ensure counterfactuals show alternative reasoning scenarios
   - Cover all major reasoning paths and nodes

UNDERSTANDING EDGE DIRECTIONS:
- A positive relationship means: when source node increases (or becomes true), target node increases (or becomes true)
- A negative relationship means: when source node increases (or becomes true), target node decreases (or becomes false)
- For sigmoid functions, positive weights create positive relationships, negative weights create negative relationships
- For threshold functions, "greater" typically creates positive relationships, "less" typically creates negative relationships

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
          "weights": [0.8], // positive weight = positive relationship, negative weight = negative relationship
          "bias": -0.2
          // For threshold function:
          "threshold": 0.6,
          "direction": "greater|less|equal" // "greater" typically creates positive relationships, "less" negative ones
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
          "direction": "positive|negative" // must match the edge function's effect direction
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

Semantic Roles:
- external_state: Observable or inferred world conditions
- internal_affect: Internal emotional or evaluative states
- behavioral_intention: Actions, intentions, or behavioral choices

Important validation rules:
1. Ensure node and edge consistency (incoming/outgoing edges must match across definitions)
2. Use both positive and negative causal relationships (not all positive)
3. "upzoning_stance" must be the only node with no outgoing edges
4. The graph must be connected with no isolated components
5. All QAs must clearly express beliefs that support edge relationships
6. Edge functions (sigmoid/threshold) must correctly implement the relationship direction

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
            
            # Determine relationship type by checking edge function
            function = edge.get('function', {})
            function_type = function.get('function_type', '')
            
            if function_type == 'sigmoid':
                weights = function.get('parameters', {}).get('weights', [0])
                relation_type = "positive" if weights[0] > 0 else "negative"
            elif function_type == 'threshold':
                direction = function.get('parameters', {}).get('direction', '')
                relation_type = "positive" if direction == "greater" else "negative"
            else:
                relation_type = "positive"  # Default
                
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
            # Increase max tokens for longer responses
            max_tokens = 8000
        else:
            model_name = "gpt-4"
            # Increase max tokens for longer responses
            max_tokens = 8000
            
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

    def generate_json_file(self, agent: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
        """Generate JSON file
        
        Args:
            agent: Agent data
            sample_id: Sample ID
            
        Returns:
            Generated JSON content
        """
        prompt = self.generate_prompt_for_json(agent, sample_id)
        response = self.call_large_model(prompt, max_tokens=8000)
        
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
        response = self.call_large_model(prompt, max_tokens=4000)
        
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

    def process_agent(self, agent: Dict[str, Any], sample_id: int, mmd_only: bool = False) -> Tuple[bool, bool]:
        """Process a single agent
        
        Args:
            agent: Agent data
            sample_id: Sample ID
            mmd_only: If True, only generate MMD file without generating JSON
            
        Returns:
            Tuple of (JSON generation success, MMD generation success)
        """
        print(f"Processing sample {sample_id}: {agent['role']}...")
        json_success, mmd_success = False, False
        
        if mmd_only:
            # Generate simple MMD file directly from agent data
            mmd_content = self.generate_simple_mmd(agent, sample_id)
            if mmd_content:
                mmd_path = self.output_dir / f"sample_{sample_id}.mmd"
                with open(mmd_path, 'w', encoding='utf-8') as f:
                    f.write(mmd_content)
                print(f"Generated MMD file: {mmd_path}")
                mmd_success = True
                json_success = True  # Mark as success even though we didn't generate JSON
            return json_success, mmd_success
        
        # Generate JSON file
        json_content = self.generate_json_file(agent, sample_id)
        if json_content:
            json_path = self.output_dir / f"sample_{sample_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False)
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

    def generate_simple_mmd(self, agent: Dict[str, Any], sample_id: int) -> str:
        """Generate a simple MMD file directly from agent data
        
        Args:
            agent: Agent data
            sample_id: Sample ID
            
        Returns:
            Generated MMD content
        """
        # Extract information from agent role and description
        role = agent.get('role', '')
        description = agent.get('description', '')
        decision_structure = agent.get('decision_structure', '')
        
        # Extract node count if available
        node_count = 10  # Default
        node_edge_match = re.search(r'(\d+)\s*nodes,\s*(\d+)\s*edges', decision_structure)
        if node_edge_match:
            node_count = int(node_edge_match.group(1))
        
        # Create a prompt for directly generating MMD
        prompt = f"""Create a Mermaid flowchart for the agent described below. The flowchart should represent their thought process about urban density changes (upzoning).

Agent Role: {role}
Agent Description: {description}
Decision Structure: {decision_structure}

REQUIREMENTS:
1. Create a flowchart with approximately {node_count} nodes
2. START with a node like "upzoning_policy" or "building_height_increase"
3. END with a node called "upzoning_stance"
4. Include a balanced mix of positive and negative relationships
5. Ensure all causal relationships follow real-world logic
6. Create a chain of reasoning that reflects the agent's perspective

FORMAT: Use this EXACT Mermaid flowchart format:
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

Example of good reasoning chain:
* Upzoning policy → Building height increases (positive)
* Building height increases → Traffic congestion increases (positive)
* Building height increases → Sunlight access decreases (negative)
* Traffic congestion → Air quality decreases (negative)
* Air quality → Health impact improves (positive)
* Health impact → Quality of life improves (positive)
* Quality of life → Support for upzoning increases (positive)

Output only the complete Mermaid code with no explanation.
"""
        
        # Call the model to generate MMD content
        response = self.call_large_model(prompt, max_tokens=4000)
        
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

    def process_agents(self, agents: List[Dict[str, Any]], start_id: int = 1, num_samples: int = 3, mmd_only: bool = False) -> Dict[str, int]:
        """Process agents
        
        Args:
            agents: List of agent data
            start_id: Starting sample ID
            num_samples: Number of samples to generate
            mmd_only: If True, only generate MMD files
            
        Returns:
            Dictionary containing processing result statistics
        """
        results = {
            "total": min(num_samples, len(agents)),
            "json_success": 0,
            "mmd_success": 0,
            "failed": 0
        }
        
        # Select agents to process
        selected_agents = agents[:num_samples] if len(agents) > num_samples else agents
        
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_id = {
                executor.submit(self.process_agent, agent, i + start_id, mmd_only): i + start_id
                for i, agent in enumerate(selected_agents)
            }
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(selected_agents), desc="Processing progress"):
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
    parser = argparse.ArgumentParser(description="Generate reasoning traces based on agent perspectives")
    parser.add_argument("--input", default="agents.json", help="Input file path containing agent data")
    parser.add_argument("--output_dir", default="data/samples_scm", help="Output directory")
    parser.add_argument("--provider", choices=["dashscope", "openai"], default="openai", help="Model provider to use")
    parser.add_argument("--api_key", help="Model API key (optional, can be read from .env file)")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of parallel worker threads")
    parser.add_argument("--start_id", type=int, default=1, help="Starting sample ID")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--mmd_only", action="store_true", help="Generate only MMD files (skip JSON generation)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ReasoningTraceGenerator(
        output_dir=args.output_dir,
        model_provider=args.provider,
        api_key=args.api_key,
        max_workers=args.max_workers
    )
    
    # Get absolute path for input file
    current_dir = Path(__file__).parent
    input_file = current_dir / args.input
    
    # Load data
    agents = generator.load_agents(input_file)
    print(f"Loaded {len(agents)} agents from {input_file}")
    
    # Process agents
    results = generator.process_agents(agents, start_id=args.start_id, num_samples=args.num_samples, mmd_only=args.mmd_only)
    
    # Print results
    print("\nProcessing complete!")
    print(f"Total: {results['total']}")
    if not args.mmd_only:
        print(f"Successfully generated JSON: {results['json_success']}")
    print(f"Successfully generated MMD: {results['mmd_success']}")
    print(f"Failed: {results['failed']}")

if __name__ == "__main__":
    main() 