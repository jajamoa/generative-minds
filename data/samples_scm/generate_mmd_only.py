#!/usr/bin/env python3
"""
Script to generate Mermaid diagrams directly from agent data.
This script reads agents.json and creates MMD files for visualizing agent reasoning patterns.
"""

import os
import json
import re
import random
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

class MermaidDiagramGenerator:
    def __init__(self, 
                 output_dir: str = "data/samples_scm", 
                 model_provider: str = "openai",
                 api_key: str = None,
                 max_workers: int = 3):
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
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                agents = json.load(f)
            return agents
        except Exception as e:
            print(f"Error loading agents from {input_file}: {e}")
            return []

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
                max_tokens=max_tokens,
                temperature=0.0  # Set to 0 for deterministic output
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling model: {e}")
            return ""

    def ensure_complete_mermaid(self, mermaid_code: str) -> str:
        """Ensure the Mermaid diagram is complete and valid
        
        Args:
            mermaid_code: Mermaid diagram code
            
        Returns:
            Fixed Mermaid diagram code
        """
        # Extract content between ```mermaid and ```
        content_match = re.search(r'```mermaid\s*(.*?)\s*```', mermaid_code, re.DOTALL)
        if not content_match:
            return mermaid_code
            
        content = content_match.group(1)
        
        # Extract edges and linkStyles
        lines = content.split('\n')
        edge_lines = []
        linkstyle_lines = []
        node_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if '-->' in line or '--x' in line:
                edge_lines.append(line)
            elif line.startswith('linkStyle'):
                linkstyle_lines.append(line)
            elif '[' in line and ']' in line:
                node_lines.append(line)
                
        # Fix missing linkStyles
        if len(edge_lines) != len(linkstyle_lines):
            new_content = []
            for line in lines:
                line = line.strip()
                if not line or not line.startswith('linkStyle'):
                    new_content.append(line)
                    
            # Add linkStyle declarations for each edge
            for i, edge in enumerate(edge_lines):
                is_positive = '-->' in edge
                if is_positive:
                    new_content.append(f'linkStyle {i} stroke:#00AA00,stroke-width:2px')
                else:
                    new_content.append(f'linkStyle {i} stroke:#FF0000,stroke-dasharray:3,stroke-width:2px')
                    
            content = '\n'.join(new_content)
        
        return "```mermaid\n" + content + "\n```"

    def generate_mermaid_diagram(self, agent: Dict[str, Any]) -> str:
        """Generate Mermaid diagram for an agent
        
        Args:
            agent: Agent data
            
        Returns:
            Mermaid diagram content
        """
        # Extract agent information
        agent_id = agent.get('id', '')
        role = agent.get('role', '')
        description = agent.get('description', '')
        decision_structure = agent.get('decision_structure', '')
        
        # Extract node count from decision structure
        node_count = 12  # Default based on example
        node_edge_match = re.search(r'(\d+)\s*nodes,\s*(\d+)\s*edges', decision_structure)
        if node_edge_match:
            node_count = int(node_edge_match.group(1))
            
        # Extract key themes from decision structure - with improved error handling
        themes = []
        biases = []
        
        try:
            for line in decision_structure.split('\n'):
                line = line.strip()
                if not line.startswith('-'):
                    continue
                    
                line = line[1:].strip()
                
                # Extract biases - with safer string operations
                if 'bias' in line.lower() or 'cognitive' in line.lower():
                    biases.append(line)
                
                # Extract themes - with safer string operations
                if 'focus on' in line.lower():
                    parts = line.lower().split('focus on')
                    if len(parts) > 1:
                        theme = parts[1].strip()
                        themes.append(theme)
                elif 'emphasis on' in line.lower():
                    parts = line.lower().split('emphasis on')
                    if len(parts) > 1:
                        theme = parts[1].strip()
                        themes.append(theme)
                elif 'values' in line.lower():
                    parts = line.lower().split('values')
                    if len(parts) > 1:
                        theme = parts[1].strip()
                        themes.append(theme)
        except Exception as e:
            print(f"Warning: Error extracting themes/biases: {e}")
            # Continue with empty lists if extraction fails
                
        # Create prompt for diagram generation
        prompt = f"""Create a Mermaid flowchart for the agent described below, representing their thought process about urban density changes (upzoning).

Agent ID: {agent_id}
Role: {role}
Description: {description}
Decision Structure: {decision_structure}

Key Themes: {', '.join(themes) if themes else 'None explicitly stated'}
Cognitive Biases: {', '.join(biases) if biases else 'None explicitly stated'}

MERMAID FORMAT REQUIREMENTS:

1. NODES:
   - Use sequential node IDs: n1, n2, n3, etc.
   - First node (n1): Key urban change concept (e.g., building_height)
   - Final node (last n): Must be named "upzoning_stance"
   - Format each node as: n1[concept_name]
   - Use simple nouns or concepts without increase/decrease modifiers
   - Node names should represent core concepts in decision-making
   - Examples:
     * Good: n1[building_height], n2[traffic], n3[sunlight]
     * Avoid: n1[building_height_increase], n2[traffic_reduction]
   - Create approximately {node_count} nodes total

2. EDGES:
   - Two types of direct relationships between pairs of concepts:

   A) Positive relationship (-->): Direct pairwise relationship
      * Syntax: n1 --> n2
      * Meaning: Looking ONLY at these two concepts:
                When n1 is higher, n2 is higher
                When n1 is lower, n2 is lower
      * Visual: Green solid line
      * Examples: 
        - traffic --> noise
          (ONLY consider traffic & noise: more traffic = more noise)
        - density --> walkability
          (ONLY consider density & walkability: more density = more walkability)

   B) Negative relationship (--x): Direct pairwise relationship
      * Syntax: n1 --x n2
      * Meaning: Looking ONLY at these two concepts:
                When n1 is higher, n2 is lower
                When n1 is lower, n2 is higher
      * Visual: Red dashed line
      * Examples:
        - traffic --x safety
          (ONLY consider traffic & safety: more traffic = less safety)
        - density --x parking
          (ONLY consider density & parking: more density = less parking)

   Rules for edges:
   - Focus ONLY on the direct relationship between the two connected concepts
   - Do not consider indirect effects through other nodes
   - Each relationship must be clear when looking at just those two concepts
   - Every node (except upzoning_stance) must have outgoing edge(s)
   - Every node must connect to upzoning_stance through some path
   - No cycles allowed (must be a DAG)

3. STYLES:
   - Each edge must have exactly one linkStyle declaration
   - Number linkStyles sequentially from 0, matching edge order
   - Positive edges (-->): 
     linkStyle N stroke:#00AA00,stroke-width:2px
   - Negative edges (--x): 
     linkStyle N stroke:#FF0000,stroke-dasharray:3,stroke-width:2px

4. STRUCTURE:
   - First list all nodes (one per line)
   - Then list all connections (one per line) 
   - Then list all linkStyle declarations (one per line)
   - Everything must be enclosed in a flowchart TD block

EXAMPLE (for reference):
```mermaid
flowchart TD
    n1[building_height]
    n2[traffic]
    n3[sunlight]
    n4[community_character]
    n5[air_quality]
    n6[public_space]
    n7[neighborhood_identity]
    n8[health]
    n9[social_interaction]
    n10[cultural_heritage]
    n11[quality_of_life]
    n12[upzoning_stance]
    n1 --> n2
    n1 --x n3
    n1 --x n4
    n2 --x n5
    n3 --> n6
    n4 --> n7
    n5 --x n8
    n6 --> n9
    n7 --> n10
    n8 --x n11
    n9 --> n11
    n10 --> n11
    n11 --x n12
    linkStyle 0 stroke:#00AA00,stroke-width:2px
    linkStyle 1 stroke:#FF0000,stroke-dasharray:3,stroke-width:2px
    linkStyle 2 stroke:#FF0000,stroke-dasharray:3,stroke-width:2px
    linkStyle 3 stroke:#FF0000,stroke-dasharray:3,stroke-width:2px
    linkStyle 4 stroke:#00AA00,stroke-width:2px
    linkStyle 5 stroke:#00AA00,stroke-width:2px
    linkStyle 6 stroke:#FF0000,stroke-dasharray:3,stroke-width:2px
    linkStyle 7 stroke:#00AA00,stroke-width:2px
    linkStyle 8 stroke:#00AA00,stroke-width:2px
    linkStyle 9 stroke:#FF0000,stroke-dasharray:3,stroke-width:2px
    linkStyle 10 stroke:#00AA00,stroke-width:2px
    linkStyle 11 stroke:#00AA00,stroke-width:2px
    linkStyle 12 stroke:#FF0000,stroke-dasharray:3,stroke-width:2px
```

KEY PRINCIPLES (MANDATORY):

1. CONNECTIVITY: The graph must represent a complete reasoning chain where:
   - EVERY node (except upzoning_stance) MUST influence the final decision
   - There MUST be at least one path from EACH node to upzoning_stance
   - This connectivity principle is NON-NEGOTIABLE

2. LINK STYLE COMPLETENESS:
   - EVERY edge declaration MUST have exactly one corresponding linkStyle
   - The number of linkStyle declarations MUST be identical to the number of edges
   - Count carefully and verify this match

3. CAUSAL RELATIONSHIP VALIDITY:
   - Each relationship MUST follow real-world logic and common sense
   - The causal chain should directly reflect the agent's decision_structure
   - If A increases B, there must be a clear, logical reason why
   - If A decreases B, there must be a clear, logical reason why
   - Challenge each relationship: "Does this make sense in reality?"

4. DECISION STRUCTURE ADHERENCE:
   - The graph MUST be built based on the agent's decision_structure
   - Key concerns and priorities from decision_structure should be central nodes
   - Follow the reasoning patterns described in the decision_structure
   - Maintain the complexity level indicated in the decision_structure

VERIFICATION STEPS (perform these checks):
1. Count your edges - ensure equal number of linkStyle declarations
2. Trace a path from each node to upzoning_stance
3. Check that upzoning_stance is the only node without outbound edges
4. Verify each causal relationship follows real-world logic
5. Confirm the graph reflects the agent's decision_structure

Output only the complete Mermaid code, without any explanation or additional text.
"""
        
        # Call the model to generate MMD
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
            
        # Ensure complete mermaid diagram
        response = self.ensure_complete_mermaid(response)
            
        return response

    def process_agent(self, agent: Dict[str, Any], sample_id: int) -> bool:
        """Process a single agent
        
        Args:
            agent: Agent data
            sample_id: Sample ID
            
        Returns:
            Success status
        """
        try:
            print(f"Processing sample {sample_id}: {agent['role']}...")
            
            # Generate MMD content
            mmd_content = self.generate_mermaid_diagram(agent)
            
            if mmd_content:
                # Save to file
                mmd_path = self.output_dir / f"sample_{sample_id}.mmd"
                with open(mmd_path, 'w', encoding='utf-8') as f:
                    f.write(mmd_content)
                print(f"Generated MMD file: {mmd_path}")
                return True
                
            return False
        except Exception as e:
            print(f"Error processing agent {sample_id}: {e}")
            return False

    def process_agents(self, agents: List[Dict[str, Any]], start_id: int = 1, num_samples: int = 3) -> Dict[str, int]:
        """Process multiple agents
        
        Args:
            agents: List of agent data
            start_id: Starting sample ID
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing processing result statistics
        """
        results = {
            "total": min(num_samples, len(agents)),
            "success": 0,
            "failed": 0
        }
        
        # Select agents to process
        selected_agents = agents[:num_samples] if len(agents) > num_samples else agents
        
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_id = {
                executor.submit(self.process_agent, agent, i + start_id): i + start_id
                for i, agent in enumerate(selected_agents)
            }
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(selected_agents), desc="Processing progress"):
                sample_id = future_to_id[future]
                try:
                    success = future.result()
                    if success:
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                except Exception as e:
                    print(f"Error processing sample {sample_id}: {e}")
                    results["failed"] += 1
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Generate Mermaid diagrams from agent descriptions")
    parser.add_argument("--input", default="agents.json", help="Input file path containing agent data")
    parser.add_argument("--output_dir", default="data/samples_scm", help="Output directory")
    parser.add_argument("--provider", choices=["dashscope", "openai"], default="dashscope", help="Model provider to use")
    parser.add_argument("--api_key", help="Model API key (optional, can be read from .env file)")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of parallel worker threads")
    parser.add_argument("--start_id", type=int, default=1, help="Starting sample ID")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MermaidDiagramGenerator(
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
    results = generator.process_agents(agents, start_id=args.start_id, num_samples=args.num_samples)
    
    # Print results
    print("\nProcessing complete!")
    print(f"Total: {results['total']}")
    print(f"Successfully generated MMD: {results['success']}")
    print(f"Failed: {results['failed']}")

if __name__ == "__main__":
    main() 