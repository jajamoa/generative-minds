#!/usr/bin/env python3
"""
Script to convert Mermaid diagrams (.mmd files) to structured JSON format.
This script reads .mmd files and creates corresponding JSON files that conform to the data schema.
"""

import os
import json
import re
import random
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Required packages
# pip install openai tqdm python-dotenv

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MmdToJsonConverter:
    def __init__(self, 
                 input_dir: str = "data/samples_scm", 
                 output_dir: str = "data/samples_scm",
                 model_provider: str = "openai",
                 api_key: str = None,
                 max_workers: int = 3):
        """Initialize the converter
        
        Args:
            input_dir: Input directory containing MMD files
            output_dir: Output directory for JSON files
            model_provider: Model provider ("dashscope" or "openai")
            api_key: API key
            max_workers: Maximum number of parallel worker threads
        """
        self.input_dir = Path(input_dir)
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

    def find_mmd_files(self) -> List[Path]:
        """Find all MMD files in the input directory
        
        Returns:
            List of paths to MMD files
        """
        return list(self.input_dir.glob("*.mmd"))

    def parse_mermaid_diagram(self, mmd_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Parse a Mermaid diagram file to extract nodes and edges
        
        Args:
            mmd_path: Path to the MMD file
            
        Returns:
            Tuple of (nodes, edges) where each is a list of dictionaries
        """
        try:
            with open(mmd_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract the content between ```mermaid and ```
            mermaid_match = re.search(r'```mermaid\s*(.*?)\s*```', content, re.DOTALL)
            if not mermaid_match:
                return [], []
                
            mermaid_content = mermaid_match.group(1)
            
            # Extract node definitions
            nodes = []
            for line in mermaid_content.split('\n'):
                node_match = re.search(r'\s*(n\d+)\[(.*?)\]', line)
                if node_match:
                    node_id = node_match.group(1)
                    node_label = node_match.group(2)
                    nodes.append({"id": node_id, "label": node_label})
            
            # Extract edge definitions
            edges = []
            # Find positive relationships (-->)
            pos_edges = re.findall(r'\s*(n\d+)\s*-->\s*(n\d+)', mermaid_content)
            for from_node, to_node in pos_edges:
                edges.append({"from": from_node, "to": to_node, "type": "positive"})
                
            # Find negative relationships (--x)
            neg_edges = re.findall(r'\s*(n\d+)\s*--x\s*(n\d+)', mermaid_content)
            for from_node, to_node in neg_edges:
                edges.append({"from": from_node, "to": to_node, "type": "negative"})
                
            return nodes, edges
        except Exception as e:
            print(f"Error parsing MMD file {mmd_path}: {e}")
            return [], []

    def generate_demographic_data(self) -> Dict[str, Any]:
        """Generate random demographic data
        
        Returns:
            Dictionary containing demographic data
        """
        # Age distribution
        age_options = [
            random.randint(18, 29),  # Young adult
            random.randint(30, 45),  # Middle-aged
            random.randint(46, 65),  # Older adult
            random.randint(66, 85),  # Senior
        ]
        age = random.choice(age_options)
        
        # Income brackets
        income_options = [
            "< $30,000",
            "$30,000 - $60,000",
            "$60,000 - $100,000",
            "> $100,000"
        ]
        income = random.choice(income_options)
        
        # Education levels
        education_options = [
            "high school",
            "some college",
            "bachelor's degree",
            "master's degree or higher"
        ]
        education = random.choice(education_options)
        
        # Occupations
        occupation_options = [
            "student",
            "professional",
            "service worker",
            "educator",
            "healthcare worker",
            "technician",
            "retired",
            "business owner"
        ]
        occupation = random.choice(occupation_options)
        
        # Housing situations
        housing_options = [
            "renter",
            "homeowner",
            "lives with family",
            "public housing"
        ]
        housing = random.choice(housing_options)
        
        return {
            "age": age,
            "income": income,
            "education": education,
            "occupation": occupation,
            "housing": housing
        }

    def call_large_model(self, prompt: str, max_tokens: int = 4000, timeout: int = 240) -> str:
        """Call the large language model
        
        Args:
            prompt: Prompt text
            max_tokens: Maximum number of tokens to generate
            timeout: Timeout in seconds for API call
            
        Returns:
            Model response
        """
        if self.model_provider == "dashscope":
            model_name = "qwen-max"  # Can be changed to other Qwen models
            # Ensure max_tokens is within the allowed range for dashscope
            if max_tokens > 8192:
                print(f"Warning: max_tokens {max_tokens} exceeds dashscope limit. Setting to 8192.")
                max_tokens = 8192
        else:
            model_name = "gpt-4"
            # For OpenAI, 12000 is typically fine for GPT-4
            
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,  # Set to 0 for deterministic output
                timeout=timeout   # Add timeout to prevent hanging
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling model: {e}")
            return ""

    def generate_json_from_mmd(self, mmd_path: Path) -> Optional[Dict[str, Any]]:
        """Generate JSON data from a Mermaid diagram
        
        Args:
            mmd_path: Path to the MMD file
            
        Returns:
            JSON data or None if error
        """
        try:
            # Extract sample ID from filename
            sample_id = re.search(r'sample_(\d+)', mmd_path.stem)
            if not sample_id:
                print(f"Could not extract sample ID from {mmd_path}")
                return None
                
            agent_id = f"sample_{sample_id.group(1)}"
            
            # Parse the Mermaid diagram
            nodes, edges = self.parse_mermaid_diagram(mmd_path)
            if not nodes or not edges:
                print(f"Failed to parse nodes and edges from {mmd_path}")
                return None
                
            # Generate demographic data
            demographics = self.generate_demographic_data()
            
            # Prepare context for LLM
            mmd_content = mmd_path.read_text()
            
            # Create prompt for generating JSON
            prompt = self.create_json_generation_prompt(agent_id, nodes, edges, demographics, mmd_content)
            
            # Set appropriate max_tokens based on model provider
            max_tokens = 8000 if self.model_provider == "dashscope" else 12000
            
            # Call LLM to generate JSON content - with retry mechanism
            for attempt in range(3):  # Try up to 3 times
                try:
                    print(f"Calling LLM for {mmd_path} (attempt {attempt+1}/3)...")
                    response = self.call_large_model(prompt, max_tokens=max_tokens, timeout=240)
                    
                    if not response:
                        print(f"Empty response from LLM on attempt {attempt+1}, retrying...")
                        continue
                        
                    # Handle possible markdown wrapping
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0].strip()
                    
                    # Try to parse the JSON
                    json_data = json.loads(response)
                    
                    # More flexible validation for edge count
                    if abs(len(json_data.get("edges", {})) - len(edges)) > 2:
                        print(f"Edge count mismatch: {len(json_data.get('edges', {}))} vs {len(edges)}")
                        print(f"Generated JSON failed validation on attempt {attempt+1}, retrying...")
                        continue
                        
                    # Check other validation criteria
                    if self.validate_json_data(json_data, agent_id, nodes, edges, flexible_edge_count=True):
                        return json_data
                    else:
                        print(f"Generated JSON failed validation on attempt {attempt+1}, retrying...")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON response on attempt {attempt+1}: {e}")
                except Exception as e:
                    print(f"Unexpected error on attempt {attempt+1}: {e}")
                
                # Wait before retrying with a progressive backoff
                if attempt < 2:  # Don't wait after the last attempt
                    import time
                    wait_time = (attempt + 1) * 5  # 5s, then 10s
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)  # Progressively longer wait
            
            print(f"All attempts failed for {mmd_path}")
            return None
                
        except Exception as e:
            print(f"Error generating JSON from MMD {mmd_path}: {e}")
            return None

    def create_json_generation_prompt(self, agent_id: str, nodes: List[Dict], edges: List[Dict], 
                                     demographics: Dict[str, Any], mmd_content: str) -> str:
        """Create a prompt for generating JSON from MMD
        
        Args:
            agent_id: Agent ID
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            demographics: Dictionary of demographic data
            mmd_content: Original MMD content
            
        Returns:
            Prompt string
        """
        # Create node and edge descriptions
        node_desc = "\n".join([f"- {node['id']}: {node['label']}" for node in nodes])
        edge_desc = "\n".join([f"- {edge['from']} {'--> (positive)' if edge['type'] == 'positive' else '--x (negative)'} {edge['to']}" 
                              for edge in edges])
        
        # Count binary vs continuous nodes (for balance)
        binary_node_count = len(nodes) // 2
        continuous_node_count = len(nodes) - binary_node_count
        
        prompt = f"""Convert the following Mermaid diagram (which represents how a person thinks about urban density changes) into a detailed JSON file strictly following the specified schema.

ORIGINAL MERMAID DIAGRAM:
{mmd_content}

EXTRACTED ELEMENTS:
Agent ID: {agent_id}

Nodes ({len(nodes)}):
{node_desc}

Edges ({len(edges)}):
{edge_desc}

Demographics:
- Age: {demographics['age']}
- Income: {demographics['income']}
- Education: {demographics['education']}
- Occupation: {demographics['occupation']}
- Housing: {demographics['housing']}

CONVERSION REQUIREMENTS:

1. NODE REPRESENTATION:
   - Convert all {len(nodes)} Mermaid nodes to JSON nodes with detailed attributes
   - Create approximately {binary_node_count} binary nodes and {continuous_node_count} continuous nodes
   - Assign appropriate semantic_roles based on node position in the causal chain
     * Early nodes: mostly external_state
     * Middle nodes: mix of external_state and internal_affect
     * Final nodes (especially upzoning_stance): mostly behavioral_intention
   - Track incoming and outgoing edges correctly

2. EDGE REPRESENTATION:
   - Create a unique edge ID for each connection in the diagram
   - Implement edge functions based on relationship type:
     * Positive edges (-->) should use positive weights in sigmoid functions or "greater" in threshold functions
     * Negative edges (--x) should use negative weights in sigmoid functions or "less" in threshold functions
   - Balance between sigmoid and threshold functions
   - Assign appropriate noise levels and confidence values

3. QA GENERATION:
   - Create 15 detailed QAs that logically support the nodes and edges
   - Each QA must express clear causal beliefs matching the diagram connections
   - Include a mix of questions about policy effects, preferences, and decisions
   - Make sure each edge has at least one supporting QA
   - Create believable human-like Q&A content

REQUIRED JSON SCHEMA:
```
{{
  "agent_id": "string",
  "demographics": {{
    "age": number,
    "income": "string",
    "education": "string",
    "occupation": "string",
    "housing": "string"
  }},
  "nodes": {{
    "n1": {{
      "label": "node_name",
      "type": "binary|continuous",
      "range": [0.0, 1.0], // if type is continuous
      "values": [true, false], // if type is binary
      "semantic_role": "external_state|internal_affect|behavioral_intention",
      "appearance": {{
        "qa_ids": ["qa_01", "qa_02"],
        "frequency": 1
      }},
      "incoming_edges": ["e1"],
      "outgoing_edges": ["e2", "e3"]
    }},
    // Other nodes...
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
          "weights": [0.8], // positive weight = positive relationship, negative weight = negative
          "bias": -0.2
          // OR for threshold function:
          "threshold": 0.6,
          "direction": "greater|less|equal" // "greater" for positive, "less" for negative
        }},
        "noise_std": 0.1,
        "support_qas": ["qa_01"],
        "confidence": 0.9
      }},
      "support_qas": ["qa_01"]
    }},
    // Other edges...
  }},
  "qas": [
    {{
      "qa_id": "qa_01",
      "question": "Ask about a specific causal relation between nodes",
      "answer": "Detailed response explaining belief",
      "parsed_belief": {{
        "belief_structure": {{
          "from": "n1",
          "to": "n2",
          "direction": "positive|negative" // Must match the edge function's effect direction
        }},
        "belief_strength": {{
          "estimated_probability": 0.8,
          "confidence_rating": 0.9
        }},
        "counterfactual": "If [condition], then [alternative outcome]."
      }}
    }},
    // Other QAs...
  ]
}}
```

IMPORTANT VALIDATION:
1. Make sure every node has at least one incoming edge (except starting nodes)
2. Make sure every node has at least one outgoing edge (except terminal nodes like upzoning_stance)
3. Ensure edge directionality in JSON strictly matches the Mermaid diagram
4. Verify that edge functions correctly implement positive/negative relationships
5. Make sure every edge has at least one supporting QA

Output only the complete valid JSON with no explanations or additional text.
"""
        return prompt

    def validate_json_data(self, json_data: Dict[str, Any], agent_id: str, 
                          nodes: List[Dict], edges: List[Dict], flexible_edge_count: bool = False) -> bool:
        """Validate the generated JSON data
        
        Args:
            json_data: Generated JSON data
            agent_id: Expected agent ID
            nodes: Original nodes from MMD
            edges: Original edges from MMD
            flexible_edge_count: Whether to allow small differences in edge count
            
        Returns:
            True if valid, False otherwise
        """
        # Check basic structure
        if "agent_id" not in json_data or json_data["agent_id"] != agent_id:
            print(f"Missing or incorrect agent_id: {json_data.get('agent_id', None)}")
            return False
            
        if "demographics" not in json_data:
            print("Missing demographics")
            return False
            
        if "nodes" not in json_data or "edges" not in json_data or "qas" not in json_data:
            print("Missing nodes, edges, or qas")
            return False
            
        # Check node count
        if len(json_data["nodes"]) != len(nodes):
            print(f"Node count mismatch: {len(json_data['nodes'])} vs {len(nodes)}")
            return False
            
        # Check edge count with flexibility
        if flexible_edge_count:
            edge_diff = abs(len(json_data["edges"]) - len(edges))
            if edge_diff > 2:  # Allow up to 2 edges difference
                print(f"Edge count mismatch too large: {len(json_data['edges'])} vs {len(edges)}")
                return False
        else:
            # Strict check
            if len(json_data["edges"]) != len(edges):
                print(f"Edge count mismatch: {len(json_data['edges'])} vs {len(edges)}")
                return False
            
        # Check QA count (at least one per edge)
        min_qa_count = max(1, len(edges) // 2)  # Need at least half as many QAs as edges
        if len(json_data["qas"]) < min_qa_count:
            print(f"Insufficient QAs: {len(json_data['qas'])} for {len(edges)} edges")
            return False
            
        # Validate node references in edges
        for edge_id, edge in json_data["edges"].items():
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            
            if from_node not in json_data["nodes"] or to_node not in json_data["nodes"]:
                print(f"Edge {edge_id} references non-existent node: {from_node} -> {to_node}")
                return False
        
        return True

    def process_mmd_file(self, mmd_path: Path) -> bool:
        """Process a single MMD file
        
        Args:
            mmd_path: Path to the MMD file
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Processing {mmd_path}...")
        
        # Generate JSON data
        json_data = self.generate_json_from_mmd(mmd_path)
        if not json_data:
            return False
            
        # Write JSON file
        json_path = self.output_dir / f"{mmd_path.stem}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        print(f"Generated JSON file: {json_path}")
        return True

    def process_mmd_files(self, mmd_files: List[Path]) -> Dict[str, int]:
        """Process multiple MMD files
        
        Args:
            mmd_files: List of paths to MMD files
            
        Returns:
            Dictionary containing processing result statistics
        """
        results = {
            "total": len(mmd_files),
            "success": 0,
            "failed": 0,
            "skipped": 0
        }
        
        # Filter out already processed files
        filtered_files = []
        for mmd_path in mmd_files:
            json_path = self.output_dir / f"{mmd_path.stem}.json"
            if json_path.exists():
                # Check if the JSON file is valid
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Basic validation - check if it has required top-level keys
                    if all(key in json_data for key in ["agent_id", "demographics", "nodes", "edges", "qas"]):
                        print(f"Skipping {mmd_path}: Valid JSON already exists")
                        results["skipped"] += 1
                        continue
                except:
                    # If there's any error reading or parsing the JSON, process the file again
                    pass
            
            filtered_files.append(mmd_path)
        
        print(f"Skipped {results['skipped']} already processed files with valid JSON")
        print(f"Processing {len(filtered_files)} remaining files")
        
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_path = {
                executor.submit(self.process_mmd_file, mmd_path): mmd_path
                for mmd_path in filtered_files
            }
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(filtered_files), desc="Processing progress"):
                mmd_path = future_to_path[future]
                try:
                    success = future.result()
                    if success:
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                except Exception as e:
                    print(f"Error processing {mmd_path}: {e}")
                    results["failed"] += 1
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Convert Mermaid diagrams to JSON files")
    parser.add_argument("--input_dir", default="data/samples_scm", help="Input directory containing MMD files")
    parser.add_argument("--output_dir", default="data/samples_scm", help="Output directory for JSON files")
    parser.add_argument("--provider", choices=["dashscope", "openai"], default="dashscope", help="Model provider to use")
    parser.add_argument("--api_key", help="Model API key (optional, can be read from .env file)")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of parallel worker threads")
    parser.add_argument("--file_id", type=int, help="Process only a specific file ID (e.g., 16 for sample_16.mmd)")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = MmdToJsonConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_provider=args.provider,
        api_key=args.api_key,
        max_workers=args.max_workers
    )
    
    # Find MMD files
    mmd_files = converter.find_mmd_files()
    print(f"Found {len(mmd_files)} MMD files in {args.input_dir}")
    
    # Filter for specific file if requested
    if args.file_id is not None:
        specific_file = f"sample_{args.file_id}.mmd"
        mmd_files = [f for f in mmd_files if f.name == specific_file]
        if not mmd_files:
            print(f"No file found with ID {args.file_id}")
            return
        print(f"Processing only file: {mmd_files[0]}")
    
    # Process MMD files
    results = converter.process_mmd_files(mmd_files)
    
    # Print results
    print("\nProcessing complete!")
    print(f"Total: {results['total']}")
    print(f"Successfully generated JSON: {results['success']}")
    print(f"Failed: {results['failed']}")

if __name__ == "__main__":
    main() 