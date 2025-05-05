#!/usr/bin/env python3
"""
Script to parse agent information from README.md to JSON format.
Each agent will have a unique ID that can be used for prompts.
"""

import re
import json
import os
import argparse
from pathlib import Path

def parse_agents_to_json(markdown_content):
    """
    Parse agent information from markdown content to JSON.
    
    Args:
        markdown_content (str): The markdown content from README.md
        
    Returns:
        list: List of agent dictionaries
    """
    # Regular expression to match sample blocks
    sample_pattern = r'### Sample (\d+): ([^\n]+)\n(.*?)(?=### Sample \d+:|$)'
    samples = re.findall(sample_pattern, markdown_content, re.DOTALL)
    
    agents = []
    
    for sample_id, title, content in samples:
        # Split description and decision structure
        parts = content.split('**Decision Structure:**')
        description = parts[0].strip()
        decision_structure = parts[1].strip() if len(parts) > 1 else ""
        
        # Create agent object
        agent = {
            "id": sample_id,
            "role": title,
            "description": description,
            "decision_structure": decision_structure
        }
        
        agents.append(agent)
    
    return agents

def main():
    parser = argparse.ArgumentParser(description='Parse agent information from README.md to JSON')
    parser.add_argument('--input', '-i', default='README.md', help='Input markdown file path (default: README.md)')
    parser.add_argument('--output', '-o', default='agents.json', help='Output JSON file path (default: agents.json)')
    args = parser.parse_args()
    
    # Get the current directory
    current_dir = Path(__file__).parent
    input_file = current_dir / args.input
    output_file = current_dir / args.output
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return
    
    # Read markdown file
    with open(input_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Parse to JSON format
    agents = parse_agents_to_json(markdown_content)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(agents, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully parsed {len(agents)} agents and saved to {output_file}")

if __name__ == "__main__":
    main() 